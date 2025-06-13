# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Mapping, Optional

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.data._messages import validate_messages

from torchtune.modules.transforms import Transform

from typing import Any, Callable, Dict, Optional, Union

from torchtune.data._messages import OpenAIToMessages, ShareGPTToMessages
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer

class MedusaDataset(Dataset):
    """
    Primary class for creating any dataset for supervised fine-tuning either from
    Hugging Face Hub, local files, or remote files. This class supports instruct,
    chat, tool, or multimodal data for fine-tuning. At a high level, this class
    will load the data from source and apply the following pre-processing steps
    when a sample is retrieved:

    1. Dataset-specific transform. This is typically unique to each dataset and extracts
       the necessary columns into torchtune's :class:`~torchtune.data.Message` format,
       a standardized API for all model tokenizers.
    2. Model-specific transform or tokenization with optional prompt template


    All datasets are formatted into a list of :class:`~torchtune.data.Message`
    because for fine-tuning, datasets can be considered as "conversations" with the model,
    or AI assistant. Thus, we can standardize all text content as messages in a conversation assigned to
    a role:

    - ``"system"`` messages contain the system prompt
    - ``"user"`` messages contain the input prompt into the model
    - ``"assistant"`` messages are the response of the model and what you actually want
      to train for and compute loss directly against
    - ``"ipython"`` messages are the return from a tool call

    Chat datasets are multiple rounds of user-assistant messages. Instruct datasets
    are typically a single round involving a specific instruction and the model's response.
    Tool datasets are a type of chat dataset that includes ipython messages. Multimodal
    datasets are a type of chat dataset that incorporates media into the user messages.

    The :class:`~torchtune.data.Message` forms the core data unit that all tokenizer
    APIs expect. The key component of this class that ensures any dataset is transformed
    into this format is the ``message_transform``. This is a callable class that takes
    in a sample dictionary - typically a single row from the source dataset - that
    processes the sample in any configurable way to output a list of messages::

        [
            Message(
                role=<system|user|assistant|ipython>,
                content=<message>,
            ),
            ...
        ]

    For any custom dataset, use the ``message_transform`` to contain all pre-processing to
    return the list of messages.

    Any model-specific pre-processing that needs to happen can be configured with the ``model_transform``
    parameter. This is another callable class that contains any custom logic tied to the
    model you are fine-tuning and will carry over to inference. For example, text + image
    multimodal datasets requires processing the images in a way specific to the vision
    encoder being used by the model and is agnostic to the specific dataset.

    Tokenization is handled by the ``model_transform``. All
    :class:`~torchtune.modules.transforms.tokenizers.ModelTokenizer` can be treated as
    a ``model_transform`` since it uses the model-specific tokenizer to transform the
    list of messages outputted from the ``message_transform`` into tokens used by the
    model for training. Text-only datasets will simply pass the
    :class:`~torchtune.modules.transforms.tokenizers.ModelTokenizer` into ``model_transform``.
    Tokenizers handle prompt templating, if configured.

    Args:
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details.
        message_transform (Transform): callable that keys into the desired fields in the sample
            and converts text content to a list of :class:`~torchtune.data.Message`. It is expected that the final list
            of messages are stored in the ``"messages"`` key. See :ref:`message_transform_usage_label` for details.
        model_transform (Transform): callable that applies model-specific pre-processing to the sample after the list of
            messages is created from ``message_transform``. This includes tokenization and any modality-specific
            transforms. It is expected to return at minimum ``"tokens"`` and ``"mask"`` keys.
        num_medusa_heads (int): Number of Medusa heads for the model. Default is 4. This determines how many
            sets of future token labels are created (k-th next token predictions).
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        filter_kwargs (Optional[dict[str, Any]]): additional keyword arguments to pass to ``filter_fn``.
        **load_dataset_kwargs (dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.
    """

    def __init__(
        self,
        *,
        source: str,
        message_transform: Transform,
        model_transform: Transform,
        num_medusa_heads: int = 4,
        filter_fn: Optional[Callable] = None,
        filter_kwargs: Optional[dict[str, Any]] = None,
        **load_dataset_kwargs: dict[str, Any],
    ) -> None:
        self._message_transform = message_transform
        self._model_transform = model_transform

        self._data = load_dataset(source, **load_dataset_kwargs)
        if filter_fn is not None:
            if filter_kwargs is None:
                filter_kwargs = {}
            self._data = self._data.filter(filter_fn, **filter_kwargs)

        self._prepare_sample = MedusaTransform(
            message_transform=self._message_transform,
            model_transform=self._model_transform,
            num_medusa_heads=num_medusa_heads,
        )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)


class MedusaTransform(Transform):
    def __init__(
        self,
        message_transform: Optional[Transform] = None,
        model_transform: Optional[Transform] = None,
        num_medusa_heads: int = 4,
    ):
        if message_transform is None and model_transform is None:
            raise ValueError(
                "At least one of message_transform or model_transform must be provided."
            )
        self._message_transform = message_transform
        self._model_transform = model_transform
        self.num_medusa_heads = num_medusa_heads

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        if self._message_transform is not None:
            transformed_sample = self._message_transform(sample)
            if "messages" in transformed_sample:
                validate_messages(transformed_sample["messages"])
        else:
            transformed_sample = sample

        if self._model_transform is not None:
            tokenized_dict = self._model_transform(transformed_sample)

            if not ("tokens" in tokenized_dict and "mask" in tokenized_dict):
                keys_str = ", ".join(tokenized_dict.keys())
                error_message = (
                    "model_transform returned the following keys: "
                    f"{keys_str}. Must return 'tokens' and 'mask' as keys."
                )
                raise ValueError(error_message)

            # Create labels for the main LM head (next token prediction)
            # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
            # Shift labels to be off by 1 from the logits.
            tokenized_dict["labels"] = list(
                np.where(
                    tokenized_dict["mask"][1:],
                    CROSS_ENTROPY_IGNORE_IDX,
                    tokenized_dict["tokens"][1:],
                )
            )
            tokenized_dict["labels"].append(CROSS_ENTROPY_IGNORE_IDX)
            
            # Create labels for each Medusa head (k-th next token predictions)
            medusa_labels = []
            tokens = tokenized_dict["tokens"]
            mask = tokenized_dict["mask"]
            seq_len = len(tokens)
            
            for k in range(1, self.num_medusa_heads + 1):
                # For k-th Medusa head, predict k steps ahead
                if k + 1 < seq_len:
                    # Shift by k+1 positions for k-th next token prediction
                    labels_k = list(
                        np.where(
                            mask[k+1:],
                            CROSS_ENTROPY_IGNORE_IDX,
                            tokens[k+1:],
                        )
                    )
                    # Pad with CROSS_ENTROPY_IGNORE_IDX to match sequence length
                    labels_k.extend([CROSS_ENTROPY_IGNORE_IDX] * (k + 1))
                else:
                    # If sequence is too short, fill with CROSS_ENTROPY_IGNORE_IDX
                    labels_k = [CROSS_ENTROPY_IGNORE_IDX] * seq_len
                
                medusa_labels.append(labels_k)
            
            # Store medusa labels
            tokenized_dict["medusa_labels"] = medusa_labels
            
            assert len(tokenized_dict["tokens"]) == len(tokenized_dict["labels"])
            for i, medusa_label in enumerate(medusa_labels):
                assert len(tokenized_dict["tokens"]) == len(medusa_label), \
                    f"Medusa head {i+1} labels length mismatch: {len(tokenized_dict['tokens'])} vs {len(medusa_label)}"
        else:
            tokenized_dict = transformed_sample

        return tokenized_dict



def medusa_chat_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    conversation_column: str,
    conversation_style: str,
    num_medusa_heads: int = 4,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[MedusaDataset, PackedDataset]:
    """
    Configure a custom dataset with conversations between user and model assistant.

    This builder function can be used to configure a custom chat dataset directly from the yaml config
    as an alternative to :class:`~torchtune.datasets.SFTDataset`, as it is made to be config friendly.

    The dataset is expected to contain a single column with the conversations:

    .. code-block:: text

        |  conversations                         |
        |----------------------------------------|
        | [{"role": "user", "content": Q1},      |
        |  {"role": "assistant", "content": A1}] |

    This will be converted to:

    .. code-block:: python

        messages = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ]

    This list of messages is then tokenized for model training.

    You may have a different structure for your conversations, such as different role names or
    different keys in the json structure. You can use the ``conversation_style`` parameter
    to choose from standard formats such as "sharegpt" (see :class:`~torchtune.data.ShareGPTToMessages`)
    or "openai" (see :class:`~torchtune.data.OpenAIToMessages`). If your dataset is not in one of these
    formats, we recommend creating a custom message transform and using it in a custom dataset
    builder function similar to :class:`~torchtune.datasets.chat_dataset`.

    If your column names are different, use the ``conversation_column`` parameter to point
    towards the column with the conversations.

    Masking of the prompt during training is controlled by the ``train_on_input`` flag, which is
    set to ``False`` by default.

    - If ``train_on_input`` is True, the prompt is used during training and
      contributes to the loss.
    - If ``train_on_input`` is False, the prompt is masked out (tokens replaced with -100).

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text"), pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details.
        conversation_column (str): name of column containing the conversations.
        conversation_style (str): string specifying expected style of conversations in the dataset
            for automatic conversion to the :class:`~torchtune.data.Message` structure.
            Supported styles are: "sharegpt", "openai"
        num_medusa_heads (int): Number of Medusa heads for the model. Default is 4.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Default is None.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``,
            such as ``data_files`` or ``split``.

    Examples:

    ::

        my_dataset.json
        [
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": "What time is it in London?",
                    },
                    {
                        "from": "gpt",
                        "value": "It is 10:00 AM in London.",
                    },
                ],
            },
            {
                "conversations": [
                    ...
                ],
            },
            ...,
        ]

    ::

        >>> from torchtune.datasets import chat_dataset
        >>> dataset = chat_dataset(
        ...     tokenizer=tokenizer,
        ...     source="json",
        ...     data_files="my_dataset.json",
        ...     conversation_column="conversations",
        ...     conversation_style="sharegpt",
        ...     train_on_input=False,
        ...     packed=False,
        ...     split="train",
        ... )
        >>> tokens = dataset[0]["tokens"]
        >>> tokenizer.decode(tokens)
        "What time is it in London?It is 10:00 AM in London."

    This can also be accomplished via the yaml config:

    .. code-block:: yaml

        dataset:
          _component_: torchtune.datasets.chat_dataset
          source: json
          data_files: my_dataset.json
          conversation_column: conversations
          conversation_style: sharegpt
          train_on_input: False
          packed: False
          split: train

    Returns:
        Union[SFTDataset, PackedDataset]: the configured :class:`~torchtune.datasets.SFTDataset`
            or :class:`~torchtune.datasets.PackedDataset` if ``packed=True``

    Raises:
        ValueError: if the conversation format is not supported
    """
    if conversation_style == "sharegpt":
        message_transform = ShareGPTToMessages(
            train_on_input=train_on_input,
            column_map={"conversations": conversation_column},
            new_system_prompt=new_system_prompt,
        )
    elif conversation_style == "openai":
        message_transform = OpenAIToMessages(
            train_on_input=train_on_input,
            column_map={"messages": conversation_column},
            new_system_prompt=new_system_prompt,
        )
    else:
        raise ValueError(f"Unsupported conversation style: {conversation_style}")

    ds = MedusaDataset(
        source=source,
        message_transform=message_transform,
        model_transform=tokenizer,
        num_medusa_heads=num_medusa_heads,
        split=split,
        filter_fn=filter_fn,
        **load_dataset_kwargs,
    )
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds
