# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torchvision.transforms import v2, InterpolationMode
from torchvision.transforms.v2 import functional as F
from PIL import Image
import math

from torchtune.data import Message
from torchtune.data._prompt_templates import _TemplateType, _get_prompt_template
from torchtune.models.clip._transform import CLIPImageTransform
from torchtune.models.qwen2_5._tokenizer import Qwen2_5Tokenizer
from torchtune.modules.tokenizers import parse_hf_tokenizer_json
from torchtune.modules.transforms import Transform
from torchtune.modules.transforms.tokenizers import ModelTokenizer
from torchtune.models.qwen2_5_vision._vision_utils import smart_resize

logger = logging.getLogger(__name__)

# HuggingFace OPENAI_CLIP constants to match their normalization
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

class Qwen2_5_VLImageTransform:
    """
    This class accepts images of any size and dynamically resizes, normalizes and patches it
    based on the image size constraints and patch size.

    The algorithm will NOT distort the image to fit a certain aspect ratio, because
    that leads to a significant degradation in image quality.

    For example, if an input image is of size 300x800, and we have:
    - patch_size = 14
    - merge_size = 2
    - min_pixels = 3136 (56 * 56)
    - max_pixels = 1003520 (28 * 28 * 1280)

    The image will be:
    1. Resized to fit within min_pixels and max_pixels constraints
    2. Divided into 14x14 patches
    3. Patches will be merged in 2x2 groups
    4. Final output will be a sequence of merged patches

    Args:
        image_mean (Optional[List[float]]): Mean values of each channel, used for normalization.
            Should be the same used for the pre-trained model. If None, uses OPENAI_CLIP_MEAN. Default None.
        image_std (Optional[List[float]]): Standard deviation values of each channel, used for normalization.
            Should be the same used for the pre-trained model. If None, uses OPENAI_CLIP_STD. Default None.
        patch_size (int): Size of the patches to divide the image into. Default 14.
        merge_size (int): Size of the patch merging factor. Default 2.
        temporal_patch_size (int): Size of the temporal patch merging factor. Default 2.
        min_pixels (int): Minimum number of pixels for the shorter edge. Default 3136 (56 * 56).
        max_pixels (int): Maximum number of pixels for the longer edge. Default 1003520 (28 * 28 * 1280).
        size (Optional[Dict[str, int]]): Size configuration with 'shortest_edge' and 'longest_edge' keys.
            If provided, overrides min_pixels and max_pixels. Default None.
        dtype (torch.dtype): Data type of the output image. Default torch.bfloat16.
        resample (str): Resampling method used when resizing images. Supports any enum of
            ``torchvision.transforms.InterpolationMode``, e.g. "nearest", "nearest_exact", "bilinear", "bicubic".
            Default 'bicubic'.

    Examples:
        >>> image_transform = Qwen2_5_VLImageTransform(
        ...    image_mean=None,
        ...    image_std=None,
        ...    patch_size=14,
        ...    merge_size=2,
        ...    temporal_patch_size=2,
        ...    min_pixels=3136,
        ...    max_pixels=1003520,
        ...    resample="bilinear",
        ...)
        >>> # create random image
        >>> image = (np.random.rand(100,200,3) * 255).astype(np.uint8)
        >>> image = PIL.Image.fromarray(image)
        >>> output = image_transform({"image": image})
        >>> print(output["pixel_values"].shape)  # [num_patches, channels * temporal_patch_size * patch_size * patch_size]
        >>> print(output["image_grid_thw"])  # [grid_t, grid_h, grid_w]
    """

    def __init__(
        self,
        *,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
        size: Optional[Dict[str, int]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
        resample: str = "bicubic",
    ) -> None:
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size
        
        # Handle size configuration - prioritize size dict over individual params
        if size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
            self.size = size.copy()
        else:
            self.size = {"shortest_edge": 56 * 56, "longest_edge": 12845056}

        # Override with individual parameters if provided
        if min_pixels is not None:
            self.size["shortest_edge"] = min_pixels
        if max_pixels is not None:
            self.size["longest_edge"] = max_pixels

        self.min_pixels = self.size["shortest_edge"]
        self.max_pixels = self.size["longest_edge"]

        self.dtype = dtype
        self.resample = getattr(InterpolationMode, resample.upper())
        
        # Use OPENAI_CLIP defaults if not provided (matches HuggingFace)
        self.mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.std = image_std if image_std is not None else OPENAI_CLIP_STD

    def __call__(
        self, sample: Mapping[str, Any], inference: bool = False
    ) -> Mapping[str, Any]:
        """
        Apply image decoding and transformations to the "image" field in the sample.

        Args:
            sample (Mapping[str, Any]): A sample with an "image" field containing
                a PIL Image or torch.Tensor
            inference (bool): Whether the template is being used for inference or not.

        Returns:
            Mapping[str, Any]: The sample with updated fields:
                - "pixel_values": Flattened patches tensor
                - "image_grid_thw": Grid dimensions (temporal, height, width)
                - "num_patches": Number of patches calculated
        """
        image = sample["image"]
        assert isinstance(
            image, (Image.Image, torch.Tensor)
        ), "Input image must be a PIL image or a torch.Tensor."

        # Convert to RGB and tensor
        if isinstance(image, Image.Image) and image.mode != "RGB":
            image = image.convert("RGB")
        image = F.to_image(image)
        
        # Convert to float and rescale to [0, 1] - this matches HF's rescaling step
        image = F.to_dtype(image, dtype=torch.float32, scale=True)

        # Get image dimensions
        height, width = image.shape[-2:]

        # Calculate resize dimensions
        resized_height, resized_width = smart_resize(
            height, 
            width,
            factor=self.patch_size * self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels
        )

        # Resize image
        image = F.resize(
            image,
            size=(resized_height, resized_width),
            interpolation=self.resample
        )

        # Normalize with OPENAI_CLIP values 
        image = F.normalize(image, mean=self.mean, std=self.std)

        image = image.to(dtype=self.dtype)

        patches = image.unsqueeze(0)
        
        if patches.shape[0] % self.temporal_patch_size != 0:
            repeats_needed = self.temporal_patch_size - (patches.shape[0] % self.temporal_patch_size)
            last_frame = patches[-1:].repeat(repeats_needed, 1, 1, 1)
            patches = torch.cat([patches, last_frame], dim=0)

        # Calculate grid dimensions
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        channels = patches.shape[1]

        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channels,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )

        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channels * self.temporal_patch_size * self.patch_size * self.patch_size
        )

        num_patches = grid_h * grid_w 
        num_image_tokens = num_patches // self.merge_size**2

        sample.update({
            "pixel_values": flatten_patches,
            "image_grid_thw": torch.tensor([[grid_t, grid_h, grid_w]]),
            "num_image_tokens": num_image_tokens,
        })

        return sample

class Qwen2_5_VLTransform(ModelTokenizer, Transform):
    # TODO: update docstring
    """
    Transform for Qwen 2.5 Vision model that handles both text tokenization and image processing.

    Args:
        path (str): Path to the tokenizer vocab.json file.
        merges_file (str): Path to the tokenizer merges.txt file.
        patch_size (int): Size of the patches used in vision processing. Default 14.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file
            structured similarly. Default is None to use the canonical Qwen 2.5 special tokens.
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        image_mean (Optional[List[float]]): Mean values of each channel, used for normalization.
            Default None to use OPENAI_CLIP_MEAN.
        image_std (Optional[List[float]]): Standard deviations for each channel, used for normalization.
            Default None to use OPENAI_CLIP_STD.
        dtype (torch.dtype): Data type of transformed image. Default torch.bfloat16.
        prompt_template (Optional[_TemplateType]): template used to format the messages based on their role.

    Examples:
        >>> model_transform = Qwen25VisionTransform("/path/to/tokenizer.model", tile_size=224, patch_size=14)
        >>> transformed_data = model_transform({"messages": user_message, "images": [img1, img2]})
        >>> print(transformed_data["tokens"])
        [1, 31587, 29644, 102, 2]
        >>> print(transformed_data["images"][0].shape)
        torch.Size([4, 3, 224, 224])
    """

    def __init__(
        self,
        path: str,
        merges_file: str,
        *,
        patch_size: int = 14,
        special_tokens_path: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        dtype: torch.dtype = torch.bfloat16,
        prompt_template: Optional[_TemplateType] = None,
    ):
        special_tokens = (
            parse_hf_tokenizer_json(special_tokens_path)
            if special_tokens_path is not None
            else None
        )
        template = (
            _get_prompt_template(prompt_template)
            if prompt_template is not None
            else None
        )
        self.tokenizer = Qwen2_5Tokenizer(
            path=path,
            merges_file=merges_file,
            special_tokens=special_tokens,
            max_seq_len=max_seq_len,
            prompt_template=template,
        )
        
        # Initialize the Qwen2.5 VL image transform
        self.image_transform = Qwen2_5_VLImageTransform(
            image_mean=image_mean,
            image_std=image_std,
            patch_size=patch_size,
            merge_size=2,  # Default merge size for Qwen2.5-VL
            temporal_patch_size=2,  # Default temporal patch size
            dtype=dtype,
            resample="bicubic",
        )

        self.stop_tokens = self.tokenizer.stop_tokens
        self.special_tokens = self.tokenizer.special_tokens
        self.max_seq_len = max_seq_len
        self.patch_size = patch_size
        self.prompt_template = prompt_template
        self.pad_id = self.tokenizer.pad_id

    @property
    def base_vocab_size(self) -> int:
        return len(self.tokenizer.encoder)

    @property
    def vocab_size(self) -> int:
        # Total vocab size includes base vocab + special tokens
        return len(self.tokenizer.encoder) + len(self.tokenizer.special_tokens)

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """
        Encode a string into a list of token ids.

        Args:
            text (str): The string to encode.
            add_bos (bool): Whether to add the tokenizer's bos_id. Default is True.
            add_eos (bool): Whether to add the tokenizer's eos_id. Default is True.

        Returns:
            List[int]: The list of token ids.
        """
        return self.tokenizer.encode(text=text, add_bos=add_bos, add_eos=add_eos)

    def decode(
        self,
        token_ids: List[int],
        truncate_at_eos: bool = True,
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode a list of token ids into a string.

        Args:
            token_ids (List[int]): The list of token ids.
            truncate_at_eos (bool): Whether to truncate the string at the end of
                sequence token. Default is True.
            skip_special_tokens (bool): Whether to show or skip special tokens in the decoded string.
                Default is True.

        Returns:
            str: The decoded string.
        """
        # Handle truncate_at_eos manually since Qwen2_5Tokenizer doesn't support it
        if truncate_at_eos and self.tokenizer.eos_id in token_ids:
            eos_index = token_ids.index(self.tokenizer.eos_id)
            token_ids = token_ids[:eos_index]
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def transform_image(
        self, image: Image.Image, inference: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Transform an image into flattened patches for the vision encoder.
        This method applies the transformations defined in `Qwen2_5_VLImageTransform`.

        Args:
            image (Image.Image): The input image.
            inference (bool): Whether to run in inference mode. This is passed to the
                underlying image transform. Default is False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]: A tuple containing:
                - The transformed image patches as a tensor.
                - The image grid dimensions (t, h, w) as a tensor.
                - The number of patches calculated.
        """
        sample = {"image": image}
        transformed = self.image_transform(sample, inference=inference)
        return transformed["pixel_values"], transformed["image_grid_thw"], transformed["num_image_tokens"]

    def tokenize_message(
        self,
        message: Message,
        *,
        add_start_tokens: bool = True,
        add_end_tokens: bool = True,
    ) -> List[int]:
        """
        Tokenize a single message into a list of token ids.

        Args:
            message (Message): The message to tokenize.
            add_start_tokens (bool): Whether to add the tokenizer's bos_id. Default True.
            add_end_tokens (bool): Whether to add the tokenizer's eos_id. Default True.

        Returns:
            List[int]: The list of token ids.
        """
        return self.tokenizer.tokenize_message(
            message=message,
            add_start_tokens=add_start_tokens,
            add_end_tokens=add_end_tokens,
        )

    def tokenize_messages(
        self,
        messages: List[Message],
        *,
        add_end_tokens: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        """
        Tokenize a list of messages into a list of token ids and masks.

        Args:
            messages (List[Message]): The list of messages to tokenize.
            add_end_tokens (bool): Whether to add the tokenizer's eos_id. Default True.

        Returns:
            Tuple[List[int], List[bool]]: The list of token ids and the list of masks.
        """
        return self.tokenizer.tokenize_messages(
            messages=messages,
            add_end_tokens=add_end_tokens,
        )

    def __call__(
        self, sample: Mapping[str, Any], inference: bool = False
    ) -> Mapping[str, Any]:
        """
        Apply image decoding, transformations and tokenization to messages in the sample.

        Args:
            sample (Mapping[str, Any]): A sample with a "messages" field.
            inference (bool): Whether to run in inference mode. Default is False.

        Returns:
            Mapping[str, Any]: The transformed sample with the following fields:
                - tokens: List[int] of tokenized messages
                - mask: List[bool] of masks for the tokenized messages
                - encoder_input: Dict[str, Any] of transformed images
        """
        encoder_input = {"image": {"hidden_states": [], "grid_thw": []}}
        messages = sample["messages"]
        for message in messages:
            for content in message.content:
                if content["type"] == "image":
                    image = content["content"]
                    
                    pixel_values, image_grid_thw, num_image_tokens = self.transform_image(image, inference=inference)
                    
                    content["num_image_tokens"] = num_image_tokens
                    
                    encoder_input["image"]["hidden_states"].append(pixel_values)
                    encoder_input["image"]["grid_thw"].append(image_grid_thw)
                    
        encoder_input["image"]["hidden_states"] = torch.stack(encoder_input["image"]["hidden_states"], dim=0)
        encoder_input["image"]["grid_thw"] = torch.cat(encoder_input["image"]["grid_thw"], dim=0)

        sample["encoder_input"] = encoder_input
        sample = self.tokenizer(sample, inference=inference)
        return sample
