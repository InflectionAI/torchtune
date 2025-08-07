#exec(open("/home/ubuntu/vanshaj/torchtune/no_kv_evaluate2.py").read())

def create_causal_mask2(
    batch_size: int,
    current_seq_len: int,  # This is (1+n) where n is number of medusa heads
    cached_seq_len: int,   # Current KV cache length
    max_cache_size,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Create a causal mask for Medusa evaluation with KV cache.
    
    Args:
        batch_size: Number of sequences in batch (typically 1 for Medusa)
        current_seq_len: Length of new tokens being processed (1 + medusa_heads)
        cached_seq_len: Length of tokens already in KV cache
        device: Device to create tensor on
        dtype: Data type for the mask
    
    Returns:
        Causal mask of shape [batch_size, current_seq_len, total_seq_len]
    """
    total_seq_len = cached_seq_len + current_seq_len
    
    # Create a lower triangular mask for the full sequence
    mask = torch.tril(
        torch.ones(
            current_seq_len, 
            total_seq_len, 
            device=device, 
            dtype=dtype
        )
    )
    suffix_mask_dim = max_cache_size - mask.shape[-1]
    
    mask_suffix = torch.zeros((current_seq_len , suffix_mask_dim), device = device, dtype = dtype)
    full_mask = torch.cat((mask, mask_suffix), dim = -1) 
    # Expand to batch dimension
    full_mask = full_mask.unsqueeze(0)  # [1, current_seq_len, total_seq_len]
    # full_mask = full_mask.expand(batch_size, current_seq_len, max_cache_size)

    return full_mask


def create_causal_mask(
    batch_size: int,
    current_seq_len: int,  # This is (1+n) where n is number of medusa heads
    cached_seq_len: int,   # Current KV cache length
    max_cache_size,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Create a causal mask for Medusa evaluation with KV cache.
    
    Args:
        batch_size: Number of sequences in batch (typically 1 for Medusa)
        current_seq_len: Length of new tokens being processed (1 + medusa_heads)
        cached_seq_len: Length of tokens already in KV cache
        device: Device to create tensor on
        dtype: Data type for the mask
    
    Returns:
        Causal mask of shape [batch_size, current_seq_len, total_seq_len]
    """
    total_seq_len = cached_seq_len + current_seq_len
    old_tokens_mask = torch.ones(current_seq_len, cached_seq_len, device=device, dtype=dtype)
    # Create a lower triangular mask for the full sequence
    new_tokens_triangular_mask = torch.tril(
        torch.ones(
            current_seq_len, 
            current_seq_len, 
            device=device, 
            dtype=dtype
        )
    )
    suffix_mask_dim = max_cache_size - total_seq_len
    mask_suffix = torch.zeros((current_seq_len , suffix_mask_dim), device = device, dtype = dtype)
    # mask = torch.cat((mask, mask_suffix), dim = -1) 
    mask = torch.cat((old_tokens_mask, new_tokens_triangular_mask, mask_suffix), dim = -1)

    # Expand to batch dimension
    mask = mask.unsqueeze(0)  # [1, current_seq_len, total_seq_len]
    # full_mask = full_mask.expand(batch_size, current_seq_len, max_cache_size)
    # print("mask:\n", mask)
    # print("bool mask:\n", mask.bool())
    mask = mask.bool()

    curr_mask = torch.cat((old_tokens_mask, new_tokens_triangular_mask), dim = -1)
    default_mask = torch.ones(current_seq_len, total_seq_len, dtype=torch.bool, device = device).tril(diagonal=0)
    print("full_mask:\n", curr_mask, curr_mask.shape)
    print("temp_mask:\n", default_mask, default_mask.shape)
    print("Mask Equality:", (curr_mask == default_mask).all().item())
    # assert False
    return default_mask.unsqueeze(0)
    return mask

def no_kv_evaluate(dataloader, model, batch, tokens_to_generate = 100):
    predictions = []
    accepted_tokens_list = []
    input_prompt = batch
    # input_tokens = batch['tokens'].to(device)
    # input_prompt = format_input(input_tokens)
    print('input_prompt:', decode(input_prompt))
    
    bs = input_prompt.shape[0]
    accepted_preds = input_prompt  # Initialize with full prompt
    tokens_generated = 0

    while tokens_generated < tokens_to_generate:
        torch.cuda.empty_cache() 
        with disable_kv_cache(model):
            output = model(accepted_preds)  # pred[0] = base logits, shape: [bs, seq, vocab]
        base_logits = output[0][:, -1, :]  # last token logits

        next_token = base_logits.argmax(dim=-1, keepdim=True)  # [bs, 1]

        # Append base prediction to input
        accepted_preds = torch.cat([accepted_preds, next_token], dim=-1)
        accepted_tokens_list.append(next_token.item())

        # Decode and store
        decoded = decode(next_token)
        predictions.extend(decoded)
        
        tokens_generated += 1

    print("Prediction: ", ''.join(predictions))
    print("Accepted token IDs:", accepted_tokens_list)
    return ''.join(predictions)

def kv_evaluate(dataloader, model, batch, tokens_to_generate = 100):
    predictions = []
    accepted_tokens_list = []

    model.reset_caches()
    model_dtype = next(model.parameters()).dtype
    
    # input_tokens = batch['tokens'].to(device)
    # input_prompt = format_input(input_tokens)
    input_prompt = batch
    print("input_prompt:", input_prompt)
    print('input_prompt:', decode(input_prompt))
    
    bs = input_prompt.shape[0]
    curr_seq_len = input_prompt.shape[1] 
    curr_kv_len = 0

    # Initial forward pass with full prompt
    # print('curr_kv_len:', curr_kv_len)
    causal_mask = create_causal_mask(bs, curr_seq_len, curr_kv_len, max_cache_size, device, model_dtype)
    # causal_mask = create_causal_mask(bs, curr_seq_len, max_cache_size, device)
    
    # input_pos = torch.arange(curr_seq_len, device=device).unsqueeze(0)
    input_pos = torch.arange(curr_seq_len, device=device).unsqueeze(0).expand(bs, -1)

    output = model(input_prompt, mask=causal_mask, input_pos=input_pos)
    base_logits = output[0][:, -1]  # shape: [bs, vocab_dim]
    next_token = base_logits.argmax(dim=-1, keepdim=True)  # shape: [bs, 1]
    
    accepted_tokens_list.append(next_token.item())
    decoded_prediction = decode(next_token)
    predictions.extend(decoded_prediction)
    
    for m in model.modules():
        if isinstance(m, TransformerSelfAttentionLayer):
            layer = m
            break
        else:
            layer = None
    curr_kv_len = curr_seq_len
    model_kv_len = layer.attn.kv_cache.size
    print("curr_kv_len, model_kv_len:", curr_kv_len, model_kv_len)
    assert (curr_kv_len == int(model_kv_len))
    tokens_generated = 1
    curr_seq_len = 1
    while tokens_generated < tokens_to_generate:
        # Update KV cache length
        
        model_kv_len = layer.attn.kv_cache.size
        print("curr_kv_len, model_kv_len:", curr_kv_len, model_kv_len)
        assert (curr_kv_len == int(model_kv_len))
        # print('curr_kv_len:', curr_kv_len)
        
        # Create mask and positions for single new token
        causal_mask = create_causal_mask(bs, curr_seq_len, curr_kv_len, max_cache_size, device, model_dtype)
        input_pos = torch.arange(curr_kv_len - 1, curr_kv_len, device=device).unsqueeze(0)
        
        # Forward pass with just the new token
        output = model(next_token, mask=causal_mask, input_pos=input_pos)
        base_logits = output[0][:, -1]  # shape: [bs, vocab_dim]
        next_token = base_logits.argmax(dim=-1, keepdim=True)  # shape: [bs, 1]
        curr_kv_len += 1
        accepted_tokens_list.append(next_token.item())
        decoded_prediction = decode(next_token)
        predictions.extend(decoded_prediction)
        
        tokens_generated += 1
    print("-----------------------------------------------------")
    print("Prediction: ", ''.join(predictions))
    print("-----------------------------------------------------")

    print("Accepted token IDs:", accepted_tokens_list)
    return ''.join(predictions)

def kv_evaluate2(dataloader, model, batch, tokens_to_generate = 100):
    predictions = []
    # initialize kv cache

    model.reset_caches()
    model_dtype = next(model.parameters()).dtype
    # empty kv cache
    input_tokens = batch['tokens'].to(device)
    input_prompt = format_input(input_tokens) # bs, seq
    # DEBUG
    input_prompt = input_prompt#[:, :4] 

    print('input_prompt:', decode(input_prompt))
    accepted_tokens_list = []
    bs = input_prompt.shape[0]; curr_seq_len = input_prompt.shape[1] 
    curr_kv_len = 0

    causal_mask = create_causal_mask(bs, curr_seq_len, curr_kv_len, max_cache_size, device, model_dtype)
    input_pos = torch.arange(curr_seq_len, device = device).unsqueeze(0)
    # print("model input: ", input_prompt)

    output = model(input_prompt, mask = causal_mask, input_pos = input_pos) # shape: [(1+n), bs, seq, vocab_dim]

    base_logits = output[0][:, -1] # shape: [bs, vocab_dim]
    pred = base_logits.argmax(dim = -1) # shape: [bs, 1]

    tokens_generated = 1
    preds = pred.unsqueeze(-1) # shape: [bs, 1+n]
    accept_lengths = []
    pass_idx = 0
    curr_kv_len = curr_seq_len
    accepted_tokens_list.append(pred.item())
    while(tokens_generated<tokens_to_generate):
        pass_idx += 1
        #now take all of the previous outputs and put them into the model as a batch
        curr_seq_len = preds.shape[1] 
        causal_mask = create_causal_mask(bs, curr_seq_len, curr_kv_len, max_cache_size, device, model_dtype)
        # shape: [bs, curr_seq, self.encoder_max_cache_seq_len], boolean mask with True representing queries to attend
        input_pos = torch.arange(curr_kv_len, curr_kv_len + curr_seq_len, device=device).unsqueeze(0)
        # All True rect mask of new_tokens x tokens_generated | upper_triangular mask of new_tokens x new_tokens + False rect mask of new_tokens x (encoder_max_cache_seq_len - (tokens_generated + new_tokens))
        # print("model input: ", preds)
        pred = model(preds, mask = causal_mask, input_pos = input_pos) # shape: [(1+n), bs, (1+n), vocab_dim]

        
        base_logits = pred[0] # shape: [bs, (1+n), vocab_dim]
        
        base_out = base_logits.argmax(dim = -1) # shape: [bs, (1+n)]
        
        
        # compare base_out with preds to see which medusa_heads in the prev inference were correct:
        mask = (base_out[:, :-1] == preds[:, 1:])
        correct_pred_mask = mask.cumprod(dim = -1)
        last_accepted_head = correct_pred_mask.sum().item()

        # accept_len denotes the last head that was correct. If the last head was correct then when it is inputted back into the model, the output will also be relevant (with the base_out also being correct). Therefore the base_out is taken as an accepted token and the medusa_out is taken as the input for the next pass.
        curr_kv_len += (last_accepted_head+1)

        # reset kv cache to curr_kv_len
        model.revert_cache_to_valid_length(curr_kv_len)
        tokens_generated += (last_accepted_head+1)
        

        # what should be the input for the next pass? The last medusa pred that was correct. Take it's output as the input for the next pass.
        
        
        preds = base_out[:, last_accepted_head: last_accepted_head + 1]
        accept_lengths.append((last_accepted_head+1))
        
        # Extract the accepted tokens for decoding
        accepted_tokens = base_out[0, :last_accepted_head+1]  # shape: [last_accepted_head+1]
        accepted_tokens_list.append(accepted_tokens.item())
        decoded_prediction = decode(accepted_tokens) 
        predictions.extend(decoded_prediction)
        # tokenizer.decode(accepted_tokens.flatten().tolist(), skip_special_tokens=False)
        # print(f"Prediction {pass_idx}: ", decoded_prediction)

        # preds is the new input for the next pass
    
    print("accept_lengths: ", accept_lengths)
    print("-----------------------------------------------------")
    print("Prediction: ", ''.join(predictions), '\n', accepted_tokens_list)
    print("-----------------------------------------------------")

    return

def no_kv_evaluate3(dataloader, model, batch, tokens_to_generate = 100):
    predictions = []
    
# initialize kv cache

    # model.reset_caches()
    # model_dtype = next(model.parameters()).dtype
    # empty kv cache
    input_tokens = batch['tokens'].to(device)
    input_prompt = format_input(input_tokens) # shape: [bs, seq]
    # DEBUG
    input_prompt = input_prompt
    n = model.medusa_num_heads
    print('input_prompt:', decode(input_prompt))
    accepted_preds_list = []
    bs = input_prompt.shape[0]; curr_seq_len = input_prompt.shape[1] 
    curr_kv_len = 0


    output = model(input_prompt) # shape: [(1+n), bs, seq, vocab_dim]
    base_logits = output[0][:, -1] # shape: [bs, vocab_dim]
    pred = base_logits.argmax(dim = -1) # shape: [bs, 1]
    preds = pred.unsqueeze(-1) # shape: [bs, 1+n]
    accept_lengths = []
    pass_idx = 0
    curr_kv_len = curr_seq_len
    accepted_preds = torch.cat((input_prompt, preds[:, 0:1]), dim = -1)
    newly_accepted_token = preds[:, 0:1]
    preds = accepted_preds
    accepted_preds_list.append(preds[:, 0:1].item())
    tokens_generated = 1
    print('First Token:', decode(newly_accepted_token))
    print("Rest:")
    while(tokens_generated<tokens_to_generate):
        # assert (accepted_preds == preds[:, :-(n)]).all()
        pass_idx += 1
        #now take all of the previous outputs and put them into the model as a batch
        curr_seq_len = preds.shape[1] 
        # print("model input: ", preds.shape) # shape: [bs, seq + (1+n)]
        pred = model(preds) # shape: [(1+n), bs, seq + (1+n), vocab_dim]
        tokens_generated += 1
        
        base_logits = pred[0][:, -1:, :] # shape: [bs, seq + (1+n), vocab_dim] -> [bs, (1+n), vocab_dim]
        base_out = base_logits.argmax(dim = -1) # shape: [bs, (1+n)]
        curr_kv_len += 1
        newly_accepted_tokens = base_out
        
        accepted_preds = torch.cat((accepted_preds, newly_accepted_tokens), dim = -1)
        preds = accepted_preds
        accepted_preds_list.append(newly_accepted_tokens.item())
        # Extract the accepted tokens for decoding
        
        decoded_prediction = decode(newly_accepted_tokens) 
        predictions.extend(decoded_prediction)
        # tokenizer.decode(accepted_tokens.flatten().tolist(), skip_special_tokens=False)
        # print(f"Prediction {pass_idx}: ", decoded_prediction)

        # preds is the new input for the next pass
        
    print("accept_lengths: ", accept_lengths)
    print("Prediction: ", ''.join(predictions), '\n', accepted_preds_list)
    return ''.join(predictions)

def no_kv_evaluate2(dataloader, model, batch, tokens_to_generate=10):
    predictions = []

    input_tokens = batch['tokens'].to(device)
    input_prompt = format_input(input_tokens)  # shape: [bs, seq]
    print('input_prompt:', decode(input_prompt))

    bs = input_prompt.shape[0]
    accepted_preds = input_prompt  # Initialize input sequence
    accepted_preds_list = []
    
    tokens_generated = 0

    while tokens_generated < tokens_to_generate:
        output = model(accepted_preds)  # pred[0] = base logits, shape: [bs, seq, vocab]
        base_logits = output[0][:, -1, :]  # last token logits
        next_token = base_logits.argmax(dim=-1, keepdim=True)  # [bs, 1]

        # Append base prediction to input
        accepted_preds = torch.cat([accepted_preds, next_token], dim=-1)
        accepted_preds_list.append(next_token.item())

        # Decode and store
        decoded = decode(next_token)
        predictions.extend(decoded)
        print(f"Generated token {tokens_generated + 1}: {decoded}")

        tokens_generated += 1

    print("Final prediction:", ''.join(predictions))
    print("Accepted token IDs:", accepted_preds_list)
    return ''.join(predictions)


    # pred_kv = kv_evaluate(dataloader, kv_model, batch)
    # pred_no_kv = no_kv_evaluate(dataloader, no_kv_model, batch)
    # print("pred_kv == pred_no_kv:", pred_kv == pred_no_kv)
def no_kv_evaluate4(dataloader, model, batch, tokens_to_generate = 100):
    predictions = []
    accepted_tokens_list = []

    input_tokens = batch['tokens'].to(device)
    input_prompt = format_input(input_tokens)
    print('input_prompt:', decode(input_prompt))
    
    bs = input_prompt.shape[0]
    accepted_preds = input_prompt  # Initialize with full prompt
    tokens_generated = 0

    while tokens_generated < tokens_to_generate:
        with disable_kv_cache(model):
            output = model(accepted_preds)  # pred[0] = base logits, shape: [bs, seq, vocab]
        base_logits = output[0][:, -1, :]  # last token logits
        next_token = base_logits.argmax(dim=-1, keepdim=True)  # [bs, 1]

        # Append base prediction to input
        accepted_preds = torch.cat([accepted_preds, next_token], dim=-1)
        accepted_tokens_list.append(next_token.item())

        # Decode and store
        decoded = decode(next_token)
        predictions.extend(decoded)
        
        tokens_generated += 1

    print("Prediction: ", ''.join(predictions))
    print("Accepted token IDs:", accepted_tokens_list)
    return ''.join(predictions)