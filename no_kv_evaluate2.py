#exec(open("/home/ubuntu/vanshaj/torchtune/no_kv_evaluate2.py").read())

from tkinter import N


def kv_evaluate(dataloader, model, batch, tokens_to_generate = 10):
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
    print("Prediction: ", ''.join(predictions), '\n', accepted_tokens_list)
    return

def no_kv_evaluate(dataloader, model, batch, tokens_to_generate = 10):
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
    return

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
    return
