def evaluate(dataloader, model, max_cache_size, device, tokens_to_generate = 4):
    
    # initialize kv cache
    for batch in dataloader:
        model.reset_caches()
        model_dtype = next(model.parameters()).dtype
        # empty kv cache
        input_tokens = batch['tokens'].to(device)
        input_prompt = format_input(input_tokens) # bs, seq
        # DEBUG
        input_prompt = input_prompt[:, :4] 

        print('input_prompt:', decode(input_prompt))

        bs = input_prompt.shape[0]; curr_seq_len = input_prompt.shape[1] 
        curr_kv_len = 0

        causal_mask = create_causal_mask(bs, curr_seq_len, curr_kv_len, max_cache_size, device, model_dtype)
        input_pos = torch.arange(curr_seq_len, device = device).unsqueeze(0)
        print("model input: ", input_prompt)

        output = model(input_prompt, mask = causal_mask, input_pos = input_pos) # shape: [(1+n), bs, seq, vocab_dim]

        base_logits = output[0][:, -1] # shape: [bs, vocab_dim]
        pred = base_logits.argmax(dim = -1) # shape: [bs, 1]

        medusa_logits = torch.stack(output[1:])[:, :, -1] # shape: [n, bs, vocab_dim]
        medusa_out = medusa_logits.argmax(dim = -1) # shape: [n, bs]
        medusa_out = medusa_out.permute((1,0)) # shape: [bs, n]
        tokens_generated = 1
        preds = torch.cat((pred.unsqueeze(-1), medusa_out), dim = -1) # shape: [bs, 1+n]
        accept_lengths = []
        pass_idx = 0
        curr_kv_len = curr_seq_len

        while(tokens_generated<tokens_to_generate):
            pass_idx += 1
            #now take all of the previous outputs and put them into the model as a batch
            curr_seq_len = preds.shape[1] 
            causal_mask = create_causal_mask(bs, curr_seq_len, curr_kv_len, max_cache_size, device, model_dtype)
            # shape: [bs, curr_seq, self.encoder_max_cache_seq_len], boolean mask with True representing queries to attend
            input_pos = torch.arange(curr_kv_len, curr_kv_len + curr_seq_len, device=device).unsqueeze(0)
            # All True rect mask of new_tokens x tokens_generated | upper_triangular mask of new_tokens x new_tokens + False rect mask of new_tokens x (encoder_max_cache_seq_len - (tokens_generated + new_tokens))
            print("model input: ", preds)
            pred = model(preds, mask = causal_mask, input_pos = input_pos) # shape: [(1+n), bs, (1+n), vocab_dim]

            
            base_logits = pred[0] # shape: [bs, (1+n), vocab_dim]
            medusa_logits = torch.stack(pred[1:]) # shape: [n, bs, (1+n), vocab_dim]
            base_out = base_logits.argmax(dim = -1) # shape: [bs, (1+n)]
            medusa_out = medusa_logits.argmax(dim = -1) # shape: [n, bs, (1+n)]
            
            # compare base_out with preds to see which medusa_heads in the prev inference were correct:
            mask = (base_out[:, :-1] == preds[:, 1:])
            correct_pred_mask = mask.cumprod(dim = -1)
            last_accepted_head = correct_pred_mask.sum().item()

            # accept_len denotes the last head that was correct. If the last head was correct then when it is inputted back into the model, the output will also be relevant (with the base_out also being correct). Therefore the base_out is taken as an accepted token and the medusa_out is taken as the input for the next pass.
            curr_kv_len += (last_accepted_head)
            # reset kv cache to curr_kv_len
            model.revert_cache_to_valid_length(curr_kv_len)
            tokens_generated += (last_accepted_head+1)
            

            # what should be the input for the next pass? The last medusa pred that was correct. Take it's output as the input for the next pass.
            accepted_head_medusa_pred = medusa_out[:, :, last_accepted_head] # shape: [n, bs]
            accepted_head_medusa_pred = accepted_head_medusa_pred.transpose(0, 1)
            # breakpoint()
            preds = torch.cat((base_out[:, last_accepted_head: last_accepted_head + 1], accepted_head_medusa_pred), dim = -1)
            accept_lengths.append((last_accepted_head+1))
            
            # Extract the accepted tokens for decoding
            accepted_tokens = base_out[0, :last_accepted_head+1]  # shape: [last_accepted_head+1]
            print(f"Prediction {pass_idx}: ", tokenizer.decode(accepted_tokens.flatten().tolist(), skip_special_tokens=False))
            # preds is the new input for the next pass
            
        print("accept_lengths: ", accept_lengths)
        return