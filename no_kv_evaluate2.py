def run():
    kv_model, no_kv_model = model
    diff_list = []
    # kv_model = model
    model_dtype = next(kv_model.parameters()).dtype
    i = 0
    for batch in dataloader:
    #     batch = torch.tensor([[128000, 128006,   9125, 128007,    271,    791,   3823,    374,   2663,
    #   21077,    323,    279,    892,    374,   7418,     11,   6250,    220,
    #    1114,    220,   2366,     18,     11,    220,   2839,     25,   1114,
    #    8971,  42084, 128009, 128006,    882, 128007,    271,   3923,    656,
    #     499,   1781,    922,  36142,   2191,     30, 128009, 128006,  78191,
    #  128007]], device='cuda:1')
        # batch = batch[:, :1]
        tokens_generated = 0
        accepted_preds = None
        next_token_kv = None
        while(tokens_generated<tokens_to_generate):
            # print("-------no_kv_evaluate--------:")
            # pred_no_kv = no_kv_evaluate(dataloader, no_kv_model, batch, tokens_to_generate)
            base_logits_no_kv, next_token_no_kv, accepted_preds = no_kv_evaluate(dataloader, no_kv_model, batch, tokens_to_generate, accepted_preds)
            # print("------kv_evaluate--------:")
            
            # pred_kv = kv_evaluate(dataloader, kv_model, batch, tokens_to_generate)
            base_logits_kv, next_token_kv = kv_evaluate(dataloader, kv_model, batch, tokens_to_generate, next_token_kv)
            tokens_generated += 1
            
            if next_token_no_kv != next_token_kv:
                p_log = F.log_softmax(base_logits_no_kv, dim=-1)
                q_log = F.log_softmax(base_logits_kv, dim=-1)
                kl = torch.sum(torch.exp(p_log) * (p_log - q_log))  # KL(p || q)
                # print("pred_kv, pred_no_kv, kl norm:", kl)
                breakpoint()
            else:
                print("Same!")