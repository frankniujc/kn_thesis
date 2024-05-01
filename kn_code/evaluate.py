import torch
import numpy as np


def slice_list(matrix, start_indices, left):
    if isinstance(matrix[0], list):
        if left:
            return [row[start_index-1:-1] for row, start_index in zip(matrix, start_indices)]
        else:
            return [row[start_index:] for row, start_index in zip(matrix, start_indices)]
    else:
        if left:
            return matrix[start_indices[0]-1:-1]
        else:
            return matrix[start_indices[0]:]

def test_prediction_acc(model, tok, hparams, prompts, targets, device, locality=False):
    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        max_length=max(hparams.max_length, max_prompt_len),
        return_tensors="pt",
    ).to(f"cuda:{device}")
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max(hparams.max_length, max_prompt_len),
        return_tensors="pt",
    )
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
    prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
    with torch.no_grad():
        outputs = model(**prompt_target_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
        answers = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        labels = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        answers = slice_list(answers,prompt_len,left=True)
        labels = slice_list(labels,prompt_len,left=False)
        if locality:
            return answers if type(answers[0]) is list else [answers,]
        if isinstance(answers[0], list):
            res = []
            for ans,label in zip(answers,labels):
                temp_acc = np.mean(np.equal(ans, label))
                if np.isnan(temp_acc):
                    continue
                res.append(temp_acc)
            return res
        else:
            return [np.mean(np.equal(answers, labels))]
