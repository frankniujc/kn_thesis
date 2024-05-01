import argparse
import logging

from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kn_code.rome_hparams import ROMEHyperParams
from kn_code.evaluate import test_prediction_acc
from kn_code.data import load_pararel, generate_symmetry, generate_synonym
from kn_code.rome import apply_rome_to_model
from kn_code.util.nethook import get_parameter


def main(model_name, relation):

    print('Start Loading Model...')

    hparams = ROMEHyperParams.from_hparams(f'./hparams/ROME/{model_name}')
    model = AutoModelForCausalLM.from_pretrained(hparams.model_name).to("cuda")
    tok = AutoTokenizer.from_pretrained(hparams.model_name)
    tok.pad_token = tok.eos_token

    print(model)


    if relation in ['P36', 'P1367']:
        get_data = generate_symmetry
        acc_lsts = [[], [], [], [], [], [], []]
    elif relation in ['P101']:
        get_data = generate_synonym
        acc_lsts = [[], [], [], [], [], []]
    else:
        raise ValueError

    for i, entry in enumerate(tqdm(get_data(relation))):

        request = [{
            'prompt': entry['source_prompt'],
            'target_new': entry['new_target'],
            'ground_truth': entry['target'],
            'subject': entry['source'],
        }]

        model_edited, orig_weights = apply_rome_to_model(model, tok, request, hparams,
            return_orig_weights=True, keep_original_weight=True)

        if relation in ['P36', 'P1367']:

            acc_lsts[0] += test_prediction_acc(model_edited, tok, hparams,
                entry['source_prompt'],
                entry['target'], 0)

            acc_lsts[2] += test_prediction_acc(model_edited, tok, hparams,
                entry['source_prompt'],
                entry['new_target'], 0)

            acc_lsts[3] += test_prediction_acc(model_edited, tok, hparams,
                entry['sym_prompt_t'],
                entry['new_source'], 0)

            acc_lsts[4] += test_prediction_acc(model_edited, tok, hparams,
                entry['sym_prompt_nt'],
                entry['source'], 0)

            acc_lsts[5] += test_prediction_acc(model_edited, tok, hparams,
                entry['sym_prompt_t'],
                entry['source'], 0)

            acc_lsts[6] += test_prediction_acc(model_edited, tok, hparams,
                entry['sym_prompt_nt'],
                entry['new_source'], 0)

            for name, weight in orig_weights.items():
                w = get_parameter(model, name)
                w[...] = weight

            acc_lsts[1] += test_prediction_acc(model, tok, hparams,
                entry['source_prompt'],
                entry['target'], 0)

        elif relation in ['P101']:


            acc_lsts[0] += test_prediction_acc(model_edited, tok, hparams,
                entry['source_prompt'].format(entry['source']),
                entry['target'], 0)

            acc_lsts[1] += test_prediction_acc(model_edited, tok, hparams,
                entry['source_prompt'].format(entry['source']),
                entry['new_target'], 0)

            acc_lsts[2] += test_prediction_acc(model_edited, tok, hparams,
                entry['syn_prompt'].format(entry['source']),
                entry['syn_target'], 0)

            acc_lsts[3] += test_prediction_acc(model_edited, tok, hparams,
                entry['syn_prompt'].format(entry['source']),
                entry['new_syn_target'], 0)

            for name, weight in orig_weights.items():
                w = get_parameter(model, name)
                w[...] = weight

            acc_lsts[4] += test_prediction_acc(model, tok, hparams,
                entry['source_prompt'].format(entry['source']),
                entry['target'], 0)

            acc_lsts[5] += test_prediction_acc(model, tok, hparams,
                entry['syn_prompt'].format(entry['source']),
                entry['syn_target'], 0)

        print(i, [acc[-1] for acc in acc_lsts])

    for i, acc in enumerate(acc_lsts):
        print(i, sum(acc), len(acc), sum(acc)/len(acc))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', choices=['llama-2-7b', 'gpt2-xl'])
    parser.add_argument('relation', choices=['P36', 'P1376', 'P101'])

    args = parser.parse_args()

    main(args.model_name, args.relation)