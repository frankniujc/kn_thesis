import json, os
from pathlib import Path
from itertools import permutations

DATA_DIR_PATH = Path(os.path.dirname(os.path.realpath(__file__))) / '../data/'

def load_pararel(rel):
    with open(DATA_DIR_PATH / 'data_all_allbags.json') as open_file:
        pararel_rel_data = json.load(open_file)

    prompt_targets = []
    for data in pararel_rel_data[rel]:
        prompt, target, _ = data[0]
        prompt = prompt.replace(' [MASK].', '')
        prompt = prompt.replace(' [MASK] .', '')
        prompt_targets.append((prompt, target))
    return prompt_targets

def generate_synonym(rel):

    assert rel == 'P101'

    with open(DATA_DIR_PATH / 'data_all_allbags.json') as open_file:
        pararel_rel_data = json.load(open_file)

    with open(DATA_DIR_PATH / 'synonyms.json') as open_file:
        synonyms = json.load(open_file)

    sts = []
    targets = []
    for data in pararel_rel_data[rel]:
        target = data[0][1]

        if target in synonyms:
            source, _ = data[0][0].split(' works in the field')
            sts.append((source, target))
            targets.append(target)

    targets = list(set(targets))
    targets_i = {t:i-1 for i,t in enumerate(targets)}

    all_data = []
    for i, (s, t) in enumerate(sts):
        nt = targets[targets_i[t]]
        p1 = '{} works in the field of'
        p2 = '{} is a'

        entry = {
            'source': s,
            'target': t, # field of work
            'syn_target': synonyms[t], # occupation
            # 'new_source': ns,
            'new_target': nt,
            'new_syn_target': synonyms[nt],

            'source_prompt': '{} works in the field of',
            'syn_prompt': '{} is a famous',
        }

        all_data.append(entry)

    return all_data


def generate_symmetry(rel):

    with open(DATA_DIR_PATH / 'data_all_allbags.json') as open_file:
        pararel_rel_data = json.load(open_file)

    sts = []
    for data in pararel_rel_data[rel]:
        if rel == 'P36':
            source, _ = data[4][0].split("'s capital")
        elif rel == 'P1376':
            source, _ = data[0][0].split(' is the capital of')
        else:
            raise 'Only allow P36 and P1376 for the symmetry experiment.'
        target = data[0][1]
        sts.append((source, target))

    all_data = []

    for i, (s, t) in enumerate(sts):
        ns, nt = sts[i-1]
        if rel == 'P36':
            entry = {
                'source': s,
                'target': t,
                'new_source': ns,
                'new_target': nt,

                'source_prompt': f'The capital of {s} is',
                'sym_prompt_t': f'{t} is the capital of',
                'sym_prompt_nt': f'{nt} is the capital of',
            }

        elif rel == 'P1376':
            entry = {
                'source': s,
                'target': t,
                'new_source': ns,
                'new_target': nt,

                'source_prompt': f'{s} is the capital of',
                'sym_prompt_t': f"The capital of {t} is",
                'sym_prompt_nt': f"The capital of {nt} is",
            }

        all_data.append(entry)

    return all_data


if __name__ == '__main__':
    all_data = generate_symmetry('P1376')