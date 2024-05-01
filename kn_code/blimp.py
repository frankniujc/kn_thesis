from datasets import load_dataset

def detok(func):
    def wrapper(p, t, get_item=False):
        if p.startswith('Ġ'):
            p = p[1:]
        if t.startswith('Ġ'):
            t = t[1:]
        return func(p, t, get_item=get_item)
    return wrapper

@detok
def determiner_noun_agreement_2_func(p, t, get_item=False):

    if get_item:
        item = p.split('[MASK] ')[1].replace(' .', '')
        item = (item,)
    else:
        item = tuple()

    if t in ['this', 'that']:
        return ('sg', t) + item
    elif t in ['these', 'those']:
        return ('pl', t) + item
    else:
        raise ValueError

@detok
def anaphor_gender_agreement_func(p, t, get_item=False):
    if t == 'herself':
        return ('f', t)
    elif t == 'himself':
        return ('m', t)
    elif t == 'itself':
        return ('i', t)
    else:
        raise ValueError

def subject_verb_agreement_1_func(p, t, get_item=False):
    if t == 'hasn':
        return ('sg', t)
    elif t == 'haven':
        return ('pl', t)
    elif t.endswith('s'):
        return ('sg', t)
    else:
        return ('pl', t)


class BLiMPDataProcessing:

    TARGET_FUNC = {
        'anaphor_gender_agreement': anaphor_gender_agreement_func,
        'determiner_noun_agreement_2': determiner_noun_agreement_2_func,
        'regular_plural_subject_verb_agreement_1': subject_verb_agreement_1_func,
    }


    def _mask_difference_mlm(self, row):
        if row['sentence_good'] == row['sentence_bad']:
            return None, [], []

        tokens_good = self.tokenizer.tokenize(row['sentence_good'])
        tokens_bad  = self.tokenizer.tokenize(row['sentence_bad'])
        masked_tokens = []

        for good, bad in zip(tokens_good, tokens_bad):
            if good == bad:
                masked_tokens.append(good)
            else:
                masked_tokens.append('[MASK]')
                break

        start_idx = len(masked_tokens)
        for end_idx, (good, bad) in enumerate(zip(tokens_good[::-1], tokens_bad[::-1])):
            if good == bad:
                masked_tokens.insert(start_idx, good)
            else:
                break

        diff_good = tokens_good[start_idx-1:len(tokens_good)-end_idx]
        diff_bad = tokens_bad[start_idx-1:len(tokens_bad)-end_idx]

        masked_sent = self.tokenizer.convert_tokens_to_string(masked_tokens)
        tgt_pos = masked_tokens.index('[MASK]') + 1 # +1 for [CLS]

        return masked_sent, diff_good, diff_bad

    def _mask_difference_decoder(self, row):
        if row['sentence_good'] == row['sentence_bad']:
            return None, [], []

        good_tokens = self.tokenizer.tokenize(row['sentence_good'])
        bad_tokens = self.tokenizer.tokenize(row['sentence_bad'])
        masked_tokens = []

        if row['sentence_good'] == row['sentence_bad']:
            return None, [], []

        good_tokens = self.tokenizer.tokenize(row['sentence_good'])
        bad_tokens = self.tokenizer.tokenize(row['sentence_bad'])
        masked_tokens = []

        for good, bad in zip(good_tokens, bad_tokens):
            if good == bad:
                masked_tokens.append(good)
            else:
                break

        start_idx = len(masked_tokens) + 1
        for end_idx, (good, bad) in enumerate(zip(good_tokens[::-1], bad_tokens[::-1])):
            if good != bad:
                break

        diff_good = good_tokens[start_idx-1:len(good_tokens)-end_idx]
        diff_bad = bad_tokens[start_idx-1:len(bad_tokens)-end_idx]

        masked_sent = self.tokenizer.convert_tokens_to_string(masked_tokens)

        return masked_sent, diff_good, diff_bad

    def get_blimp_prompts(self, paradigm):
        ds = load_dataset('blimp', paradigm)

        prompts, good_gts, bad_gts = [], [], []

        if self.model_type == 'bert':
            mask_difference = self._mask_difference_mlm
        else:
            mask_difference = self._mask_difference_decoder

        for row in ds['train']:
            masked_sent, diff_good, diff_bad = mask_difference(row)
            if len(diff_good) != 1 or len(diff_bad) != 1:
                continue
            prompts.append(masked_sent)
            good_gts.append(diff_good[0])
            bad_gts.append(diff_bad[0])

        return prompts, good_gts, bad_gts

    def get_blimp_source_target_pairs(self, paradigm, search_target=None):
        prompts, good_gts, bad_gts = self.get_blimp_prompts(paradigm)

        target_func = self.TARGET_FUNC[paradigm]
        source_target_pairs = []

        for prompt, gt in zip(prompts, good_gts):
            target_type, target = target_func(prompt, gt)
            if target_type == search_target:
                source_target_pairs.append((prompt, target))
        return source_target_pairs

    def get_blimp_combinations(self, paradigm):
        prompts, good_gts, bad_gts = self.get_blimp_prompts(paradigm)
        target_func = self.TARGET_FUNC[paradigm]

        target_item_pairs = set()

        for prompt, gt in zip(prompts, good_gts):
            target_type, target, item = target_func(prompt, gt, True)
            target_item_pairs.add((target, item))

        return target_item_pairs