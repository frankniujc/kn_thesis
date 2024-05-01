import torch
import numpy as np

from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .patch import unpatch_ff_layers


def to_percent(y, position):
    s = str(int(100 * y))
    if plt.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

def pre_post_edit_probs(kn, neurons, data, filter_fn, token_idxs):
    pre_probs, post_probs = [], []

    for prompt, good_gt, bad_gt in zip(*data):
        if filter_fn(prompt, good_gt, bad_gt):
            pre_probs.append(kn.generate(prompt).squeeze().detach()[token_idxs])
            modified_layers = kn.erase(prompt, neurons)
            post_probs.append(kn.generate(prompt).squeeze().detach()[token_idxs])

            unpatch_ff_layers(
                model=kn.model,
                layer_indices=modified_layers,
                transformer_layers_attr=kn.transformer_layers_attr,
                ff_attrs=kn.input_ff_attr,
            )
    return pre_probs, post_probs

def plot(
    pre_probs, post_probs, tokens, figsize,
    suptitle, suptitle_y=1.05,
    ttest=True, pvalue_threshold=0.05, grey_vlines=[],
    tick_range=None, bar_width=0.4,
    y_label='Probability Change', save_path=None):

    before = pre_probs.mean(dim=0).cpu()
    after = post_probs.mean(dim=0).cpu()

    fig, ax = plt.subplots(figsize=figsize)

    if tick_range is not None:
        ax.set_yticks(tick_range)

    width = 0.4
    n_tokens = len(tokens)

    ax.bar(np.arange(n_tokens), (after-before)/torch.min(before, after), width=width)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))

    ax.set_xticks(np.arange(n_tokens), tokens[:n_tokens], rotation=90)

    for vline_x in grey_vlines:
        ax.axvline(vline_x, color='grey', linewidth=0.5)

    if ttest:
        for i, xtick in enumerate(ax.get_xticklabels()):
            test = ttest_ind(pre_probs[:,i].tolist(), post_probs[:,i].tolist())
            if test.pvalue < pvalue_threshold:
                xtick.set_color('r')

    ax.set_ylabel(y_label)
    ax.yaxis.grid()
    fig.suptitle(suptitle, y=1.05)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig