# What does the Knowledge Neuron Thesis Have to do with Knowledge?

This repository provides the implementation of the ICLR 2024 paper
[_What does the Knowledge Neuron Thesis Have to do with Knowledge?_](https://openreview.net/forum?id=2HJRwwbV3G)

Links:
- [Blog Post](https://www.cs.toronto.edu/~niu/publications/iclr2024/)
- [OpenReview](https://openreview.net/forum?id=2HJRwwbV3G)
- [Poster](https://www.cs.toronto.edu/~niu/research/iclr2024/iclr2024_poster.pdf)

**Abstract**: We reassess the Knowledge Neuron (KN) Thesis: an interpretation of the mechanism underlying the ability of large language models to recall facts from a training corpus. This nascent thesis proposes that facts are recalled from the training corpus through the MLP weights in a manner resembling key-value memory, implying in effect that "knowledge" is stored in the network. Furthermore, by modifying the MLP modules, one can control the language model's generation of factual information. The plausibility of the KN thesis has been demonstrated by the success of KN-inspired model editing methods (Dai et al., 2022; Meng et al., 2022).

We find that this thesis is, at best, an oversimplification. Not only have we found that we can edit the expression of certain linguistic phenomena using the same model editing methods but, through a more comprehensive evaluation, we have found that the KN thesis does not adequately explain the process of factual expression. While it is possible to argue that the MLP weights store complex patterns that are interpretable both syntactically and semantically, these patterns do not constitute "knowledge." To gain a more comprehensive understanding of the knowledge representation process, we must look beyond the MLP weights and explore recent models' complex layer structures and attention mechanisms.

## Installation
We recommend `virtualenv` for managing dependencies. To start, create a virtual environment and install the required dependencies.

```bash
virtualenv env -p python3
source env/bin/activate
pip install -r requirements.txt
```

## Localising Syntactic Phenomena

[`localise_kn.ipynb`](localise_kn.ipynb) demonstrates our experiments in localising various syntactic phenomena. It also shows our how our analysis is conducted.

## Causal Tracing and ROME

[`ct_rome.ipynb`](localise_kn.ipynb) shows our experiments related to causal tracing and ROME edit.

## Citation
```bibtex
@inproceedings{niu2024what,
  title={What does the Knowledge Neuron Thesis Have to do with Knowledge?},
  author={Jingcheng Niu and Andrew Liu and Zining Zhu and Gerald Penn},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=2HJRwwbV3G}
}
```
