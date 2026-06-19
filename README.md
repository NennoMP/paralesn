<div align="center">

# ParalESN: Enabling parallel information processing in Reservoir Computing

[![arXiv](https://img.shields.io/badge/arXiv-2601.22296-b31b1b.svg)](https://arxiv.org/abs/2601.22296)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2601.22296)

![code-quality](https://github.com/nennomp/research-code-template/actions/workflows/code-quality.yml/badge.svg)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/nennomp/research-code-template)

</div>

This repository contains the official code for the paper:

```
ParalESN: Enabling parallel information processing in Reservoir Computing,
Matteo Pinna, Giacomo Lagomarsini, Andrea Ceni, Claudio Gallicchio,
International Conference on Machine Learning (ICML), 2026.
```

## Abstract
Reservoir Computing (RC) has established itself as an efficient paradigm for temporal processing. However, its scalability remains severely constrained by the need to process temporal data sequentially and the prohibitive memory footprint of high-dimensional reservoirs. To address these limitations, we revisit RC through the lens of structured operators and state space modeling, introducing Parallel Echo State Network (ParalESN). Leveraging diagonal linear recurrence in the complex domain, ParalESN enables parallel processing of temporal data and the construction of efficient, high-dimensional reservoirs. A thorough theoretical analysis demonstrates that the Echo State Property and the universality guarantees of traditional Echo State Networks are preserved, while also admitting an equivalent representation of arbitrary linear reservoirs in the complex diagonal form. Empirically, ParalESN achieves competitive predictive accuracy with traditional RC and with fully trainable sequence models, while delivering computational savings by orders of magnitude. Overall, ParalESN offers a scalable and principled pathway for integrating RC within the deep learning landscape.

<div align="center">
<img src="assets/paralesn.png?raw=true" alt="ParalESN" title="ParalESN">
</div>

## Setup
To install the required dependencies:
```
conda create -n paralesn python=3.12
conda activate paralesn
pip install -e .
```

## Experiments
TODO

## Citation
If you use the model or code in this repository, consider citing our paper:
```
@article{pinna2026paralesn,
  title={ParalESN: Enabling parallel information processing in Reservoir Computing},
  author={Pinna, Matteo and Lagomarsini, Giacomo and Ceni, Andrea and Gallicchio, Claudio},
  journal={arXiv preprint arXiv:2601.22296},
  year={2026}
}
```
