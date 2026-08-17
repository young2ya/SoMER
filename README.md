# SoMER
> Soft Metric learning-based Entropy Regularization for Semi-supervised Fault Diagnosis

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)

---

## Overview

Obtaining labeled fault data in real industrial environments is costly and time-consuming. Existing semi-supervised methods rely on high-confidence pseudo-labels, which can lead to **confirmation bias** — once an incorrect label is assigned, the model continuously reinforces the error during training. SoMER addresses these challenges by proposing a semi-supervised fault diagnosis framework that achieves high diagnostic accuracy with only a small fraction of labeled data, without relying on hard pseudo-labels.

The framework combines three complementary objectives:
- **Cross-entropy loss** — supervised learning from labeled samples with MixUp augmentation
- **Entropy minimization** — encourages confident predictions on unlabeled data
- **Soft triplet loss** — metric learning using probability-weighted positive/negative degrees, enabling use of low-confidence unlabeled data without hard pseudo-labels, mitigating confirmation bias

---

## Pipeline

```
Vibration Signal → STFT Spectrogram → Augmentation → CNN Encoder → Fault Classification
```

---

## Results

Classification accuracy with 25 labeled samples per class:

| Dataset | Accuracy | Description |
|---------|----------|-------------|
| CWRU | 99.69% | Bearing fault benchmark |
| HUST | 99.92% | Compound fault benchmark |
| PU | 99.55% | Diverse fault benchmark |
| LAB | 99.23% | In-house experiment |

---

## Datasets

- **CWRU** — Case Western Reserve University bearing dataset
- **HUST** — Huazhong University of Science and Technology dataset
- **PU** — Paderborn University dataset
- **LAB** — In-house rolling element bearing experimental platform

---

## Keywords

`Semi-supervised Learning` `Fault Diagnosis` `Metric Learning` `Entropy Regularization` `Vibration Signal` `PHM`
