# Political Evasion Classifier

An NLP comparison project for classifying political interview answers as **Clear Reply**, **Ambivalent**, or **Clear Non-Reply**. It progresses from interpretable classical baselines to efficient and higher-capacity transformer fine-tuning.

## Status

The methodology and source notebooks are packaged, but **results are pending a clean end-to-end rerun**. The Kaggle source exports did not retain a complete, trustworthy set of outputs, so this repository intentionally makes no accuracy or F1 claim.

## Approaches

1. **Classical baselines** — TF-IDF and averaged Google News Word2Vec features with class-balanced logistic regression, stratified cross-validation, learning curves, and error analysis.
2. **DistilBERT** — sentence-pair tokenization, weighted cross-entropy, AdamW, linear warm-up/decay, gradient clipping, stratified validation, and early stopping.
3. **DeBERTa-v3** — dynamic padding, mixed precision, label smoothing, class weighting, checkpoint selection, per-class evaluation, and length-based subgroup analysis.

Questions and answers are encoded as a sentence pair so the classifiers can model whether the response resolves the proposition asked, rather than judging the answer in isolation.

## Repository layout

```text
.
├── notebooks/
│   ├── 01_classical_baselines.ipynb
│   ├── 02_distilbert.ipynb
│   └── 03_deberta.ipynb
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Reproduce

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

Run the notebooks in numerical order. Transformer notebooks benefit from a CUDA GPU; reduce batch size if memory is limited.

The notebooks pin QEvasion to revision:

```text
3afc18f0b582b3cfdb927822cff57ddc6e871f9c
```

Model weights and the Google News vectors are downloaded by their respective libraries and are never stored in this repository.

## Evaluation protocol

- Fixed random seed (`42`)
- Stratified train/validation splits
- Class-aware training for the imbalanced target
- Accuracy, macro F1, weighted F1, precision, recall, and per-class reports
- Confusion matrices, learning/training curves, and answer-length subgroup analysis

Before publishing results, rerun every cell in a fresh environment, save the exact environment and hardware details, and report both macro and weighted F1. Do not copy historical notebook scores into this README.

## Original Kaggle sources

- [Classical baseline notebook](https://www.kaggle.com/code/pandeliskotsas/sdi2300239-dsbf)
- [DistilBERT notebook](https://www.kaggle.com/code/pandeliskotsas/distilbert)
- [DeBERTa-v3 notebook](https://www.kaggle.com/code/pandeliskotsas/deberta)

These are provenance links. This repository contains cleaned source and does not include Kaggle outputs, credentials, private paths, or course-submission material.

## Data and model licenses

- [QEvasion dataset](https://huggingface.co/datasets/ailsntua/QEvasion) — **CC BY-NC-ND 4.0**. The dataset is not redistributed here. Review the [license terms](https://creativecommons.org/licenses/by-nc-nd/4.0/) before use, especially the non-commercial and no-derivatives restrictions.
- [DistilBERT base uncased](https://huggingface.co/distilbert/distilbert-base-uncased) — **Apache-2.0**.
- [Microsoft DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) — **MIT**.
- Google News Word2Vec vectors are an external download and are excluded from this repository and its MIT license. Verify their upstream terms before use or redistribution.

Raw interview text, processed dataset copies, model checkpoints, submissions, vector weights, and caches are intentionally excluded.

## Responsible use

This is an experimental text-classification workflow, not a factuality detector or a definitive assessment of any public figure. Labels reflect the dataset annotation scheme and can encode ambiguity or annotator judgment. Review the QEvasion dataset card, model-card limitations, class-level errors, and subgroup behavior before drawing conclusions.

## License

Pantelis Kotsas's original code and documentation in this repository are MIT licensed. The dataset, pretrained models, downloaded vector weights, source interview text, and generated model artifacts remain governed by their respective upstream terms; see [LICENSE](LICENSE).
