# contrastive-learning-text-classification-finetuning

# Fine-tuning all-mpnet-base-v2 for Text Classification on AG News using Contrastive Learning

## Description

This repository contains a Jupyter notebook and a Python script for dataset preparation.

The notebook demonstrates a **fine-tuning pipeline of the `all-mpnet-base-v2` model for text classification** on the **AG News dataset**, using a **Contrastive Loss** training objective.
Instead of training a traditional classification head, the model learns to map text inputs into an embedding space, where classification is performed based on similarity between embeddings.

Two training strategies are explored:
- fine-tuning using the **original AG News labels**
- fine-tuning using **descriptive labels** to improve class separability in the embedding space

The second strategy leads to a slight improvement in performance by using more descriptive labels.

## Repository structure
```text
.
├── all-mpnet-base-v2-agnews-finetuned.ipynb    # Fine-tuning experiments and evaluation
├── extract_subsets.py      # Script to generate dataset subsets
└── README.md
