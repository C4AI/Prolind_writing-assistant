# AI Training – PROLIND Writing Assistant

This directory contains the code used to **train and evaluate the models** behind the PROLIND Writing Assistant (PROLIND / PROLIND_writing-assistant project).

---

## 1. Overview

The goal of this module is to provide a **reproducible training pipeline** for language models used in the writing assistant, including:

- Data loading and preprocessing (e.g., tokenization, text normalization)
- Model definition and loading
- Training loop (optimization, scheduling, logging, checkpoints)
- Evaluation (accuracy, BLEU / chrF / other metrics)
- Utilities for exporting trained models for **inference services** (FastAPI, CLI, etc.)

Typical use cases:

- Training / fine-tuning translation, next-word prediction, or spelling-correction models
- Running evaluation on held-out corpora
- Generating artifacts to be deployed in the `services` or `inference` part of the project

---

## 2. Directory Structure (template)

_Adapt this section to match your actual files._

```text
src/ai-training/
|-- prepare_dataset.py       # Converts a parallel corpus to the three tasks supported: translation, next-word prediction, spell correction
|-- s2s-model_train_test.py  # A single script to train a model and (optionally) test it and generate a report
|-- run_pipeline.sh          # A sample script to run an end-to-end pipeline using sample dataset (Brazil's constitution in Nheengatu)
|-- README.md                # This file
