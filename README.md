# fine-tune-classification-model-with-adapters

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square&logo=huggingface)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%7C%20DoRA-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

An industry-style pipeline for fine-tuning pretrained transformer models (BERT, RoBERTa, DeBERTa) on text classification tasks using adapter-based Parameter-Efficient Fine-Tuning (PEFT) methods. Instead of retraining all model weights, we inject small trainable adapter modules — specifically LoRA and DoRA — into the attention layers, reducing GPU memory usage by up to 60% while preserving near full fine-tune accuracy. This makes the approach practical for teams working with limited compute resources or managing multiple classification tasks on a single base model.

## What this repo covers

- Fine-tuning pretrained encoder models for binary and multi-class text classification using LoRA and DoRA adapters
- Comparing adapter-based training against a full fine-tune baseline to understand the accuracy-efficiency tradeoff
- Applying industry best practices such as layer-wise learning rate decay, differential weight decay, linear warmup scheduling, gradient clipping, and mixed precision training
- Exporting and merging trained adapter weights back into the base model for clean, self-contained inference
- Configuring all experiments through YAML files with no hardcoded values, enabling clean reproducibility across runs

## Who this is for

ML engineers and researchers who want a clean, no-bloat reference implementation for adapter-based classification fine-tuning that reflects how it is actually done in production — not just in tutorials.
