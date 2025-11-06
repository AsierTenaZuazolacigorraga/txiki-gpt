# txiki-gpt

A minimal GPT implementation for training and fine-tuning language models. This project provides a clean, demo implementation of the GPT architecture with support for training from scratch or fine-tuning pre-trained models.

## Features

- **Full GPT implementation** in a single file (`model.py`) with Flash Attention support
- **Character-level tokenization** for small datasets (e.g., Shakespeare)
- **Training and inference scripts** with configurable hyperparameters
- **Support for GPT-2 fine-tuning** via Hugging Face transformers
- **Efficient training** with gradient accumulation, learning rate scheduling, and checkpointing

## Quick Start

Train a model from scratch:
```bash
python train.py
```

Run inference:
```bash
python inference.py
```

## Model Architecture

The implementation includes:
- Causal self-attention with Flash Attention (PyTorch 2.0+)
- Multi-layer transformer blocks with residual connections
- Weight tying between token embeddings and output layer
- Configurable model size (layers, heads, embedding dimensions)

## Configuration

Edit `config.py` to adjust:
- Model architecture (n_layer, n_head, n_embd)
- Training hyperparameters (learning rate, batch size, etc.)
- Dataset and data paths
- Inference parameters (temperature, top_k)

## Reference

This project is inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - the simplest, fastest repository for training/finetuning medium-sized GPTs.

