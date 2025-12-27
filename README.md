# txiki-gpt

A minimal GPT implementation for training and fine-tuning language models. Inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT), this project provides a clean, educational implementation of the GPT architecture focused on simplicity and clarity.

## Overview

txiki-gpt is a from-scratch implementation of the GPT (Generative Pre-trained Transformer) architecture, designed to be simple to understand and easy to experiment with. The entire model fits in a single file, making it perfect for learning how transformers work or quickly prototyping language model experiments.

## Features

- **Single-file GPT implementation** (`model.py`) with Flash Attention support for modern PyTorch 2.0+
- **Character-level tokenization** for small datasets (perfect for the included Shakespeare corpus)
- **Training from scratch** or **fine-tuning pre-trained GPT-2 models**
- **Clean training loop** with gradient accumulation, cosine learning rate scheduling, and automatic checkpointing
- **Simple configuration** via a single `config.py` file
- **Fast inference** with temperature and top-k sampling

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

## Project Structure

```
txiki-gpt/
├── model.py          # Complete GPT implementation with Flash Attention
├── train.py          # Training script with loss estimation and checkpointing
├── inference.py      # Text generation script
├── config.py         # All hyperparameters and settings
└── data/            # Training datasets (Shakespeare included)
```

## Inspiration

This project is directly inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT), the simplest and fastest repository for training/finetuning medium-sized GPTs. Like nanoGPT, txiki-gpt prioritizes:

- **Hackability**: Clean, readable code that's easy to understand and modify
- **Simplicity**: Minimal dependencies, everything you need in ~500 lines of code
- **Educational value**: Perfect for learning how GPT models actually work
- **Speed**: Efficient implementation using modern PyTorch features

The name "txiki" means "small" in Basque, reflecting the project's minimalist philosophy.

