# Transformer Model from Scratch

## Overview
This repository contains an implementation of the **Transformer** model as described in the paper [\"Attention is All You Need\"](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). The model is built from scratch using **PyTorch** and consists of **Self-Attention, Multi-Head Attention, Encoder, and Decoder blocks**. 

## Features
- Implements the **Transformer architecture from scratch** without external ML libraries.
- Includes **Self-Attention and Multi-Head Attention** mechanisms.
- Supports **Positional Encoding** for sequence order preservation.
- Implements **Encoder and Decoder** blocks with Layer Normalization and Feed-Forward Networks.
- Fully customizable **hyperparameters** (embedding size, number of heads, dropout, etc.).
- Designed for **sequence-to-sequence** tasks like Machine Translation.

## Model Architecture
The Transformer model consists of:

1. **Self-Attention Mechanism** - Computes attention scores between tokens in a sequence.
2. **Multi-Head Attention** - Uses multiple attention heads to learn diverse representations.
3. **Positional Encoding** - Adds position information to token embeddings.
4. **Feed-Forward Networks** - Fully connected layers with non-linearity.
5. **Encoder Block** - Contains Self-Attention and Feed-Forward layers.
6. **Decoder Block** - Uses Self-Attention, Encoder-Decoder Attention, and Feed-Forward layers.

## Dependencies
To run this project, install the required Python libraries:

```bash
pip install torch numpy matplotlib
