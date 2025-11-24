# LLM from Scratch

This repository contains a comprehensive, step-by-step implementation of a Large Language Model (specifically a GPT-like architecture) built from the ground up using PyTorch. It is designed for educational purposes, breaking down every component of an LLM into understandable chunks.

## Project Overview

The core of this project is located in the `Understanding each component` directory, which contains a series of numbered Jupyter Notebooks. These notebooks guide you through the entire process, from basic tokenization to pre-training and fine-tuning a full model.

## Roadmap & Notebooks

The notebooks are structured to be followed in order, guiding you from the basics to a fully functional and fine-tuned LLM.

### 1. Fundamentals & Data Processing
*   **`01_Simple_Tokenization.ipynb`**:
    *   **Purpose**: Introduces the basics of text tokenization, the first step in processing text for LLMs.
    *   **Key Concepts**: Splitting text into tokens, creating a vocabulary, mapping tokens to integer IDs, and handling unknown words with special tokens like `<|unk|>`.
    *   **Code**: Implements a simple `Tokenizer_v1` class using Python's `re` module.
*   **`02_BPE_Tokenizer.ipynb`**:
    *   **Purpose**: Implements Byte Pair Encoding (BPE), the advanced tokenization method used by models like GPT-2 and GPT-3.
    *   **Key Concepts**: Iteratively merging frequent character pairs to form subword tokens, balancing vocabulary size and sequence length.
    *   **Code**: Uses the `tiktoken` library to demonstrate efficient BPE tokenization.
*   **`03_Embeddings.ipynb`**:
    *   **Purpose**: Explains how to represent tokens as dense vectors (embeddings) that capture semantic meaning.
    *   **Key Concepts**: Token embeddings vs. Positional embeddings. Understanding how LLMs process sequence order.
    *   **Code**: Uses `torch.nn.Embedding` to create learnable embedding layers for both tokens and positions.

### 2. Attention Mechanisms
*   **`04_Simplified_Attention_Mechanism.ipynb`**:
    *   **Purpose**: Demystifies the core mathematical concept of "Attention" without learnable weights.
    *   **Key Concepts**: Dot-product attention, calculating attention scores, and computing context vectors as weighted sums of input vectors.
*   **`05_Self_Attention_with_weights.ipynb`**:
    *   **Purpose**: Introduces the "Self-Attention" mechanism with trainable parameters.
    *   **Key Concepts**: Query, Key, and Value matrices. How the model learns to focus on different parts of the input sequence.
    *   **Code**: Implements `SelfAttention_v1` and `SelfAttention_v2` classes using PyTorch linear layers.
*   **`06_Causal_Attention.ipynb`**:
    *   **Purpose**: Adapts self-attention for generative models (like GPT) by preventing the model from "peeking" into the future.
    *   **Key Concepts**: Causal masking (using a lower-triangular matrix), masking with `-inf` before softmax, and dropout for regularization.
    *   **Code**: Refines the attention module into a `CausalAttention` class.
*   **`07_Mutli_head_Attention.ipynb`**:
    *   **Purpose**: Extends causal attention to "Multi-Head Attention," allowing the model to capture multiple types of relationships simultaneously.
    *   **Key Concepts**: Splitting embeddings into multiple heads, parallel processing, and projecting concatenated outputs.
    *   **Code**: Implements an efficient `MultiHeadAttention` class using tensor reshaping and transposition.

### 3. Transformer Architecture
*   **`08_DummyGPT.ipynb`**:
    *   **Purpose**: Provides a high-level architectural overview of a GPT model using placeholder components.
    *   **Key Concepts**: The overall data flow: Embeddings -> Transformer Blocks -> LayerNorm -> Output Head.
    *   **Code**: Defines the skeleton `DummyGPTModel` class to visualize input/output shapes.
*   **`09_Layer_Normalization.ipynb`**:
    *   **Purpose**: Implements Layer Normalization, a critical component for stabilizing training in deep networks.
    *   **Key Concepts**: Normalizing inputs across the feature dimension, learnable scale and shift parameters.
    *   **Code**: Creates a custom `LayerNorm` class matching PyTorch's implementation.
*   **`10_FeedForward_with_GELU.ipynb`**:
    *   **Purpose**: Builds the position-wise Feed-Forward Network (FFN) used within each Transformer block.
    *   **Key Concepts**: GELU (Gaussian Error Linear Unit) activation function, expansion and contraction of embedding dimensions.
    *   **Code**: Implements the `GELU` activation and the `FeedForward` module.
*   **`11_Shortcut_connections.ipynb`**:
    *   **Purpose**: Explains Residual (Shortcut) Connections and their role in enabling deep network training.
    *   **Key Concepts**: Mitigating the vanishing gradient problem by adding the input to the output of a layer.
    *   **Code**: Demonstrates gradient flow improvements with a simple neural network example.
*   **`12_Entire_Transformer_block.ipynb`**:
    *   **Purpose**: Assembles the Attention, Feed-Forward, and Normalization layers into a single Transformer Block.
    *   **Key Concepts**: Pre-LayerNorm architecture (standard in modern LLMs) and integrating residual connections.
    *   **Code**: Defines the `TransformerBlock` class, the building block of the GPT model.

### 4. Building the Full Model
*   **`13_Entire_GPT_Model.ipynb`**:
    *   **Purpose**: Combines all components to create the full GPT architecture.
    *   **Key Concepts**: Stacking Transformer blocks, final layer normalization, and the output head. Parameter counting and memory estimation.
    *   **Code**: Implements the complete `GPTModel` class and a simple text generation function.
*   **`14_LLM_loss_function.ipynb`**:
    *   **Purpose**: Defines the objective function used to train the model.
    *   **Key Concepts**: Cross-Entropy Loss, converting logits to probabilities, and perplexity.
    *   **Code**: Implements loss calculation using `torch.nn.functional.cross_entropy`.
*   **`15_Training_Validation_loss.ipynb`**:
    *   **Purpose**: Sets up the training infrastructure.
    *   **Key Concepts**: Creating data loaders for training and validation, batch processing, and calculating loss over epochs.
    *   **Code**: Implements `calc_loss_batch` and `calc_loss_loader` utility functions.
*   **`16_LLM_pre-Training_pipeline.ipynb`**:
    *   **Purpose**: Executes the full pre-training loop for the LLM.
    *   **Key Concepts**: The training loop (forward pass, loss computation, backpropagation, optimizer step), monitoring training progress.
    *   **Code**: Defines the `train_model_simple` function to train the model on the `the-verdict.txt` dataset.

### 5. Generation & Fine-tuning
*   **`17_LLM_Decoding_Strategies.ipynb`**:
    *   **Purpose**: Explores different methods for generating text from the trained model.
    *   **Key Concepts**: Greedy decoding, Temperature scaling, Top-k sampling, and Nucleus (Top-p) sampling.
    *   **Code**: Implements a flexible `generate` function supporting various sampling strategies.
*   **`18_GPT.ipynb`**:
    *   **Purpose**: Demonstrates how to load pre-trained weights (e.g., from OpenAI's GPT-2) into our custom architecture.
    *   **Key Concepts**: Weight mapping, handling tensor shape mismatches, and verifying model performance with pre-trained weights.
    *   **Code**: Implements `load_weights_into_gpt` to transfer weights from OpenAI's format to our PyTorch model.
*   **`19_Classification_finetuning.ipynb`**:
    *   **Purpose**: Adapts the generative model for a classification task (Spam Detection).
    *   **Key Concepts**: Modifying the output head for classification, creating a balanced dataset, and fine-tuning with a classification loss.
    *   **Code**: Implements `SpamDataset` and the fine-tuning loop for binary classification.
*   **`20_Instruction_finetuning.ipynb`**:
    *   **Purpose**: Fine-tunes the model to follow user instructions (Instruction Tuning).
    *   **Key Concepts**: Formatting data with instruction templates (Alpaca style), custom data collation for variable-length sequences, and training on instruction-response pairs.
    *   **Code**: Defines `InstructionDataset` and a custom collate function for efficient instruction tuning.

## Data

The `Data` directory contains the datasets used for training and experiments:
*   **`the-verdict.txt`**: A small text corpus used for initial pre-training experiments.
*   **`sms_spam_collection/`**: A dataset used for the classification fine-tuning task.
*   **`instruction-data.json`**: Dataset containing instruction-response pairs for instruction fine-tuning.

## Models

The `model` directory stores saved model checkpoints and artifacts:
*   **`gpt2-small124M-sft.pth`**: Checkpoint for a supervised fine-tuned model.
*   **`review_classifier.pth`**: Checkpoint for the fine-tuned classification model.
*   **`model.pth`** & **`model_and_optimizer.pth`**: Saved states of the model during training.

## Getting Started

1.  Clone this repository.
2.  Install the required dependencies (primarily PyTorch).
3.  Navigate to `Understanding each component/` and start with notebook `01`.

```bash
pip install torch numpy jupyter
jupyter notebook
```
