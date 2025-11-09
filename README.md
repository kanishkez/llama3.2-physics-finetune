# Physics-Tuned LLaMA 3.2 3B

This repository contains code for fine-tuning the LLaMA 3.2 3B model on physics-related content using the CAMEL-AI physics dataset. The fine-tuning process leverages Unsloth's efficient training framework with LoRA and PEFT techniques for parameter-efficient adaptation.

## Model Overview

The base model is Meta's LLaMA 3.2 3B, which has been fine-tuned on the [camel-ai/physics dataset](https://huggingface.co/datasets/camel-ai/physics) from Hugging Face. This dataset contains high-quality physics problems, explanations, and solutions suitable for training language models on scientific reasoning.

A pre-trained version of this model is available on Hugging Face: [kanishkez/llama3.2-fine-tuned-on-physics](https://huggingface.co/kanishkez/llama3.2-fine-tuned-on-physics). This model was trained on an A100 80GB GPU and can be downloaded directly for inference without requiring local training.

## Technical Approach

### Fine-Tuning Framework

This implementation uses Unsloth, an optimized fine-tuning library that significantly reduces memory usage and training time. Shoutout to Unsloth for providing an efficient framework that makes fine-tuning large language models accessible. 
Note: They have notebooks where you can do this yourself on colab. 

### LoRA and PEFT

The training process employs Low-Rank Adaptation (LoRA) and Parameter-Efficient Fine-Tuning (PEFT) techniques. These methods allow us to adapt the model by training only a small subset of additional parameters rather than updating all model weights. This approach offers several advantages:

- Reduced memory requirements during training
- Faster training times
- Smaller model checkpoint sizes
- Preservation of the base model's general capabilities while adding domain-specific knowledge

LoRA works by injecting trainable rank decomposition matrices into the transformer layers, enabling efficient adaptation with minimal overhead.

## Setup Instructions

### 1. Install Dependencies

First, install all required packages from the requirements file:

```bash
pip install -r requirements.txt
```

The requirements include Unsloth, transformers, datasets, and other necessary libraries for training and inference.

### 2. Train the Model

To train the model make sure you have enough gpu compute.

Run the training script:

```bash
python train.py
```

This script will:
- Load the LLaMA 3.2 3B base model
- Download and prepare the CAMEL-AI physics dataset
- Configure LoRA adapters for efficient training
- Fine-tune the model with optimal hyperparameters
- Save the trained model and adapters

Training was originally performed on an A100 80GB GPU. Depending on your hardware, you may need to adjust batch sizes or gradient accumulation steps in the training script.

### 3. Run the Chat Interface

After training is complete (or if using the pre-trained model), launch the interactive chat interface:

```bash
python chat.py
```

This provides a command-line interface where you can ask physics-related questions and receive responses from the fine-tuned model.

## Model Configuration

The fine-tuning configuration includes:

- Base model: Meta-LLaMA 3.2 3B
- LoRA rank: Optimized for physics domain adaptation
- Training dataset: CAMEL-AI physics dataset
- Framework: Unsloth with PEFT integration

## Use Cases

This model is designed for:

- Physics problem solving and explanations
- Scientific concept clarification
- Educational tutoring in physics topics
- Research assistance in physics domains


