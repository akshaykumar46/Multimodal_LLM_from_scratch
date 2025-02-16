# Multimodal LLM from Scratch

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

An educational implementation of a multimodal language model combining vision and text capabilities, built from fundamental components.

![Multimodal Architecture Diagram](architecture.png)

## Features
- **Dual Modality Processing** - Handles both images and text inputs
- **Vision Encoder** - SigLIP-based image understanding
- **Language Model** - Gemma variant for text generation
- **Efficient Inference** - Optimized for single GPU usage

## Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- CUDA or MPS support (for faster inference)

```git clone https://github.com/akshaykumar46/Multimodal_LLM_from_scratch cd Multimodal_LLM_from_scratch```
### Install dependencies
```pip install -r requirements.txt  # Create this file with your dependencies```
### Download pretrained weights
```chmod +x download_weights.sh ./download_weights.sh```


## Quick Start

### Basic Inference
```python inference.py
–image-path samples/cricket.jpg
–prompt “Identify the sport being played”
–temperature 0.7
–max-new-tokens 50
```
