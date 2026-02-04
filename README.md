<p align="center">
 <h2 align="center"> Parameter Restoration Analysis </h2>
 <p align="center">
  A Tool for Analyzing Supervised Fine-Tuning of Large Language Models
 </p>
</p>

<p align="center">
 <a href="https://github.com/UmeanNever/ParamRestore/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/UmeanNever/ParamRestore"></a>
 <a href="https://arxiv.org/abs/2509.16596"><img alt="Paper" src="https://img.shields.io/badge/📖-Paper-red"></a>
</p>


<p align="center">
  <img src="/assets/restore.png" alt="llustration of the intuition behind RSR." width="450"/>
  <br>
  <em>
    Figure 1: Illustration of parameter restoration. <br> We find that SFT introduces many unnecessary parameter updates, and model performance can be significantly improved by restoring some of the most updated parameters in the fine-tuned model to their pre-SFT values.
  </em>
</p>


## 📋 Overview

This repository provides code for analyzing Supervised Fine-Tuning (SFT) of Large Language Models (LLMs) via parameter restoration, accompanying our paper *“Analyzing the Effects of Supervised Fine-Tuning on Model Knowledge from Token and Parameter Levels.”*


- 📖 **Paper**: [Read our paper on arXiv](https://arxiv.org/abs/2509.16596). Accepted to EMNLP 2025 (main conference).
- 🛠️ **Code**: This repository provides a clean, lightweight implementation of the proposed **Parameter Restoration Analysis** method, designed to be easily extensible to new models and fine-tuning setups.


In the paper, we study five LLMs from two model families on the CBQA task and show that both the category and scale of fine-tuning data can influence model knowledge in unexpected ways.

Our analysis further reveals that **up to 90% of parameter updates during SFT are unnecessary** and can even degrade model knowledge. Restoring some of the most heavily updated parameters in the fine-tuned model to their original values before SFT can significantly improve performance.

This motivates a simple, effective, and generally applicable analysis tool:
**selectively restore subsets of parameters** in a fine-tuned model to their pre-SFT values and measure the resulting impact.

## 🚀 Quick Start

The code in this repository is refactored from our original research code to provide a clean, self-contained implementation of the parameter restoration analysis method.

**Given:**
- an original model (e.g., pre-trained), and
- a target model (e.g., fine-tuned) with the same architecture,

**this toolkit performs the following steps:**
1. Computes each parameter's relative change (relative differences) from the original model to the target model.
2. Globally selects the top (or bottom / random) *k%* of parameters based on their relative changes, within a specified region (e.g., all transformer layers, embeddings, MLPs, or attention modules).  
3. Restores the selected parameters in the fine-tuned model to their corresponding values in the original model.  
4. Saves the restored model for downstream evaluation, along with concise TSV/JSON logs for analysis.

### Files
- `param_restore.py`: Core implementation (diff → select → restore → save + logging).  
- `launch_param_restore.py`: **Entrypoint** and batch runner. 
- See inline comments in these files for full details.

### Simple Run

Define model paths, restoration settings, and other configurations in `launch_param_restore.py`, then run the script to perform parameter restoration:
```
python launch_param_restore.py
```

### Environment
You may need to install the following dependencies in your Python environment:
```
pip install torch transformers numpy fire
```
GPU is recommended for large models, but not strictly required.

Notes:
- Dtype/device: the script runs with the model dtype you specify (default `bf16`) and supports both CPU and GPU execution.
- Downstream evaluation of the restored models is left to the user. Users may choose appropriate LLM evaluation toolkits based on their specific use cases.
