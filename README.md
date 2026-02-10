<p align="center">
 <h2 align="center"> Parameter Restoration </h2>
 <p align="center">
  A Tool for Analyzing and Improving Supervised Fine-Tuning of Large Language Models
 </p>
</p>

<p align="center">
 <a href="https://github.com/UmeanNever/ParamRestore/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/UmeanNever/ParamRestore"></a>
 <a href="https://arxiv.org/abs/2509.16596"><img alt="Paper" src="https://img.shields.io/badge/📖-Paper-red"></a>
</p>


<p align="center">
  <img src="/assets/restore.png" alt="llustration of parameter restoration." width="400"/>
  <br>
  <em>
    Figure: Illustration of parameter restoration. <br> We find that SFT introduces many unnecessary parameter updates, and model performance can be significantly improved by restoring some of the most updated parameters in the fine-tuned model to their pre-SFT values.
  </em>
</p>


## 📋 Overview

This repository provides code for analyzing and improving Supervised Fine-Tuning (SFT) of Large Language Models (LLMs) via parameter restoration, accompanying our paper.


- 📖 **Paper**: [Analyzing the Effects of Supervised Fine-Tuning on Model Knowledge from Token and Parameter Levels](https://arxiv.org/abs/2509.16596) . Accepted to **EMNLP 2025** (Main Conference). 🎉
- 🛠️ **Code**: This repository provides a clean, lightweight implementation of the proposed **Parameter Restoration** method, designed to be **easily extensible** to new models and fine-tuning setups.


In the paper, we study five LLMs from two model families on the CBQA task and show that both the category and scale of fine-tuning data can influence model knowledge in unexpected ways.

Our analysis further reveals that **up to 90% of parameter updates during SFT are unnecessary** and can even undermine the model’s ability to leverage its knowledge when answering questions. Restoring some of the most heavily updated parameters in the fine-tuned model to their original values before SFT can significantly improve performance.

This motivates a simple, effective, and generally applicable tool:
**selectively restore subsets of parameters** in a fine-tuned model to their pre-SFT values and measure the resulting impact.

For more information and a detailed introduction to parameter restoration, please refer to our paper.

## 🚀 Quick Start

The code in this repository is refactored from our original research code to provide a **clean, self-contained implementation** of the parameter restoration method.

**Given:**
- an original model (e.g., pre-trained), and
- a target model (e.g., fine-tuned) with the same architecture,

**this toolkit performs the following steps:**
1. Computes each parameter's relative change (relative differences) from the original model to the target model.
2. Globally selects the top (or bottom / random) *k%* of parameters based on their relative changes, within a specified region (e.g., all transformer layers, embeddings, MLPs, or attention modules).  
3. Restores the selected parameters in the fine-tuned model to their corresponding values in the original model.  
4. Saves the restored model for downstream evaluation, along with concise TSV/JSON logs for analysis.

### File Structure
- `launch_param_restore.py`: **Entry point** and batch runner. It calls `param_restore.py` with your customized configuration. 
- `param_restore.py`: Core implementation (diff → select → restore → save + logging).  
- See inline comments in these files for full details.

### Running the Code

Define model paths, restoration settings, and other configurations in `launch_param_restore.py`, then run the script to perform parameter restoration:
```
python launch_param_restore.py
```

### Dependencies

You may need to install the following dependencies in your Python environment:
```
pip install torch transformers numpy fire
```
GPU is recommended for large models, but not strictly required.

Notes:
- Dtype/device: the script runs with the model dtype you specify (default `bf16`) and supports both CPU and GPU execution.
- Downstream evaluation of the restored models is left to the user. Users may choose appropriate LLM evaluation toolkits based on their specific use cases.

## 📝 Citation

If you find our work helpful, please consider citing our paper:

```bibtex
@inproceedings{ye-etal-2025-analyzing,
    title = "Analyzing the Effects of Supervised Fine-Tuning on Model Knowledge from Token and Parameter Levels",
    author = "Ye, Junjie  and
      Yang, Yuming  and
      Nan, Yang  and
      Li, Shuo  and
      Zhang, Qi  and
      Gui, Tao  and
      Huang, Xuanjing  and
      Wang, Peng  and
      Shi, Zhongchao  and
      Fan, Jianping",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.25/",
    doi = "10.18653/v1/2025.emnlp-main.25",
    pages = "471--513",
    ISBN = "979-8-89176-332-6"
}
```
