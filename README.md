<div align="center">

<h1> MetaMem: Evolving Meta-Memory for Knowledge Utilization through Self-Reflective Symbolic Optimization
</h1>

<h5 align="center">
<a href='https://arxiv.org/abs/xxxx.xxxxx'><img src='https://img.shields.io/badge/Paper-MetaMem-red?logo=arxiv&logoColor=white'></a>
<a href='https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen3--30B--A3B--Instruct-blue'></a>
<a href='https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Llama3.1--70B--Instruct-blue'></a>
<a href='https://huggingface.co/Qwen/Qwen3-235B-A22B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen--235B--A22B-blue'></a>
<a href='https://huggingface.co/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LLMLingua--2-blue'></a>
<a href='https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-all--MiniLM--L6--v2-blue'></a>

Haidong Xin<sup>1*</sup>,
Xinze Li<sup>1*</sup>,
Zhenghao Liu<sup>1†</sup>,
Yukun Yan<sup>2</sup>,
Shuo Wang<sup>2</sup>,
Cheng Yang<sup>3</sup>,
Yu Gu<sup>1</sup>,
Ge Yu<sup>1</sup>,
Maosong Sun<sup>2</sup>

<sup>1</sup>Northeastern University, <sup>2</sup>Tsinghua University, <sup>3</sup>BUPT

</h5>
</div>


## 📖 Introduction

MetaMem addresses the challenge of fragmented memory and degraded reasoning in long-horizon interactions by constructing a self-evolving meta-memory framework. It iteratively distills transferable knowledge utilization experiences through self-reflection and environmental feedback, guiding LLMs to accurately extract critical evidence from scattered memory units. MetaMem demonstrates strong generalization capabilities by significantly enhancing performance in multi-session integration and temporal reasoning tasks across various retrieval-augmented architectures.

![](figs/pipeline.png)

## ⚙️ Setup

### 1. Create Conda Environment

```shell
conda create -n metamem python=3.11 -y
conda activate metamem
```

### 2. Install [LightMem](https://github.com/zjunlp/LightMem)

```shell
git clone https://github.com/zjunlp/LightMem.git
cd LightMem
pip install -e .
```

### 3. Pretrained LLM weights

```shell
# Qwen3-30B-A3B-Instruct
hf download Qwen/Qwen3-30B-A3B-Instruct-2507

# Llama3.1-70B-Instruct
hf download meta-llama/Llama-3.1-70B-Instruct

# Qwen3-235B-A22B
hf download Qwen/Qwen3-235B-A22B

# LLMLingua2
hf download microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank

# all-MiniLM-L6-v2
hf download sentence-transformers/all-MiniLM-L6-v2
```

### 4. Deploy OpenAI Model Serve

```shell
# Qwen3-30B-A3B-Instruct
docker run -d --gpus all \
-e CUDA_VISIBLE_DEVICES=0,1 \
-v /parent_dir_to_models:/workspace \
-p 29001:29001 \
--ipc host \
--name sglang_qwen_30b \
lmsysorg/sglang:latest \
python3 -m sglang.launch_server \
--model-path /workspace/Qwen3-30B-A3B-Instruct-2507 \
--served-model-name qwen3-30b \
--host 0.0.0.0 \
--port 29001 \
--tp 2 \
--mem-fraction-static 0.85 \
--trust-remote-code

# Qwen3-235B-A22B
docker run -d --gpus all \
-e CUDA_VISIBLE_DEVICES=2,3,4,5 \
-v /parent_dir_to_models:/workspace \
-p 29002:29002 \
--ipc host \
--name sglang_qwen_235b \
lmsysorg/sglang:latest \
python3 -m sglang.launch_server \
--model-path /workspace/Qwen3-235B-A22B \
--served-model-name qwen3-235b \
--host 0.0.0.0 \
--port 29002 \
--tp 4 \
--mem-fraction-static 0.85 \
--trust-remote-code
```

## 🔧 Reproduction Guide

### 1. Dataset Preprocessing

```shell
wget -c https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json -O data/longmemeval_s_cleaned.json
```

### 2. Construct Memory

```shell
bash scripts/construct_memory.sh
```

### 3. Training MetaMem

```shell
# process train data
bash scripts/process_train_data.sh

# k-fold split
bash scripts/split_data.sh

# train
bash scripts/train_metamem.sh
```

### 4. Evaluate MetaMem

```shell
bash scripts/eval_metamem.sh
```

### 5. Inference

```shell
bash scripts/infer_metamem.sh
```

## 📁 Repository Structure

```
MetaMem/
├── README.md
├── LICENSE
├── figs/                      # README figures
├── scripts/                   # The scripts used to run the experiments
└── src/
    ├── construct_memory.py    # Construct the factual memory via LightMem
    ├── eval_metamem.py        # Evaluate the trained meta memory
    ├── infer_metamem.py       # Inference the trained meta memory
    ├── process_train_data.py  # Preprocess the dataset
    ├── split_data.py          # Split the dataset for k-fold validation
    └── train_metamem.py       # Train meta memory
```

## 📄 Acknowledgement 

Our work is built on the following codebases, and we are deeply grateful for their contributions.

- [LightMem](https://github.com/zjunlp/LightMem): We utilize LightMem to consturct factual memory.
- [SGLang](https://docs.sglang.io/): We utilize SGLang framework to deploy LLM serve.

## 🥰 Citation

We appreciate your citations if you find our paper related and useful to your research!

```bibtex
@article{xin2026metamem,
    author = {Xin, Haidong and Li, Xinze and Liu, Zhenghao and Yan, Yukun and Wang, Shuo and Yang, Cheng and Gu, Yu and Yu, Ge and Sun, Maosong},
    journal = {ArXiv preprint},
    title = {MetaMem: Evolving Meta-Memory for Knowledge Utilization through Self-Reflective Symbolic Optimization},
    volume = {abs/xxxx.xxxxx},
    year = {2026}
}
```

## 📧 Contact

For questions, suggestions, or bug reports, please contact:

```
xinhaidong@stumail.neu.edu.cn
```