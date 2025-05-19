# Quantum-Inspired Self-Attention in a Large Language Model<!-- omit in toc -->

:scroll: [Paper](https://arxiv.org/abs/your-paper-id)  &nbsp; :computer: [Usage](#using-qsa-in-practice)  &nbsp; :books: [Related Projects](https://github.com/Nikait/QISA)

**TL;DR:** We replace the classical self‑attention in GPT‑1 with a quantum‑inspired attention mechanism, achieving logarithmic compression of some attention layer, approx eight times lower cross-entropy loss compared to standard self-attention, and only a ~2.1 longer inference time.

> [!TIP]
> For a concise overview, see **Section Methods** (pp. X–Y) and **Fig. 1** in the [paper PDF](https://arxiv.org/pdf/your-paper-id.pdf).

## Using QSA in Practice

To integrate Quantum Self‑Attention (QSA) into your own GPT‑1 training or inference pipeline, you only need:

- [QSA.py](QSA.py): the core QSA layer implementation, including both training (slow) and inference (fast) branches.
- [main.py](main.py): end‑to‑end example for training and evaluation with Hydra-based configuration.
- [conf/config.py](conf/config.py): default hyperparameters and setup.

### Quick Install

bash
pip3 install hydra-core torch torchtext numpy torchquantum
pip3 install qiskit==0.46.2 qiskit-aer==0.13.3 qiskit-ignis==0.7.0 qiskit-terra==0.46.2



Then clone the repo or copy the relevant modules into your project:

bash
git clone https://github.com/Nikait/QSA.git
cd QSA

 

## Table of Contents

- [Using QSA in Practice](#using-qsa-in-practice)
  - [Quick Install](#quick-install)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
  - [Key Components](#key-components)
  - [Performance Highlights](#performance-highlights)
- [Speed Comparison](#speed-comparison)
- [Performance](#performance)
- [Set Up Environment](#set-up-environment)
  - [Software Dependencies](#software-dependencies)
  - [Dataset Preparation](#dataset-preparation)
- [Running the Code](#running-the-code)
  - [Code Overview](#code-overview)
  - [Training](#training)

---

## Overview

This repository provides:

- **Quantum Inspired Self‑Attention (QISA)**: v1, v2 and v3 implementations leveraging amplitude encoding and Pauli measurement for queries, keys, and values.
- **GPT-1 Integration**: Drop‑in replacement of multi‑head self‑attention heads in the GPT‑1 architecture.
- **Two Execution Modes**:
  - **Slow (Training)**: Step‑by‑step quantum simulation with TorchQuantum.
  - **Fast (Inference)**: Precomputed total unitary for rapid matrix‑vector application.

### Key Components

| Module                     | Description                                      |
| :------------------------- | :----------------------------------------------- |
| QSA.py                   | Defines QSA, ValueLayerSlow, ValueLayerFast |
| main.py                  | Training and evaluation launcher via Hydra        |
| conf/config.py           | Experiment configurations (model, data, training) |
| model.py                 | GPT-1 model wrapper integrating QSA layers        |
| dataset.py               | Shakespeare dataset loader and tokenizer         |
| utils/                   | Logging, metrics, and helper functions           |


### Performance Highlights

| Metric                  | CSA          | QISA v2          | QISA v3           | AQSA v3 (w/ precomp)               |
| ----------------------- | ------------ | ---------------- | ----------------- | ---------------------------------- |
| Params per head         | 3×d×dₕ       | O(log d)         | 2×d×dₕ + O(log d)  | 2×d×dₕ + O(log d)                  |
| Inference Time (T4 GPU) | 1×           | 46.7×            | 8.9×              | 2.1× (with 22.3× speed vs. QISA v2)|
| Cross‑Entropy Loss      | baseline     | improves         | improves          | improves                           |

## Speed Comparison

![Training / Inference Time per Batch](https://github.com/user-attachments/assets/95094ce5-47b1-4366-9c2a-557fe7f2f81d)

*Figure 1. Time per batch (batch size = 1024) on a single NVIDIA T4 GPU for CSA and different versions of QSA with embedding sizes $\{4, 16\}$. The fastest inference variant with unitaries and observables precomputation at QSAv3 achieves a 22.3$\times$ speed-up over the standard QSAv2 inference.*

## Performance
![loss](https://github.com/user-attachments/assets/23dbb511-da43-4224-af58-497d3062e313)

*Figure 2. Training cross-entropy loss: CSA vs. QISA. Setup: 1 epoch, batch size = 128, 1 head, context length = 16, embedding size = 128, 7 qubits.*

## Set Up Environment

### Software Dependencies

Tested on **Ubuntu 22.04**, Python 3.10+:

bash
pip3 install qiskit==0.46.2 qiskit-aer==0.13.3 qiskit-ibm-provider==0.10.0 qiskit-ibm-runtime==0.20.0 qiskit-ibmq-provider==0.19.0 qiskit-ignis==0.7.0 qiskit-terra==0.46.2 
pip3 install hydra-core torchquantum



> **Note:** For GPU acceleration, install a CUDA‑enabled PyTorch build.

### Dataset Preparation

1. Download the Shakespeare text dataset from Kaggle:
   

bash
   mkdir -p data && cd data
   wget https://www.kaggle.com/datasets/adarshpathak/shakespeare-text/download -O shakespeare.txt


2. Ensure conf/config.py points to data/shakespeare.txt and char_level tokenizer.



## Running the Code


### Training

Run a single training experiment with your config:

bash
python3 main.py



Outputs (loss) will be saved under logs.txt/.
Also you may add checkpoint saver by editing conf/config.py
