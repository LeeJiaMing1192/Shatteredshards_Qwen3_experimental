# Shatteredshards Qwen3 Experimental

An experimental research project exploring architectural modifications to **Qwen3-based models**, with the goal of improving performance while preserving or enhancing reasoning ability.

This repository includes multiple experimental variants (**1.7B, 4B**), custom Transformer components, dynamic-KV memory mechanisms, and benchmarking utilities.

---

## Overview

Shatteredshards Qwen3 Experimental is a collection of **heavily-modified Qwen3 model variants** designed for research on model architecture, memory systems, and reasoning dynamics.

The project experiments with:

* **Custom Transformer block designs**
* **Dynamic key-value caching mechanisms**
* Modified activation paths
* Structural variations targeting improved **inference efficiency**
* Benchmarks intended to measure both performance and **reasoning strength**

> ⚠️ The repository is strictly experimental and intended for research, prototyping, and exploration of novel architectural ideas.

---

## Repository Structure

| Directory | Description |
| :--- | :--- |
| `/Transformer_Lib_architecture` | Core experimental **Transformer components**, modified block structures, and architectural extensions beyond standard Qwen3. |
| `/qwen-dynamic-kv-1.7B` | A **1.7B-parameter variant** integrating a **dynamic KV cache system** for more efficient inference and long-context behavior. |
| `/Shatteredshards_4B_experimental` | A **4B-parameter version** of the architecture with expanded capacity for reasoning and structured tasks. |
| `/benches_code` | Benchmarking scripts for measuring speed, throughput, and reasoning-related evaluation metrics. |
| `/visualizations` | Plots, charts, or model diagrams used during development. |

---

## Features

### Dynamic Key-Value Memory

An experimental memory mechanism aiming to improve **cache efficiency** and contextual retention across long sequences.

### Modified Transformer Architecture

Custom architectural adjustments—activation patterns, normalization variants, attention structure changes—designed to explore **performance–reasoning trade-offs**. 

### Multiple Model Sizes

Includes experiments at both **1.7B** and **4B** parameter scales.

### Benchmarking

Tools for evaluating:

* **Inference speed**
* Memory footprint
* **Reasoning-related metrics** (e.g., MR-style internal reasoning evaluations)

---

## 🛠️ Installation

```bash
git clone [https://github.com/LeeJiaMing1192/Shatteredshards_Qwen3_experimental.git](https://github.com/LeeJiaMing1192/Shatteredshards_Qwen3_experimental.git)
cd Shatteredshards_Qwen3_experimental
pip install -r requirements.txt   



Dependencies generally include:

PyTorch

Transformers (modified local version)

Python scientific stack (numpy, matplotlib, etc.)

Usage
Each experimental branch or model folder contains its own runnable scripts.

Typical Flow:

Choose the architecture variant (dynamic KV 1.7B or 4B).

Load or provide model weights.

Run the corresponding script inside /benches_code to benchmark performance.

Use the modified Transformer components in /Transformer_Lib_architecture for custom training or inference setups.

Research Goals
This repository aims to explore questions such as:

How do architectural perturbations influence reasoning depth?

Can dynamic KV structures reduce compute without harming performance?

What tradeoffs appear when scaling modified architectures?

Can “thinking” metrics (MR-style scores) be preserved under aggressive performance optimizations?

The project is intended as a sandbox for architectural innovation, comparative studies, and exploratory training runs.

⚠️ Status
This project is experimental and under active development. Documentation, stability, and API consistency are not guaranteed.

🧑‍💻 Contributors
Name,Role
Lee Jia Ming,Project Lead
Vo Ngoc Khang,Collaborator

