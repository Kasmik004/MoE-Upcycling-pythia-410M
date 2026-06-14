# Sparse MoE-Upcycling-pythia-410M


This repository contains the Jupyter Notebook `moe-upcycling-with-metrics (2).ipynb`, which implements, trains, and analyzes a **Dense-to-MoE (Mixture of Experts) upcycling pipeline**.

The project demonstrates how to convert a pretrained dense language model into a sparse MoE model and evaluate its behavior using advanced routing diagnostics and expert specialization metrics.

---

## Overview

The pipeline transforms a pretrained **Pythia-410M** dense model into an upcycled **Mixture of Experts (MoE)** model.

Specifically, the implementation targets transformer layers **6–9**, replacing their standard feed-forward networks (FFNs) with custom **MoE Transformer Blocks**. Each MoE block consists of:

* **4 parallel experts**
* Experts initialized from the original dense FFN weights
* Small noise perturbations to encourage specialization
* A **Top-2 router** responsible for expert selection

This approach preserves the knowledge of the original dense model while enabling sparse computation and expert diversification.

---

## Dataset

The model is fine-tuned and evaluated using the **WikiText** dataset.

* Dataset: WikiText
* Hugging Face Dataset: https://huggingface.co/datasets/Salesforce/wikitext

---

## Key Features

### Architectural Upcycling

* Replaces dense FFN modules in transformer layers **6, 7, 8, and 9** with custom `MoELayer` implementations.
* Initializes each expert as a copy of the original dense FFN.
* Applies small perturbations to expert weights to promote divergence and specialization.

### Top-2 Routing Mechanism

* Uses a learnable router to select the top two experts for each token.
* Enables sparse activation while maintaining robust token representations.
* Incorporates auxiliary load-balancing objectives to prevent expert collapse.

### Efficient Sequence Packing

* Concatenates text streams before tokenization.
* Segments sequences into fixed-length chunks of **512 tokens**.
* Maximizes GPU utilization and minimizes padding overhead during training.

### Dual-Axis Training Analytics

The training pipeline simultaneously tracks:

* Language Modeling (LM) Loss
* Total Loss (LM Loss + Auxiliary Routing Loss)
* Validation Loss
* Validation Perplexity

### Comprehensive Routing Diagnostics

The notebook provides detailed insight into expert utilization through metrics such as:

#### Router Dynamics

* **Normalized Entropy**

  * Measures how evenly routing probabilities are distributed.

* **Coefficient of Variation (CV)**

  * Quantifies imbalance in expert assignments.

* **Routing Confidence**

  * Captures the certainty of router decisions.

* **Dead Expert Fraction**

  * Tracks experts receiving little to no traffic.

#### Representation Divergence

* Computes cosine-distance-based divergence between corresponding expert weight projections.
* Monitors how experts evolve from their shared initialization.
* Provides evidence for the emergence of expert specialization over training.

---

## Training Procedure

The notebook performs the following steps automatically:

1. Download the pretrained **Pythia-410M** checkpoint.
2. Download and preprocess the **WikiText** dataset.
3. Convert selected dense transformer blocks into MoE blocks.
4. Initialize experts from dense FFN weights.
5. Fine-tune the upcycled MoE model.
6. Collect routing and specialization diagnostics throughout training.
7. Evaluate the final model and generate validation summaries.

By default, training executes for **320 optimization steps** (or the number configured within the notebook).

---

## Dependencies

Ensure that your environment has access to a CUDA-enabled GPU and install the required packages:

```bash
pip install torch transformers datasets accelerate tabulate
```

---

## Running the Notebook

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Configure the Environment

Make sure that:

* CUDA drivers are installed.
* A compatible GPU is available.
* The required dependencies have been installed.

### 3. Launch Jupyter

```bash
jupyter notebook
```

Open:

```text
moe-upcycling-with-metrics (2).ipynb
```

### 4. Run All Cells

Executing the notebook will automatically:

* Download the Pythia-410M model checkpoint,
* Download and preprocess WikiText,
* Perform Dense-to-MoE upcycling,
* Train the model,
* Track routing diagnostics,
* Evaluate validation performance, and
* Produce a post-training summary of expert behavior and model quality.

---

## Outputs

After training, the notebook provides:

* Training and validation loss curves,
* Validation perplexity measurements,
* Router utilization statistics,
* Expert load-balancing diagnostics,
* Dead expert analysis,
* Expert specialization/divergence metrics, and
* A final evaluation summary of the upcycled MoE model.

---

## Citation

If you use this repository in academic work, please cite the original resources used in this implementation:

* **Pythia**: Biderman et al., *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*.
* **WikiText**: Merity et al., *Pointer Sentinel Mixture Models*.
* Relevant literature on **Mixture of Experts upcycling and sparse transformer architectures**.
