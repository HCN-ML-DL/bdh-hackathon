# 🧠 BDH Inference-Time Learning (Hackathon Project)

## 🚀 Overview

This project demonstrates a **BDH (Brain-Driven Hebbian) language model** that can **learn during inference**, unlike traditional Transformer models.

The system compares:

* 🧠 **BDH model** → updates internal state dynamically using Hebbian learning
* 🤖 **GPT-2** → static weights, no learning during inference

A Streamlit interface allows real-time text generation and visualization of **state updates**.

---

## 🧠 Key Idea

Traditional models:

```
output = softmax(QKᵀ)V
# weights are frozen during inference
```

BDH model:

```
output = query @ state
state += keyᵀ @ value   # Hebbian update
```

👉 This enables:

* Learning new patterns **without retraining**
* Dynamic memory updates
* More biologically inspired computation

---

## ✨ Features

* ⚡ Inference-time learning via Hebbian updates
* 🧪 Side-by-side comparison with GPT-2
* 📊 Visualization of per-layer state changes
* 🧬 Sparse neuron-style activations
* 🔁 Linear attention (no softmax bottleneck)
* 🖥 Interactive Streamlit UI

---

## 📂 Project Structure

```
.
├── app.py              # Streamlit demo (inference + visualization)
├── train.py            # Training pipeline (optimized for L4 GPU)
├── bdh_model.py        # BDH architecture implementation
├── requirements.txt    # Dependencies
├── checkpoints/        # Model weights (not included)
└── bdh.ipynb           # Notebook experiments
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Demo

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 🧪 What the Demo Does

1. Enter a story prompt
2. BDH generates text **while updating internal state**
3. GPT-2 generates text **without learning**
4. Compare outputs + state changes

---

## 📊 Example Output

* BDH shows:

  * Layer-wise state updates
  * Total "learning magnitude"

* GPT-2 shows:

  * ❌ No updates (static model)

---

## 🏋️ Training

Train BDH model:

```bash
python train.py
```

### Training Details

* Dataset: HuggingFace dataset (TinyStories expected) 
* Byte-level tokenization
* Sequence length: 256
* Optimized for L4 GPU (TF32 enabled) 
* Gradient accumulation + mixed precision

---

## 🧠 Model Architecture

Defined in `bdh_model.py` 

### Core Components:

* Byte-level embedding (vocab=256)
* Multi-layer BDH blocks
* Rotary positional embeddings
* Hebbian state updates
* Sparse gated activations (~5%)

### Key Innovation:

* **State matrix evolves during generation**
* No KV-cache growth like transformers
* Memory is compressed + dynamic

---

## ⚠️ Checkpoints

Model weights are not included.

Place your trained model here:

```
checkpoints/bdh_final.pt
```

---

## 🛠 Tech Stack

* PyTorch
* Streamlit
* HuggingFace Transformers (GPT-2 baseline)
* Datasets library 

---

## 🔥 Future Work

* Larger scale training
* Better evaluation metrics
* Visualization of neuron activations
* Real-world knowledge injection tests

---

## 👥 Acknowledgment

Based on BDH architecture concepts from:
https://github.com/pathwaycom/bdh 

---

## 🧠 Key Takeaway

> BDH introduces a shift from static models to **adaptive systems that learn while generating**.

This project demonstrates a step toward **continual learning in LLMs without retraining**.
