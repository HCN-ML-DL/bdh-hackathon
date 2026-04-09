# 🐉 Dragon Hatchling (BDH) – Hackathon Demo

## 🚀 Overview
This repository contains our hackathon implementation of Dragon Hatchling (BDH), a biologically inspired language modeling architecture that combines ideas from Transformers, state-space models, and brain-inspired graph dynamics.

Unlike traditional Transformer models, BDH focuses on sparse activations, local interactions, and Hebbian-style updates to enable more interpretable and dynamic reasoning.

---

## 🎯 Problem Statement
Modern language models struggle with:
- Incorporating new facts at inference time
- Maintaining interpretable internal representations
- Avoiding heavy reliance on retraining or fine-tuning

This project explores a BDH-style approach to dynamically update knowledge and answer questions about previously unseen facts.

---

## 💡 Approach
The BDH pipeline in this project follows:

1. Token / byte embedding  
2. Expansion to high-dimensional space  
3. ReLU-based sparsification  
4. Linear attention (causal)  
5. Hebbian-style gating mechanism  
6. Compression back to model dimension  
7. Output projection  

This allows sparse neuron-like activations and dynamic memory updates.

---

## 🧪 Demo Workflow

Step 1: Load trained BDH model  

Step 2: Inject new facts (unseen during training)

Example:
The CEO of NeuraCorp is Alice Zhang  
The capital of Zephyria is Windholm  

Step 3: Query the model  

Example:
Who is the CEO of NeuraCorp? → Alice Zhang  
What is the capital of Zephyria? → Windholm  

---

## 📂 Repository Structure

.
├── app.py                # Inference / demo script  
├── train.py              # Training pipeline  
├── bdh_model.py          # Model architecture  
├── requirements.txt      # Dependencies  
└── checkpoints/          # Model checkpoints (not included)  

---

## ⚙️ Setup

Install dependencies:

pip install -r requirements.txt

---

## ▶️ Run

python app.py

If needed:

python app.py --checkpoint checkpoints/bdh_final.pt

---

## 🏋️ Training

python train.py

---

## ⚠️ Model Checkpoints

Model weights are not included due to GitHub file size limits.

Place your checkpoint manually in:

checkpoints/bdh_final.pt

---

## 🧠 Key Concepts

- Sparse activations  
- Linear attention  
- Hebbian learning  
- State + graph hybrid reasoning  

---

## 🔥 Future Work

- Better demo UI  
- Transformer vs BDH comparison  
- Visualization of learned connections  
- Larger scale training  

---

## 🛠 Tech Stack

- Python  
- PyTorch  
- Custom BDH architecture  

---

## 👥 Team

Hackathon project

---
