# Building a Denoising Diffusion Model from Scratch (CPU-Only)

## 📖 About This Project
While I've spent significant time working with Generative AI—deploying LLMs, building RAG pipelines, and managing MLOps workflows—I realized I was treating diffusion models as a black box. I could use Hugging Face `diffusers` to generate images, but I didn't truly understand the foundational math and mechanics underneath.

So I challenged myself: **Can I build a Denoising Diffusion Probabilistic Model (DDPM) entirely from scratch using raw PyTorch?**

To make it even more interesting (and to prove I understand the core mechanics without relying on hardware brute-force), I restricted myself to training and sampling using **only my laptop's CPU**.

## 🧠 How It Works
This project implements the core DDPM framework. The model learns to generate images (trained on the MNIST dataset) through two main phases:

1. **The Forward Process (Diffusion):** We take a clean image and slowly add Gaussian noise over $T$ timesteps until it becomes pure static. The math relies on a precomputed noise schedule (alphas and betas).
2. **The Reverse Process (Denoising):** We train a neural network to predict the exact noise that was added at any given timestep $t$. During generation, we start with pure random noise and iteratively subtract the predicted noise to "sculpt" a clean image.

### The Network Architecture
I implemented a lightweight Convolutional U-Net. Since self-attention mechanisms scale quadratically and would freeze a CPU, I stuck to standard convolutions. The network is conditioned on the timestep $t$ using **Sinusoidal Positional Embeddings** (the same math used in Transformers), which are injected into the network via addition at various layers.

## 🛠️ The CPU-Only Constraints
Training diffusion models usually requires massive GPUs. To make this run on a CPU in a reasonable timeframe (a few minutes per epoch), I had to make specific architectural compromises:
- **Dataset:** MNIST (small 28x28 grayscale images).
- **Timesteps ($T$):** Reduced from the standard 1000 down to **100**.
- **No Attention:** Replaced heavy attention blocks with spatial convolutions.

## 🚀 How to Run
This project is structured as a step-by-step Jupyter Notebook for easy learning and experimentation.

### 1. Prerequisites
Ensure you have a minimal, CPU-only PyTorch installation to save disk space:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install Pillow matplotlib
