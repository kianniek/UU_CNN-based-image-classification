# TODO
## Contributers: [Kian Hamidi](https://github.com/kianniek) & [Vinn Majarocon](https://github.com/Veerific/)

> To install all dependencies please run:
> ```bash
> pip install -r requirements.txt
> ```
>
> ---
>
> **Setup Guide — Run this project reliably**
>
> - **Prerequisites:** Python 3.8+ recommended. Verify with:
> ```bash
> python --version
> pip --version
> ```
>
> - **Install Python (if missing):**
>   - Windows: Install from https://python.org and check "Add Python to PATH".
>   - macOS: `brew install python` or use the official installer.
>   - Linux: `sudo apt install python3 python3-venv python3-pip` (Debian/Ubuntu).
>
> - **Clone or place the project root** so you can see `requirements.txt` and `main.py`.
>
> - **Create and activate a virtual environment (recommended):**
>   - Windows (PowerShell):
> ```powershell
> python -m venv venv
> .\venv\Scripts\Activate.ps1
> ```
>   - Windows (cmd):
> ```cmd
> python -m venv venv
> venv\Scripts\activate
> ```
>   - macOS / Linux:
> ```bash
> python3 -m venv venv
> source venv/bin/activate
> ```
>
> - **Upgrade pip and install dependencies:**
> ```bash
> pip install --upgrade pip setuptools wheel
> pip install -r requirements.txt
> ```
> If `torch` fails, follow the instructions at https://pytorch.org/get-started/locally to install the wheel that matches your OS, Python and CUDA.
>
> - **Prepare CIFAR dataset (if missing):** the repo includes `data/cifar-10-batches-py/`. To download automatically:
> ```bash
> python - <<'PY'
> from torchvision import datasets
> datasets.CIFAR10(root='data', train=True, download=True)
> datasets.CIFAR10(root='data', train=False, download=True)
> print('CIFAR-10 downloaded to data/')
> PY
> ```
> Or see `src/data_loader.py` for dataset helpers.
>
> - **Quick smoke-run:**
> ```bash
> python main.py
> ```
> Check `main.py` or `src/train.py` for available CLI args (epochs, batch size, train/eval).
>
> - **GPU check:** in Python run:
> ```python
> import torch
> print(torch.__version__, torch.cuda.is_available())
> ```
> If CUDA is not available but you have an NVIDIA GPU, install the appropriate CUDA-enabled `torch` wheel from the PyTorch site and ensure NVIDIA drivers are installed.
>
> - **Run tests (optional):**
> ```bash
> pip install -r requirements.txt
> pytest -q
> ```
>
> - **Troubleshooting tips:**
>   - Permission issues: try an elevated shell or `pip install --user ...`.
>   - Missing `venv` module: install `python3-venv` (Linux).
>   - Torch install errors: clear pip cache (`pip cache purge`) and re-run the PyTorch selector install command.
>   - Dataset errors: confirm `data/` path or update data path in `src/data_loader.py`.
>
> If you want, I can add this guide as a new README section and open a PR with the change.

---

### 🏗️ Phase 1: Foundation & Baseline (CIFAR-10)

* [x] **[Kian]** **Data & Init:** Implement CIFAR-10 loading (80/20 split) and apply `torch.nn.init.kaiming_uniform` to all models.
* [ ] **[Vinn]** **Baseline Architecture:** Build **CIFAR10_lenet**. *Constraint: 3-channel, 32x32x3, no zero-padding, Max/Avg Pool.*
* [x] **[Kian]** **Training Engine:** Create the script with Adam ($\text{LR}=0.001$, Batch 32) and Cross-Entropy loss.
* [ ] **[Vinn]** **Model Evolution:** Create **Model 1** (1 change from LeNet) and **Model 2** (1 change from Model 1). Document the "Why" for each.
* [x] **[Kian]** **Choice 1 (5 pts):** Implement LR Scheduler (halve every 5 epochs). *Required: Plot LR decay vs. Epochs.*
* [x] **[Kian]** **Choice 5 (5 pts):** Add Data Augmentation (3+ techniques). *Required: Compare performance with vs. without augmentation.*

---

### 🧪 Phase 2: Advanced Validation & Optimization

* [ ] **[Kian]** **Choice 2 (10 pts):** Implement **5-fold Cross-Validation** for the baseline. Compare results to the fixed 80/20 split.
* [ ] **[Kian]** **Choice 3 (10 pts):** Perform **Evolutionary Hyperparameter Search**.
* Evaluate 3 Optimizers, 3 LRs, 2 Weight Decays, and 2 Batch Sizes.


* [ ] **[Vinn]** **Choice 4 (15 pts):** Add **Auxiliary Output Layers** to Conv/Pool layers.
* *Required: Extract and explain class predictions from these intermediate stages.*


* [ ] **[Vinn]** **Choice 6 (10 pts):** Generate **t-SNE visualizations** of the final FC layer.
* *Required: Analyze clusters vs. expected class confusions.*



---

### 🔄 Phase 3: Transfer Learning (CIFAR-100)

* [ ] **[Kian]** **CIFAR-100 Prep:** Load CIFAR-100 and adapt the "Best Architecture" for **20 class outputs**.
* [ ] **[Kian]** **Scratch Training:** Train **CIFAR100_model** until convergence using original hyperparameters.
* [ ] **[Vinn]** **Fine-Tuning:** Revert to 10 outputs; fine-tune on CIFAR-10 at half speed ($\text{LR}=0.0005$).
* [ ] **[Vinn]** **Benchmarking:** Final Test of "Scratch Best Model" vs. **CIFAR10_pretrained**. *Required: Confusion Matrices for both.*

---

### 🌍 Phase 4: Cross-Dataset Expansion (Tiny ImageNet)

* [ ] **[Vinn]** **Choice 7 (15 pts):** Load **Tiny ImageNet**, filter for classes overlapping with CIFAR-10, and resize to 32x32.
* [ ] **[Vinn]** **Evaluation:** Test your best model on this new data. *Required: Accuracy + Confusion Matrix.*
* [ ] **[Kian]** **Choice 8 (5 pts):** Fine-tune the best CIFAR-10 model on these Tiny ImageNet overlapping classes. Compare against Choice 7.

---

### 📝 Phase 5: Final Report & Delivery

* [ ] **[Vinn]** **Visuals & Tables:** Loss/Acc graphs (all models), LR graph, t-SNE plot, and the Top-1 Accuracy summary table.
* [ ] **[Vinn]** **Architectural Discussion:** Write the pair-wise comparisons (Baseline $\rightarrow$ M1 $\rightarrow$ M2) and explain the auxiliary output findings.
* [ ] **[Kian]** **Logic Justification:** Explain the choice of 80/20 split, the data augmentation impact, and the hyperparameter search results.
* [ ] **[Both]** **Final Polish:** Ensure the report is 2–5 pages and covers the "Generalization" discussion (Train vs. Val vs. Test performance).

---

# CNN Architectural Study: CIFAR-10 & CIFAR-100

This repository contains a structured experimental study on Convolutional Neural Networks (CNNs) using **PyTorch**. The project transitions from a classic **LeNet-5** baseline to optimized variants, incorporating transfer learning and rigorous validation strategies.

---

## 🎯 Project Goals
* **Model Design:** Recreating and modifying CNN architectures for color image classification.
* **Optimization:** Comparing the impact of structural changes (pooling, dropout, layers) on model convergence.
* **Transfer Learning:** Evaluating how pre-training on **CIFAR-100** influences performance on **CIFAR-10**.



---

## 📊 Dataset & Setup
The experiments are conducted on the **CIFAR** datasets ($32 \times 32$ images):
* **Training Set:** 50,000 images, further split into a custom **Validation set** (motivated by a specific ratio) to prevent test-set leakage during tuning.
* **Test Set:** 10,000 images used strictly for final benchmarking and confusion matrix generation.



---

## 🏗️ Architecture Evolution
1.  **Baseline (CIFAR10_lenet):** Standard LeNet-5 adapted for 3-channel input, utilizing `kaiming_uniform` initialization and the Adam optimizer.
2.  **Variants (Model 1 & 2):** Iterative improvements where only one structural aspect is changed per version to isolate performance drivers.
3.  **Pre-trained (CIFAR10_pretrained):** Best architecture trained on CIFAR-100 (20 classes) and fine-tuned for CIFAR-10 with a reduced learning rate.

---

## 🛠️ Requirements & Installation
This project requires **Python 3.x** and the following libraries:
* `torch` & `torchvision`
* `opencv-python` (for specific preprocessing tasks)
* `matplotlib` & `numpy`

### Running the Code
Ensure you have a CUDA-enabled GPU for significant speed-up. If running on **Google Colab**, ensure the GPU runtime is active.

# Example initialization in your script
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

📈 Evaluation Metrics
The final report (included in the docs/ folder) details:
 * Training/Validation Curves: Loss and accuracy plotted against epochs.
 * Confusion Matrices: Visualizing class-wise performance and common misclassifications.
 * Top-1 Accuracy: A comparative table across all five models.
