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

# Project Workflow: CNN-based Image Classification

> To ensure training until **convergence**, the `--epochs` parameter is set to a high value (100+) to allow the built-in **Early Stopping** logic (which monitors validation loss) to stop the training at the optimal point.

## The Baseline
Train the standard LeNet-5 architecture to establish a performance benchmark on CIFAR-10.
```bash
python main.py --model simple --epochs 100
```

---

## Architecture Evolution
Iteratively improve the model by testing deeper or wider architectures.

### 1. Medium CNN
```bash
python main.py --model medium --epochs 100
```

### 2. Deep CNN (Best Architecture)
```bash
python main.py --model deep --epochs 100
```

---

## Choice Tasks

We have to choose the best performing model (e.g., `medium`) to optimize and validate your results.

### Choice 1: Learning Rate Scheduling
Train with the `StepLR` scheduler to visualize how decreasing the LR helps convergence.
```bash
python main.py --model medium --epochs 100 --scheduler-step 5 --scheduler-gamma 0.5
```

### Choice 2: 5-Fold Cross-Validation
Verify the stability of your model across different data splits.
```bash
python main.py --model medium --epochs 100 --kfold 5 --compare-kfold-split
```

### Choice 3: Hyperparameter Search
Automatically test different optimizers and learning rates to find the global optimum.
```bash
python main.py --model medium --hyperparameter-search --epochs 20
```

### Choice 5: Data Augmentation Comparison
Quantify the impact of transformations (flips, crops, etc.) on model generalization.
```bash
python main.py --model medium --epochs 100 --compare-augmentation
```

### Choice 6: t-SNE output
```bash
python main.py --model medium --test-model --kfold 0
```
---

## Phase 3: Transfer Learning
Leverage knowledge from a related dataset (CIFAR-100) to improve performance on CIFAR-10.

### 1. Pre-training on CIFAR-100
Train the model on the 20 superclasses of CIFAR-100.
```bash
python main.py --model cifar100 --epochs 150
```

### 2. Fine-tuning on CIFAR-10
Load the pre-trained weights and fine-tune on the original task with a smaller learning rate.
```bash
python main.py --model finetune --epochs 50 --lr 0.0005 --no-scheduler 
```

---

## Final Benchmarking
Run the final evaluation on the **held-out test set** to generate your official accuracy and confusion matrix for the report.
```bash
python main.py --model medium --test-model
```

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
