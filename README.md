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
