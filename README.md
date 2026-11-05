# GDG School: Machine Learning & Deep Learning Projects

Hands-on notebooks built during the **GDG School** workshop *"Fundamentals of Machine Learning and Deep Learning"*, organized by **GDG Algiers** (4 November 2026).

The repository walks through three classic problems, from classical machine learning on tabular data to neural networks and convolutional networks on images.

## Projects

| # | Notebook | Task | Model | Result |
|---|----------|------|-------|--------|
| 1 | [`1_pokemon_battle_simulator.ipynb`](notebooks/1_pokemon_battle_simulator.ipynb) | Predict the winner of a Pokémon battle | Random Forest (200 trees) | ~93.7% test accuracy |
| 2 | [`2-handwritten-classification.ipynb`](notebooks/2-handwritten-classification.ipynb) | Recognize handwritten digits (MNIST) | MLP (fully connected network) | ~94.6% test accuracy |
| 3 | [`3-cifar10-cnn-classification.ipynb`](notebooks/3-cifar10-cnn-classification.ipynb) | Classify 10 types of color images (CIFAR-10) | CNN with BatchNorm and Dropout | ~78.7% test accuracy |

### 1. Pokémon Battle Simulator
A full machine learning pipeline on the [Complete Pokémon dataset](https://www.kaggle.com/datasets/rounakbanik/pokemon) from Kaggle:
- Data profiling, cleaning (missing values, duplicates, outliers) and feature engineering
- Encoding of categorical labels and scaling of numerical values
- Generation of 50,000 simulated battles to build a labeled training set
- Random Forest classifier that predicts the winner from the two fighters' features
- A turn-based battle simulator to compare actual fights against the model's predictions

### 2. Handwritten Digit Recognition (MLP)
- Loads the MNIST dataset (70,000 grayscale images of 28x28 pixels)
- Normalizes pixels, then builds a Multi-Layer Perceptron with Keras (`Flatten` + 3 `Dense` layers)
- Trains with Adam and sparse categorical cross-entropy, then evaluates on unseen images
- Saves and reloads the trained model, then predicts a single image

### 3. CIFAR-10 Image Classification (CNN)
- Explores CIFAR-10 (60,000 color images of 32x32 pixels, 10 classes)
- Builds a CNN with three Conv, BatchNorm, Dropout and MaxPooling blocks followed by a dense classifier
- Trains with early stopping and learning-rate reduction
- Evaluates with a per-class classification report (precision, recall, F1-score) and runs inference on random test images

## Tech stack

Python, TensorFlow / Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, Jupyter / Google Colab.

## Run the notebooks

The notebooks are designed to run in **Google Colab** with no local setup (all libraries are preinstalled and datasets download automatically).

1. Open [colab.research.google.com](https://colab.research.google.com) and choose **File > Upload notebook**, or open a notebook from this repository directly via GitHub.
2. For notebooks 2 and 3, enable a GPU: **Runtime > Change runtime type > GPU**.
3. Run **Runtime > Run all**.

To run locally instead:

```bash
git clone https://github.com/<your-username>/gdg-school-ai.git
cd gdg-school-ai
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn kagglehub jupyter
jupyter notebook
```

## Repository structure

```
.
├── notebooks/
│   ├── 1_pokemon_battle_simulator.ipynb
│   ├── 2-handwritten-classification.ipynb
│   └── 3-cifar10-cnn-classification.ipynb
├── certificate/
│   └── GDGA-H6K9-AHA3.pdf
└── README.md
```

## Certificate

Certificate of Attendance for *Fundamentals of Machine Learning and Deep Learning*, issued by GDG Algiers on 4 November 2026 (ID: `GDGA-H6K9-AHA3`). See the [`certificate/`](certificate/) folder.

## Author

**Mohamed Amine Sellami**

## Acknowledgments

- [GDG Algiers](https://gdg.community.dev/) for the GDG School workshop
- [Rounak Banik](https://www.kaggle.com/rounakbanik) for the Pokémon dataset
- MNIST (Yann LeCun et al.) and CIFAR-10 (Krizhevsky, Hinton) datasets, loaded through Keras
