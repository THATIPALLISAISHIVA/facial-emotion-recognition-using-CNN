# Facial Expression Recognition Using CNN

## 📌 Project Overview
This project implements a **Deep Convolutional Neural Network (DCNN)** to recognize facial expressions from images. It focuses on classifying three specific emotions: **Happiness**, **Sadness**, and **Neutral**, using the **FER-2013** dataset.

The model is designed to be robust and efficient, utilizing techniques like **Batch Normalization**, **ELU Activation**, and **Dropout** to prevent overfitting and improve generalization.

## 📊 Dataset: FER-2013
The project uses the **Facial Expression Recognition Challenge 2013 (FER-2013)** dataset from Kaggle.

- **Source**: [Kaggle FER-2013 Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- **Image Properties**: 48x48 pixels, Grayscale.
- **Total Images**: 35,887
- **Filtered Subset Used**: 21,264 images
- **Classes**:
  - Original: 7 (Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral)
  - **Used**: 3 (Happiness, Sadness, Neutral)
- **Data Split**:
  - **Training**: 90% (19,137 images)
  - **Validation**: 10% (2,127 images)

## 🏗️ Model Architecture
The model is a custom **Deep CNN** with the following structure:

### **Input Layer**
- Shape: `(48, 48, 1)`

### **Convolutional Block 1**
- `Conv2D` (64 filters, 5x5 kernel) + `ELU` + `BatchNorm`
- `Conv2D` (64 filters, 5x5 kernel) + `ELU` + `BatchNorm`
- `MaxPooling2D` (2x2)
- `Dropout` (0.4)

### **Convolutional Block 2**
- `Conv2D` (128 filters, 3x3 kernel) + `ELU` + `BatchNorm`
- `Conv2D` (128 filters, 3x3 kernel) + `ELU` + `BatchNorm`
- `MaxPooling2D` (2x2)
- `Dropout` (0.4)

### **Convolutional Block 3**
- `Conv2D` (256 filters, 3x3 kernel) + `ELU` + `BatchNorm`
- `Conv2D` (256 filters, 3x3 kernel) + `ELU` + `BatchNorm`
- `MaxPooling2D` (2x2)
- `Dropout` (0.5)

### **Fully Connected Layers**
- `Flatten` (Output: 9,216 features)
- `Dense` (128 units) + `ELU` + `BatchNorm`
- `Dropout` (0.6)
- `Dense` (3 units) + `Softmax` (Output Layer)

### **Parameter Count**
- **Total Parameters**: 2,395,075
- **Trainable Parameters**: 2,393,027
- **Non-trainable Parameters**: 2,048

## 🛠️ Technology Stack
- **Language**: Python
- **Deep Learning Framework**: TensorFlow / Keras
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn, Scikit-plot
- **Environment**: Jupyter Notebook

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/THATIPALLISAISHIVA/facial-emotion-recognition.git
cd facial-emotion-recognition
```

### 2. Install Dependencies
Ensure you have Python installed. Install the required libraries using:
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
1. Go to the [Kaggle FER-2013 Dataset page](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
2. Download `fer2013.csv`.
3. Place the dataset file in the project directory (or update the path in the notebook).

### 4. Run the Project
Open the Jupyter Notebook to train and evaluate the model:
```bash
jupyter notebook "facial-emotion-recognition.ipynb"
```

## 📈 Results
The model is trained on the filtered dataset to distinguish between happy, sad, and neutral expressions. The architecture involves heavy regularization (Dropout, Batch Normalization) to handle the variability in facial expressions effectively.

