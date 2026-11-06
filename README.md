# PLANT-DISEASE-DETECTION-WITH-EXPLAINABILITY
# 🌿 Plant Disease Detection with Explainable AI

**A Deep Learning-Based Plant Disease Classification System Using CNN, Transfer Learning, and Grad-CAM**

A computer vision project developed using Python, TensorFlow, and Keras to identify plant diseases from leaf images. The project implements and compares a custom Convolutional Neural Network (CNN), ResNet50, and MobileNetV2, and explores Explainable AI using Grad-CAM to visualize regions that influence model predictions.

The project uses the PlantVillage dataset containing 38 plant disease and healthy-leaf categories and is implemented in a Kaggle Notebook.

---

## 📊 Project Highlights

- **Dataset:** PlantVillage
- **Dataset Source:** Kaggle
- **Total Images:** 54,305
- **Number of Classes:** 38
- **Best Model:** ResNet50
- **Best Recorded Test Accuracy:** 98.75%
- **Custom CNN Test Accuracy:** 97.22%
- **MobileNetV2 Test Accuracy:** 96.19%
- **Image Size:** 224 × 224
- **Framework:** TensorFlow / Keras
- **Explainability:** Grad-CAM
- **Development Environment:** Kaggle Notebook

---

## 📌 Project Overview

Plant diseases are a significant challenge in agriculture, affecting crop productivity, quality, and economic outcomes.

Identifying diseases at an early stage can help farmers make informed decisions about crop management and disease control.

Traditional disease identification often relies on manual inspection and agricultural expertise. However, visual symptoms such as leaf spots, discoloration, and damaged tissue can be difficult to distinguish between different disease categories.

This project explores the application of deep learning and computer vision to automatically classify plant diseases from leaf images.

The system learns visual features from plant leaf images and predicts the corresponding disease category.

In addition to classification, the project explores Explainable AI techniques to make model predictions more interpretable through visual heatmaps.

---

## 🎯 Problem Statement

Manual plant disease identification can be time-consuming and requires domain expertise.

Different plant diseases may exhibit similar symptoms, making accurate classification challenging.

The objective of this project is to develop a deep learning-based image classification system that can:

- Identify plant disease categories from leaf images.
- Learn visual patterns associated with plant diseases.
- Compare different deep learning architectures.
- Evaluate classification performance using independent test data.
- Visualize important image regions influencing predictions.
- Explore the potential of explainable plant disease classification.

---

## 💡 Proposed Solution

The proposed system uses deep learning-based image classification to identify plant diseases from leaf images.

The implementation follows a complete machine learning workflow:

1. Load the PlantVillage dataset.
2. Collect image paths and disease labels.
3. Perform exploratory data analysis.
4. Visualize the dataset and class distribution.
5. Split the dataset into training, validation, and testing subsets.
6. Apply image preprocessing and augmentation.
7. Build a custom CNN architecture.
8. Implement transfer learning using ResNet50 and MobileNetV2.
9. Train and evaluate the models.
10. Compare classification performance.
11. Generate Grad-CAM visualizations.
12. Analyze the model predictions and classification results.

---

## 📂 Dataset Source

### PlantVillage Dataset

The project uses the PlantVillage dataset, a publicly available collection of plant leaf images organized into disease and healthy-leaf categories.

**Dataset:** PlantVillage Dataset

**Kaggle Source:**  
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

**Dataset Publisher:** abdallahalidev

**Dataset Used:** Color images

### Dataset Statistics

| Property | Value |
|---|---|
| Dataset | PlantVillage |
| Total Images | 54,305 |
| Number of Classes | 38 |
| Image Type | RGB |
| Image Channels | 3 |
| Image Size Used | 224 × 224 |
| Data Format | Image files |
| Source | Kaggle |

### Dataset Directory

The notebook loads the dataset from:

```python
base_dir = "/kaggle/input/plantvillage-dataset/color"
```

The dataset is organized into class-specific directories.

Each directory represents a plant disease or healthy-leaf category.

The notebook traverses these directories, collects image paths, and assigns the corresponding directory name as the image label.

### Dataset Loading

The dataset is loaded using Python's `os` module and organized into a Pandas DataFrame.

The DataFrame contains:

| Column | Description |
|---|---|
| `filepaths` | Image file paths |
| `labels` | Corresponding plant disease category |

The dataset is then used for exploratory analysis, preprocessing, splitting, and model training.

**Dataset attribution:** The PlantVillage dataset is provided by its original publisher on Kaggle. Dataset ownership and licensing remain with the original contributors. Refer to the dataset page for applicable terms of use.

---

## 🧪 Exploratory Data Analysis (EDA)

The project includes exploratory data analysis to understand the dataset before model training.

### Implemented Analysis

- Dataset size and shape
- Number of disease categories
- Class distribution
- Random sample image visualization
- Image label inspection
- RGB channel visualization
- Dataset organization

### Dataset Information

The notebook records:

```text
Total Images: 54305
Total Classes: 38
```

The dataset is organized into 38 distinct categories.

EDA helps understand the distribution of images and provides an overview of the visual characteristics of the dataset.

---

## 🖼️ Image Preprocessing

The images are resized to a fixed resolution before being passed to the models.

### Image Configuration

| Parameter | Value |
|---|---|
| Image Height | 224 |
| Image Width | 224 |
| Channels | 3 |
| Image Format | RGB |

### Data Augmentation

The custom CNN training pipeline uses image augmentation to introduce variation into training images.

The implemented transformations include:

- Rotation
- Width shifting
- Height shifting
- Zoom
- Horizontal flipping
- Image rescaling

These transformations help the model learn from variations in image orientation and appearance.

Validation and test datasets are processed separately from the augmented training data.

### Dataset Splitting

The notebook uses stratified dataset splitting to divide images into training, validation, and testing subsets.

The intended split configuration is:

| Dataset | Percentage |
|---|---:|
| Training | 80% |
| Validation | 10% |
| Testing | 10% |

Stratification helps preserve the class distribution across the subsets.

---

## 🧠 Deep Learning Models

Three deep learning architectures are implemented and evaluated.

### 1. Custom CNN

The Custom CNN is developed using TensorFlow/Keras.

The architecture learns visual features directly from plant leaf images.

#### Architecture

```text
Input Image
224 × 224 × 3
       │
       ▼
Convolutional Block 1
Conv2D
Batch Normalization
Swish Activation
Conv2D
Batch Normalization
Swish Activation
MaxPooling
Dropout
       │
       ▼
Convolutional Block 2
Conv2D
Batch Normalization
Swish Activation
Conv2D
Batch Normalization
Swish Activation
MaxPooling
Dropout
       │
       ▼
Convolutional Block 3
Conv2D
Batch Normalization
Swish Activation
Conv2D
Batch Normalization
Swish Activation
MaxPooling
Dropout
       │
       ▼
Global Average Pooling
       │
       ▼
Dense Layer
256 Units
Batch Normalization
Swish Activation
Dropout
       │
       ▼
Softmax Output
38 Classes
```

#### Main Components

**Convolutional Layers**

Extract image features such as edges, textures, spots, and visual patterns.

**Batch Normalization**

Helps stabilize intermediate activations during training.

**Swish Activation**

Introduces non-linearity into the network.

**Max Pooling**

Reduces spatial dimensions while retaining important feature information.

**Dropout**

Helps reduce overfitting by randomly disabling activations during training.

**Global Average Pooling**

Aggregates spatial features before classification.

**Softmax Output**

Produces class probabilities across the 38 categories.

---

### 2. ResNet50

ResNet50 is a deep convolutional neural network architecture used in the transfer-learning experiments.

The model uses residual connections to support learning in deeper networks.

The implementation uses:

- ResNet50 architecture
- ImageNet-pretrained feature extraction
- ResNet-specific image preprocessing
- Transfer-learning classification
- Evaluation on the plant disease test dataset

ResNet50 achieved the highest recorded test accuracy among the three evaluated architectures.

**Recorded Test Accuracy: 98.75%**

---

### 3. MobileNetV2

MobileNetV2 is implemented as another transfer-learning architecture.

It uses an efficient convolutional architecture designed to reduce computational complexity.

The implementation includes:

- MobileNetV2 architecture
- Pretrained feature extraction
- MobileNetV2-specific preprocessing
- Classification on the plant disease dataset
- Test evaluation

**Recorded Test Accuracy: 96.19%**

MobileNetV2 provides an alternative architecture for comparing classification performance and computational efficiency.

---

## ⚙️ Model Training Configuration

The notebook uses TensorFlow/Keras for model development and training.

### Custom CNN Configuration

| Parameter | Value |
|---|---|
| Framework | TensorFlow / Keras |
| Input Shape | 224 × 224 × 3 |
| Batch Size | 32 |
| Maximum Epochs | 25 |
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Loss Function | Categorical Cross-Entropy |
| Label Smoothing | 0.1 |
| Evaluation Metric | Accuracy |
| Random Seed | 42 |

### Training Callbacks

The custom CNN training pipeline includes:

**EarlyStopping**

Monitors validation loss and stops training when the model stops improving.

**ReduceLROnPlateau**

Reduces the learning rate when validation loss reaches a plateau.

**ModelCheckpoint**

Saves the best model based on validation performance.

These callbacks support training stability and model selection.

---

## 📊 Model Performance & Accuracy

The project evaluates three deep learning architectures using the plant disease classification dataset.

### Recorded Test Accuracy

| Model | Test Accuracy |
|---|---:|
| Custom CNN | 97.22% |
| ResNet50 | **98.75%** |
| MobileNetV2 | 96.19% |

### Best Performing Model

**ResNet50 achieved the highest recorded test accuracy of 98.75%.**

The evaluation was performed using the held-out test dataset containing 5,431 images across 38 classes.

### Recorded ResNet50 Evaluation

| Metric | Result |
|---|---:|
| Test Images | 5,431 |
| Number of Classes | 38 |
| Test Accuracy | 98.75% |
| Test Loss | 0.7802 |

### Performance Summary

The recorded results show that ResNet50 achieved the highest classification accuracy among the three evaluated architectures.

The Custom CNN also demonstrated strong classification performance, while MobileNetV2 provided an alternative transfer-learning approach.

The comparison highlights the effectiveness of different deep learning architectures for plant disease image classification.

**Important:** These accuracy values represent the recorded notebook evaluation results. Performance on real-world agricultural images may differ from performance on the PlantVillage test dataset.

---

## 🔍 Explainable AI (XAI)

Explainable AI is incorporated to explore how the model interprets plant leaf images.

A classification model can predict a disease category, but understanding which image regions influence the prediction can provide additional insight into its behavior.

This project uses Grad-CAM (Gradient-weighted Class Activation Mapping) to generate visual explanations.

### Grad-CAM Workflow

```text
Plant Leaf Image
       │
       ▼
Trained CNN Model
       │
       ▼
Disease Prediction
       │
       ▼
Gradient Calculation
       │
       ▼
Grad-CAM Heatmap
       │
       ▼
Heatmap Overlay
       │
       ▼
Visual Explanation
```

### Grad-CAM Implementation

The notebook includes functions for:

- Identifying the last convolutional layer
- Creating a gradient model
- Calculating gradients
- Generating activation heatmaps
- Resizing heatmaps
- Applying heatmap overlays
- Visualizing original and predicted images

### Visualization Output

The notebook generates a visualization comparing:

1. Original plant leaf image
2. Grad-CAM heatmap
3. Heatmap overlay on the original image

The visualization also displays the predicted disease category and prediction confidence.

Grad-CAM provides a visual interpretation of the image regions influencing the model prediction.

**Note:** Heatmaps indicate regions associated with model predictions. They do not independently establish biological causation or guarantee that the model is focusing on disease-specific features.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | Model building and training |
| NumPy | Numerical operations |
| Pandas | Dataset organization |
| OpenCV | Image processing |
| PIL | Image handling |
| Matplotlib | Visualization |
| Seaborn | Exploratory data analysis |
| Plotly | Interactive visualization |
| Scikit-learn | Dataset splitting and evaluation |
| Kaggle Notebook | Development environment |

---

## 🚀 Project Workflow

```text
PlantVillage Dataset
        │
        ▼
Dataset Loading
        │
        ▼
DataFrame Construction
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Dataset Splitting
        │
        ▼
Image Preprocessing
        │
        ▼
Data Augmentation
        │
        ▼
┌──────────────────────────────┐
│        Model Training        │
├──────────────────────────────┤
│ Custom CNN                  │
│ ResNet50                   │
│ MobileNetV2                 │
└──────────────────────────────┘
        │
        ▼
Model Evaluation
        │
        ▼
Accuracy Comparison
        │
        ▼
Disease Classification
        │
        ▼
Grad-CAM Explainability
        │
        ▼
Visual Interpretation
```

---

## 📁 Project Structure

```text
plant-disease-detection-with-explainable-ai/
│
├── plant-disease-detection-with-explainable-ai-xai.ipynb
│
├── dataset/
│   └── PlantVillage Dataset
│
├── model_checkpoints/
│   └── best_custom_cnn.keras
│
├── visualizations/
│   ├── EDA visualizations
│   ├── Class distribution
│   └── Grad-CAM outputs
│
├── README.md
│
└── requirements.txt
```

The dataset is attached separately through Kaggle.

Model checkpoints and visualization outputs are generated during notebook execution.

---

## ▶️ How to Run the Project

### 1. Open the Notebook

Open the notebook in Kaggle:

`plant-disease-detection-with-explainable-ai-xai(1).ipynb`

### 2. Add the Dataset

Navigate to the notebook's Input section.

Add the PlantVillage dataset:

https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

Ensure the dataset directory is accessible.

The original notebook uses:

```python
base_dir = "/kaggle/input/plantvillage-dataset/color"
```

If Kaggle mounts the dataset under a different directory, update the path accordingly.

### 3. Configure the Environment

The project uses Python and TensorFlow/Keras.

Select an available compatible accelerator in Kaggle Notebook settings.

### 4. Execute the Notebook

Run the notebook cells in order:

1. Import libraries
2. Configure reproducibility
3. Load dataset
4. Construct DataFrame
5. Perform EDA
6. Split dataset
7. Configure image preprocessing
8. Build Custom CNN
9. Train and evaluate Custom CNN
10. Implement transfer learning
11. Evaluate ResNet50
12. Evaluate MobileNetV2
13. Compare model performance
14. Generate Grad-CAM visualizations

### 5. Review Results

After execution, review:

- Training accuracy
- Validation accuracy
- Test accuracy
- Test loss
- Classification report
- Confusion matrix
- Model comparison
- Grad-CAM visualizations

---

## 📌 Project Status

| Component | Status |
|---|---|
| Dataset loading | Implemented |
| DataFrame construction | Implemented |
| Exploratory data analysis | Implemented |
| Image visualization | Implemented |
| Class distribution analysis | Implemented |
| Dataset splitting | Implemented |
| Image preprocessing | Implemented |
| Data augmentation | Implemented |
| Custom CNN | Implemented |
| ResNet50 | Implemented |
| MobileNetV2 | Implemented |
| Model evaluation | Implemented |
| Accuracy comparison | Implemented |
| Grad-CAM functions | Implemented |
| Grad-CAM visualization | Implemented |

---

## 🔮 Future Improvements

- Evaluate the models using additional real-world plant images.
- Improve generalization to different lighting conditions and backgrounds.
- Compare additional deep learning architectures.
- Optimize inference speed and model size.
- Develop a user-friendly image upload interface.
- Build a web application for plant disease prediction.
- Explore mobile deployment.
- Integrate agricultural recommendations with disease classification.
- Improve explainability and validate model attention regions.
- Evaluate performance across individual plant disease categories.

---

## ⚠️ Limitations

- The dataset is limited to the visual conditions represented in PlantVillage.
- Real-world images may contain different backgrounds, lighting, and image quality.
- Similar disease symptoms may result in classification errors.
- High test accuracy does not guarantee equivalent performance in field conditions.
- Grad-CAM visualizations are interpretability aids, not proof of disease diagnosis.
- Agricultural decisions should be validated using appropriate expertise.

---

## 👨‍💻 Author

**Shafiqur Rahaman**

Computer Science and Engineering

Project: Plant Disease Detection with Explainable AI

---

## 📄 Disclaimer

This project is developed for educational, research, and experimentation purposes.

The system predicts plant disease categories from leaf images and is intended as an AI-assisted classification approach.

It should not be treated as a replacement for professional agricultural diagnosis or field inspection.

Dataset ownership and licensing remain with the original dataset publisher.

---

## 🙏 Acknowledgements

- PlantVillage dataset contributors
- Kaggle dataset publisher: abdallahalidev
- TensorFlow and Keras
- Python open-source community
- Contributors to computer vision and explainable AI research

---

## ⭐ Support

If you find this project useful, consider starring the repository and sharing feedback or suggestions.

**Built with Python, TensorFlow, Keras, Computer Vision, and Explainable AI.**
