# Brain-tumor-MRI-Image-classification
Overview
This project aims to classify brain tumors using MRI images with state-of-the-art deep learning architectures: CNN (Convolutional Neural Network), DenseNet121, and EfficientNet-B0. The models predict four types of brain conditions:

Glioma
Meningioma
Pituitary Tumor
No Tumor
The dataset is sourced from Kaggle and undergoes preprocessing, augmentation, and model evaluation using metrics such as accuracy, loss, and ROC-AUC curves.

Table of Contents
Dataset Description
Project Workflow
Requirements
How to Run
Model Architectures
Evaluation Metrics
Results
Conclusion
Future Improvements
References
Dataset Description
The dataset is split into Training and Testing sets.
Classes: glioma, meningioma, nontumor, pituitary.
Images are resized to 224x224 pixels for uniform input dimensions.
Preprocessing Steps:

Normalization: Pixel values scaled to [0, 1].
Augmentation: Rotation, width/height shift, shear, zoom, and horizontal flip.
Project Workflow
Data Loading and Preprocessing:

Extract dataset.
Normalize and augment images.
Model Building:

CNN Model
DenseNet121 Model
EfficientNet-B0 Model
Training:

Use EarlyStopping and ReduceLROnPlateau callbacks.
Evaluation:

Evaluate accuracy, loss, and ROC-AUC curves.
Prediction:

Classify new MRI images.
Optimization:

Fine-tune hyperparameters and handle class imbalance.
Requirements
Ensure you have the following libraries installed:

bash
Copy code
pip install tensorflow keras numpy matplotlib seaborn scikit-learn pillow
Hardware Recommendations:
GPU: NVIDIA CUDA-enabled GPU (Recommended)
RAM: 16GB or more
How to Run
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
Place your dataset (archive.zip) in the drive folder.

Run the script:

bash
Copy code
python brain_tumor_mri_image_classification.py
To test a specific image:

python
Copy code
img_path = '/content/test_subset/glioma/Te-gl_0018.jpg'
Results will be displayed with the actual and predicted tumor type.

Model Architectures
1. Convolutional Neural Network (CNN)
Layers: Conv2D, MaxPooling2D, Flatten, Dense, Dropout
Optimizer: Adam
Loss Function: Categorical Crossentropy
2. DenseNet121
Pretrained Weights: ImageNet
Custom Layers: GlobalAveragePooling2D, Dense, Dropout
Optimizer: Adam
3. EfficientNet-B0
Pretrained Weights: ImageNet
Custom Layers: GlobalAveragePooling2D, Dense, Dropout
Optimizer: Adam
Evaluation Metrics
Accuracy: Measures the percentage of correct predictions.
Loss: Measures prediction error.
ROC-AUC Curve: Evaluates multi-class predictive performance.
Results
Model	Accuracy (%)	Validation Loss	ROC-AUC
CNN	89.29	0.35	0.85
DenseNet121	81.19	0.50	0.78
EfficientNet-B0	25.00	1.20	0.55
Conclusion
The CNN model outperformed DenseNet121 and EfficientNet-B0 in accuracy and overall reliability.
DenseNet121 showed moderate performance with some limitations on class imbalance.
EfficientNet-B0 struggled to adapt to the dataset despite pre-trained weights.
The project highlights the potential of CNN architectures in medical image classification and emphasizes the importance of hyperparameter tuning and dataset quality.

Future Improvements
Implement ensemble models combining CNN, DenseNet, and EfficientNet.
Use a larger and more balanced dataset for improved generalization.
Apply advanced data augmentation techniques.
Explore transfer learning with more specialized architectures.
