# **Brain Tumor MRI Image Classification using Deep Learning**

##  **Project Overview**

This project focuses on **classifying brain tumors** from MRI images into four categories:  
- **Glioma**  
- **Meningioma**  
- **Pituitary Tumor**  
- **No Tumor**

Using advanced **deep learning models** like **Convolutional Neural Networks (CNN)**, **DenseNet121**, and **EfficientNet-B0**, the aim is to evaluate their performance using key metrics such as **accuracy**, **loss**, and **ROC-AUC curves**. 

The project involves:  
- **Dataset Preparation and Preprocessing**  
- **Model Training and Evaluation**  
- **Performance Comparison and Hyperparameter Tuning**  
- **Real-Time Image Prediction**
---
##  **Dataset Details**

- **Source:** Kaggle  
- **Categories:** Glioma, Meningioma, Pituitary, No Tumor  
- **Train-Test Split:** Training and Testing subsets  
- **Image Resolution:** 224x224 pixels  

**Preprocessing Steps:**  
- **Normalization:** Pixel values scaled to `[0, 1]`  
- **Augmentation:** Rotation, width/height shift, shear, zoom, horizontal flip  

---

##  **Project Workflow**

1. **Data Preprocessing:**  
   - Dataset extraction  
   - Image normalization and augmentation  
   - Visualization of dataset class distribution and image properties  

2. **Model Building:**  
   - **CNN Architecture**  
   - **DenseNet121 (Pre-trained on ImageNet)**  
   - **EfficientNet-B0 (Pre-trained on ImageNet)**  

3. **Training and Tuning:**  
   - Early Stopping  
   - Learning Rate Reduction on Plateau  

4. **Evaluation Metrics:**  
   - Accuracy  
   - Validation Loss  
   - ROC-AUC Curves  

5. **Prediction on New Images:**  
   - Real-time classification of uploaded MRI images  

---

## **Technologies Used**

- **Python 3.x**  
- **TensorFlow / Keras**  
- **NumPy**  
- **Matplotlib**  
- **Seaborn**  
- **Scikit-learn**  
- **Pillow (PIL)**  
- **Google Colab (Optional for cloud execution)**  

---

## **How to Run the Project**

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
```

### **2. Install Required Libraries**
```bash
pip install tensorflow keras numpy matplotlib seaborn scikit-learn pillow
```

### **3. Place the Dataset**
- Ensure your dataset (`archive.zip`) is in the project directory.

### **4. Run the Script**
```bash
python brain_tumor_mri_image_classification.py
```

### **5. Test with an Image**
Update the `img_path` variable in the script:
```python
img_path = '/content/test_subset/glioma/Te-gl_0018.jpg'
```

Run the script again to see the predicted class and the actual class.

---

##  **Model Architectures**

### **1. CNN (Convolutional Neural Network)**
- Multiple convolutional and pooling layers  
- Dense layers for final classification  
- Dropout layers for overfitting prevention  

### **2. DenseNet121**
- Pre-trained on **ImageNet**  
- Dense connectivity between layers  
- Fine-tuned top layers for classification  

### **3. EfficientNet-B0**
- Pre-trained on **ImageNet**  
- Compound scaling for better efficiency  
- Custom layers for final classification  

---

##  **Results and Observations**

- The **CNN model** delivered the **best results**, with an accuracy of **89.29%**. Its consistent performance across training and validation data highlights its ability to extract meaningful features from MRI images.

- **DenseNet121**, while showing promise with an accuracy of **81.19%**, struggled with class imbalance. Despite leveraging pre-trained weights, it required more extensive fine-tuning for optimal performance.

- **EfficientNet-B0** underperformed significantly, achieving only **25% accuracy**. The architecture, though powerful in general image classification tasks, did not adapt well to the MRI dataset, indicating a need for further optimization.

- **ROC-AUC curves** across all models revealed challenges in achieving robust predictive power, suggesting that additional work is needed on data augmentation, preprocessing, and hyperparameter tuning.

**Key Insight:**  
**CNN architecture is the most suitable for brain tumor classification tasks based on this dataset, offering a good balance of accuracy and computational efficiency.**

---

##  **How to Make Predictions on New Images**

1. Place your MRI image in the project folder.  
2. Update the file path in the script:
```python
img_path = '/content/test_subset/glioma/Te-gl_0018.jpg'
```
3. Run the prediction code:
```python
prediction = cnn_model.predict(img_array)
```
4. The output will display:  
   - **Actual Class:** `Glioma`  
   - **Predicted Class:** `Glioma`

---

##  **Future Improvements**

1. Implement **Ensemble Models** to combine predictions from CNN, DenseNet121, and EfficientNet-B0.  
2. Utilize **Advanced Data Augmentation Techniques** for better generalization.  
3. Train the models on a **larger and more diverse dataset**.  
4. Optimize **EfficientNet-B0** with customized preprocessing and fine-tuning.  
5. Experiment with **Transfer Learning on Medical Datasets**.  

---


