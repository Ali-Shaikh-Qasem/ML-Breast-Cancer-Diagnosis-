# Machine Learning Classification Methods

## Overview
This project explores various machine learning classification techniques to diagnose breast cancer using the **Breast Cancer dataset**. The goal is to implement, analyze, and compare different classification models based on performance metrics.

## Dataset
We use the **Breast Cancer dataset**, a well-known dataset for medical diagnosis, to classify tumors as benign or malignant. This dataset provides features derived from cell nuclei, helping train models to predict cancerous conditions.

## Tools and Libraries
The project utilizes the following Python libraries:
- **NumPy & Pandas** for data manipulation
- **Scikit-learn** for model implementation and evaluation
- **Matplotlib & Seaborn** for visualization

## Classification Models Implemented
### **1. K-Nearest Neighbors (KNN)**
- Implemented using different distance metrics (Euclidean, Manhattan, Cosine)
- Used cross-validation to find the optimal value of K
- Analyzed the impact of distance metrics on model performance

### **2. Logistic Regression**
- Applied L1 and L2 regularization techniques
- Compared results with KNN to evaluate model efficiency

### **3. Support Vector Machines (SVM)**
- Experimented with different kernels (Linear, Polynomial, RBF)
- Evaluated how kernel choice affects classification accuracy

### **4. Ensemble Methods**
- **Boosting (AdaBoost):** Improved weak models by iteratively adjusting weights
- **Bagging (Random Forest):** Combined multiple decision trees for robust predictions
- Compared the effectiveness of Boosting and Bagging

## Model Evaluation Metrics
Each model was assessed using:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Correct positive predictions vs. total predicted positives
- **Recall**: Ability to identify actual positives
- **F1-score**: Balance between precision and recall
- **ROC-AUC**: Measure of model separability between classes

## Key Findings
- The best **K** for KNN was determined using cross-validation.
- Regularization improved Logistic Regression's stability.
- **SVM with RBF kernel** provided high accuracy but required careful tuning.
- **Boosting outperformed Bagging** in handling complex decision boundaries.

## Conclusion
This project provides insights into the strengths and weaknesses of different classification methods for medical diagnosis. The findings emphasize the importance of model selection, hyperparameter tuning, and feature engineering in achieving high classification accuracy.

## Authors
This project was completed by:
- Ali Shaikh Qasem.
- Abdelrahman Jaber.


