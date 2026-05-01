# Thyroid Recurrence Prediction

This project implements machine learning models to predict the recurrence of thyroid conditions in patients based on clinical data. It compares the performance of **Decision Tree**, **K-Nearest Neighbors (KNN)**, and **Random Forest** classifiers to determine the most effective model for this medical diagnostic task.

---

##  Project Overview

- **Objective:** Predict the target variable **Recurred (Yes/No)** based on patient attributes.  
- **Models Implemented:**  
  - Decision Tree Classifier  
  - K-Nearest Neighbors (KNN)  
  - Random Forest Classifier
- **Best Performing Model:**  
  - **Decision Tree** (Accuracy:  95.45% , Recall: 0.92 )  
- **Key Techniques:** Mode Imputation, Label Encoding, One-Hot Encoding, Feature Scaling (StandardScaler)

---

##  Dataset

The project uses the `filtered_thyroid_data.csv` dataset, which contains patient records with the following attributes:

- **Demographics:** Age, Gender  
- **Medical History:** Hx Radiotherapy, Adenopathy  
- **Pathology:** Pathology type, Focality, Risk  
- **Staging:** T, N, M, Stage  
- **Outcome:** Response, **Recurred (Target Variable)**

## Usage

Prepare Data: Ensure `filtered_thyroid_data.csv` is located in the data/ directory.

Run the Analysis: Execute the main script:
- `Thyroid.ipynb`

## Web Interface (Optional)

This project includes a **Streamlit web application** for easy data input and prediction. To run the web interface:

```bash
# Install Streamlit (if not already installed)
pip install streamlit

# Run the app
streamlit run app.py
```

The web interface provides:
- User-friendly input form for patient data
- Model selection (Decision Tree or Random Forest)
- Real-time prediction with probability breakdown
- Visual results display

# Results Summary

| Model         | Accuracy    | Precision (Recurred) | Recall (Recurred) | F1-Score |  AUC  |
| ------------- | ----------- | -------------------- | ----------------- | -------- | ----- |
| Decision Tree | 95.45%      | 0.96                 | 0.92              | 0.94     | 0.949 |
| KNN (K=5)     | 92.42%      | 1.00                 | 0.81              | 0.89     | 0.971 |
| Random Forest | 93.94%      | 1.00                 | 0.85              | 0.92     | 0.989 |

## Conclusion
The Decision Tree was selected as the primary model due to its highest accuracy (95.45%) and strong balance between precision and recall. In particular, it achieved a higher recall compared to both KNN and Random Forest, making it more effective at identifying actual recurrence cases—an essential requirement in medical diagnosis where missing positive cases can be critical.

Although KNN and Random Forest achieved perfect precision (1.00), indicating no false positive predictions, both models exhibited lower recall, meaning they missed a greater number of true recurrence cases. However, Random Forest demonstrated the highest AUC score (0.989), indicating superior overall classification capability and robustness.

Therefore, while the Decision Tree is preferred for its balanced and reliable performance, Random Forest serves as a strong alternative when overall classification strength is prioritized, and KNN remains useful in scenarios where precision is the primary concern.
