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
  - **Decision Tree** (Accuracy: ~97.4%, Recall: 95%)  
- **Key Techniques:** Mode Imputation, Label Encoding, One-Hot Encoding, Feature Scaling (StandardScaler)

---

##  Dataset

The project uses the `filtered_thyroid_data.csv` dataset, which contains patient records with the following attributes:

- **Demographics:** Age, Gender  
- **Medical History:** Hx Radiotherapy, Adenopathy  
- **Pathology:** Pathology type, Focality, Risk  
- **Staging:** T, N, M, Stage  
- **Outcome:** Response, **Recurred (Target Variable)**

---

##  Installation

### 1. Clone the repository

```bash
git clone  https://github.com/Bini-fish/Thyroid_Recurrence_Prediction.git
cd thyroid-recurrence-prediction
```

### 2. Create a virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

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

| Model         | Accuracy                   | Precision (Recurred) | Recall (Recurred) | F1-Score |
| ------------- | -------------------------- | -------------------- | ----------------- | -------- |
| Decision Tree | 97.40%                     | 0.95                 | 0.95              | 0.95     |
| KNN (K=5)     | 94.81%                     | 1.00                 | 0.79              | 0.88     |
| Random Forest | (see notebook for results) |                      |                   |          |

## Conclusion
The Decision Tree was selected as the superior model. Although KNN achieved perfect precision, the Decision Tree provided a significantly higher recall rate (0.95 vs 0.79), meaning it is much better at identifying actual recurrence cases. The Random Forest classifier was also implemented for advanced comparison—see the notebook for its detailed results and performance metrics.