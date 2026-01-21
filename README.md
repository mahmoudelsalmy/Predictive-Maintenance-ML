# ⚙️ Predictive Maintenance System (Machine Learning)

A web-based **Predictive Maintenance System** built using **Machine Learning** and **Streamlit** to monitor equipment health and predict potential faults based on sensor vibration data.

This project demonstrates an end-to-end ML pipeline including feature extraction, dimensionality reduction, model inference, and an interactive web interface.

---

## 🚀 Live Demo
🔗 **Streamlit App:**  
https://predictive-maintenance-system-ml.streamlit.app/

---

## 🧠 Project Overview

Predictive maintenance aims to detect equipment failures **before they happen**, reducing downtime and maintenance costs.

This system:
- Accepts **sensor vibration readings**
- Extracts statistical features
- Applies **scaling + PCA**
- Predicts equipment condition (**Healthy / Faulty**)
- Displays confidence scores and maintenance recommendations

---

## 🛠️ Tech Stack

- **Python 3.11**
- **Streamlit** – Web Application
- **Scikit-learn** – Machine Learning & PCA
- **NumPy / Pandas** – Data Processing
- **Plotly** – Interactive Visualizations
- **Joblib** – Model Serialization

---

## 📁 Project Structure

```text
predictive-maintenance-streamlit/
├── app.py
│   └── Main Streamlit application that handles UI, inputs, and predictions.
├── model.py
│   └── Contains model-related helper functions and logic.
├── trained_model.pkl
│   └── Pre-trained machine learning model used for inference.
├── scaler.pkl
│   └── Feature scaling object applied before prediction.
├── pca.pkl
│   └── PCA transformer used for dimensionality reduction.
├── model_metadata.json
│   └── Stores metadata such as feature names and model configuration.
├── requirements.txt
│   └── Python dependencies required to run the application.
└── README.md
    └── Project documentation and usage instructions.
```


> ⚠️ **Note:**  
Training datasets are intentionally excluded from the deployed application to follow best practices for production ML systems.

---

## 🔍 Features

### ✅ Single Equipment Prediction
- Manual input of sensor readings
- Real-time fault prediction
- Confidence gauge visualization
- Actionable maintenance recommendations

### ✅ Batch Prediction
- Upload CSV files containing sensor data
- Sliding window feature extraction
- Predict equipment condition for multiple samples

### ✅ Prediction History
- Stores prediction logs during the session
- Confidence tracking over time
- Downloadable prediction history

---

## 📊 Machine Learning Pipeline

1. **Input:** Raw vibration sensor values  
2. **Feature Extraction:**  
   - Mean  
   - Standard Deviation  
   - RMS  
   - Minimum / Maximum  
   - Skewness  
   - Kurtosis  
3. **Preprocessing:**  
   - Standard Scaling  
   - Principal Component Analysis (PCA)  
4. **Model:** Supervised classification model  
5. **Output:**  
   - Equipment condition (Healthy / Faulty)  
   - Prediction confidence  

---

## 📁 Dataset

The model was trained using the **Gearbox Fault Diagnosis Dataset** from Kaggle:

🔗 https://www.kaggle.com/datasets/brjapon/gearbox-fault-diagnosis

Due to file size limitations, the dataset is **not included** in this repository.

---

## 🧪 Local Setup

```bash
git clone https://github.com/mahmoudelsalmy/predictive-maintenance-streamlit.git
cd predictive-maintenance-streamlit
pip install -r requirements.txt
streamlit run app.py
