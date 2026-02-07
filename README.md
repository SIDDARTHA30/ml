# 🎓 End-to-End Machine Learning Project  
## SVM Classification Web App (Streamlit)

This project is a **complete end-to-end Machine Learning pipeline** built using Python and Streamlit.  
It demonstrates the full ML workflow from **data ingestion → cleaning → model training → evaluation**.

This project was created as part of my **Machine Learning portfolio for placements**.

---

## 🚀 Project Highlights

- Download or upload datasets
- Perform Exploratory Data Analysis (EDA)
- Handle missing values automatically
- Save cleaned datasets with timestamps
- Train Support Vector Machine (SVM) model
- View Accuracy & Confusion Matrix
- Fully interactive Streamlit web app

---

## 📸 Application Screenshots

<p align="center">
  <img src="screenshots/home.png" width="600"/>
</p>

<p align="center">
  <img src="screenshots/eda.png" width="600"/>
</p>

<p align="center">
  <img src="screenshots/training.png" width="600"/>
</p>

---

## 🛠 Tech Stack

- Python  
- Streamlit  
- Pandas  
- NumPy  
- Scikit-Learn  
- Matplotlib  
- Seaborn  

---

## 📂 Project Structure

ml/
│
├── app.py
├── requirement.txt
├── README.md
│
├── data/
│ ├── raw/
│ │ └── iris.csv
│ └── clean/
│ └── cleaned_*.csv
│
└── screenshots/

## ⚙️ How to Run the Project

### 1️⃣ Install Dependencies
```bash

pip install -r requirement.txt
2️⃣ Run the Streamlit App
streamlit run app.py
3️⃣ Open in Browser
http://localhost:8501
🎯 Workflow of the App
Step 1 — Data Ingestion
Download Iris dataset OR upload CSV

Step 2 — Exploratory Data Analysis
Dataset preview

Missing values

Correlation heatmap

Step 3 — Data Cleaning
Choose missing value strategy:

Mean

Median

Drop rows

Step 4 — Save Clean Dataset
Dataset saved with timestamp

Step 5 — Train SVM Model
Select:

Kernel

Regularization (C)

Gamma

Step 6 — Model Evaluation
Accuracy Score

Confusion Matrix

📊 Dataset Used
Default dataset: Iris Flower Classification



