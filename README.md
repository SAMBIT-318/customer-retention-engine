# Customer Retention Engine

## 🎯 Overview
The Customer Retention Engine is a data science and machine learning project designed to analyze customer behavior, predict potential churn, and provide insights to improve overall customer retention.

## 📂 Project Structure
```text
customer-retention-engine/
│
├── app/
│   └── streamlit_app.py          # Your main Streamlit script
│
├── model/
│   └── best_churn_model.pkl      # Your exported model file (Must be uploaded!)
│
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── src/
│   ├── data_loader.py            # Renamed from data_loader.py.py
│   └── train.py
│
├── .gitignore
├── requirements.txt              # Lists all dependencies
└── README.md
