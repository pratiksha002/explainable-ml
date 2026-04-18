# 🏠 Explainable ML System with SHAP & LIME

## 📌 Overview

This project is an **end-to-end Explainable Machine Learning System** that not only predicts house prices but also explains the reasoning behind each prediction.

Unlike traditional ML projects that stop at prediction, this system focuses on **interpretability, transparency, and real-world usability** by integrating explainability techniques, API deployment, database storage, and a visualization dashboard.

---

## 🎯 Problem Statement

In real-world applications, predictions alone are not sufficient. Stakeholders need to understand:

- Why was this prediction made?
- Which features influenced the outcome?
- Can we trust the model?

This project addresses these questions by combining **prediction + explanation + analysis**.

---

## 🚀 Features

### 🔹 Machine Learning
- Regression model for house price prediction
- Handles structured input data
- Supports real-time predictions via API

---

### 🔹 Explainability
- **SHAP (SHapley Additive Explanations)**
  - Feature contribution for each prediction
  - Global + local interpretability

- **LIME (Local Interpretable Model-agnostic Explanations)**
  - Rule-based local explanations

- **SHAP vs LIME Comparison**
  - Identifies agreement and differences between explanations

---

### 🔹 Backend (API)
Built using **FastAPI**

**Endpoints:**
- `POST /predict` → Get prediction  
- `POST /explain` → Get prediction + explanations  
- `GET /history` → Retrieve stored results  

---

### 🔹 Database Integration
- Uses **MongoDB**
- Stores:
  - Input data  
  - Predictions  
  - SHAP values  
  - LIME explanations  
  - Comparison insights  
  - Timestamp  

---

### 🔹 Visualization Dashboard
Built with **Streamlit**

Features:
- 📈 Prediction trends over time  
- 🔥 Feature importance (SHAP)  
- 📌 Feature vs prediction relationship  
- 🎛️ Interactive filters  

---

## 🧠 System Architecture
 - User Input
 - ↓
 - FastAPI Backend
 - ↓
 - ML Model (Prediction)
 - ↓
 - SHAP + LIME (Explainability)
 - ↓
 - Comparison Engine
 - ↓
 - MongoDB (Storage)
 - ↓
 - Streamlit Dashboard (Visualization)


---

## 🛠️ Tech Stack

- **Language:** Python  
- **ML:** Scikit-learn  
- **Explainability:** SHAP, LIME  
- **Backend:** FastAPI  
- **Database:** MongoDB  
- **Dashboard:** Streamlit  

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
 - git clone https://github.com/your-username/explainable-ml.git
 - cd explainable-ml

### 2. Create Virtual Environment
 - python -m venv .venv
 - Activate:
 - 
 - Windows:
 - .venv\Scripts\activate
 - Mac/Linux:
 - source .venv/bin/activate

### 3. Install Dependencies
 - pip install -r requirements.txt

###4. Setup MongoDB
 - Create MongoDB Atlas cluster or use local MongoDB
 - Add your connection string in:
 - app/database/db.py

###5. Run FastAPI Server
 - python -m uvicorn app.main:app --reload

- Open:

 http://127.0.0.1:8000/docs

 ###6. Run Dashboard
 - streamlit run dashboard.py
 - 📡 API Endpoints
 - 🔹 Predict
 - POST /predict
 - 🔹 Explain
 - POST /explain
 - 
 - Returns:
 - 
 - Prediction
 - SHAP values
 - LIME values
 - Human-readable explanations
 - Comparison insights
 - 🔹 History
 - GET /history
 - 
 - Returns all stored predictions
 - 
 - 📊 Dashboard Insights
 - Prediction trends over time
 - Feature importance (SHAP)
 - Feature vs prediction relationships
 - Interactive filtering
---
 - 🧠 Key Learnings
 - Model interpretability (SHAP vs LIME)
 - Backend + ML integration
 - Data storage and tracking
 - Building end-to-end ML systems
 - Real-world debugging and validation

---

 - 🔍 Real-World Applications
 - 🏦 Finance → Loan decision explanations
 - 🏥 Healthcare → Diagnosis transparency
 - 🏠 Real Estate → Price justification
 - 📊 Business → Decision support systems

---

 - 🚀 Future Improvements
 - Deploy backend & dashboard
 - Add advanced SHAP visualizations
 - Implement bias detection
 - Add authentication
 - 📌 Conclusion
 - 
 - This project demonstrates that machine learning is not just about building models, but about creating interpretable, reliable, and production-ready systems.