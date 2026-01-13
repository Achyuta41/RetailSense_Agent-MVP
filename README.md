# 🛒 RetailSense Agent
### An Agentic AI System for Smart Retail Inventory Management - Minimal version or Basic version

RetailSense Agent is an Agentic AI-powered retail assistant that predicts product demand and automatically alerts store managers when stock is likely to run out. It helps retailers avoid stock-out losses and maintain optimal inventory levels.

---

## 🚀 Project Overview

In retail businesses, running out of stock leads to lost sales and unhappy customers.  
RetailSense Agent solves this problem by:

- Analyzing historical sales data  
- Predicting future demand using Machine Learning  
- Comparing predicted demand with current inventory  
- Automatically sending email alerts when stock is at risk  

This system follows the Agentic AI workflow:

Observe → Analyze → Decide → Act

---

## 🎯 Key Features

- Sales forecasting using Machine Learning  
- Stock-out risk detection  
- Automatic email alerts  
- Reorder quantity recommendations  
- Interactive Streamlit dashboard  
- CSV data upload support  

---

## 🏪 Use Case

Retail stores struggle to track inventory and predict which products will run out.  
RetailSense Agent continuously monitors sales trends and warns managers before stock reaches critical levels.

---

## 👥 Target Users

- Retail store owners  
- Inventory managers  
- Supermarkets  
- Small and medium retail businesses  

---

## 🧠 How It Works

1. Sales data is uploaded to the system  
2. A machine learning model predicts future demand  
3. The system compares predicted demand with available stock  
4. If risk is detected, the agent automatically sends an email alert  

---

## 🧩 System Architecture

Sales Data (CSV)  
        ↓  
Demand Forecasting Model  
        ↓  
Decision Engine  
        ↓  
Action Engine (Email Alerts)  
        ↓  
Retail Manager  

---

## ⚙️ Technology Stack

- Python  
- Pandas & NumPy  
- Scikit-Learn or Prophet  
- Streamlit  
- SMTP (Email Automation)  
- Walmart Sales Dataset (Kaggle)  

---

## 🧪 Dataset

We use the Walmart Sales Forecasting Dataset.  
A custom `current_stock` column is added to simulate real-world inventory levels.

---

## 📩 Example Email Alert

Subject: Low Stock Alert  

Product: Milk  
Current Stock: 40  
Predicted Demand: 75  
Recommended Reorder: 35 units  

---

## 🌟 Why RetailSense Agent?

- Prevents stock-out losses  
- Saves manual monitoring time  
- Improves customer satisfaction  
- Enables smarter business decisions  

---

## 📈 Future Enhancements

- WhatsApp and SMS alerts  
- POS system integration  
- IoT-based shelf monitoring  
- Dynamic pricing suggestions  
- Vendor auto-ordering  

---

