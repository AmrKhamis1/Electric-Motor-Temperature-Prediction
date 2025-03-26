🚀 Electric Motor Temperature Prediction - Machine Learning Project

📌 Overview

This project uses Machine Learning (ML) models to predict electric motor temperature based on sensor data. The dataset includes various factors affecting motor temperature, such as ambient temperature and rotational speed.

📊 Dataset

Source: [https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature]

Features: Voltage, Current, Rotational Speed, Torque, Ambient Temperature

Target: Motor Temperature (°C)

🔬 Machine Learning Workflow

1️⃣ Data Preprocessing

Load dataset using pandas

Handle missing values and outliers

Normalize numerical features using StandardScaler

Feature engineering

2️⃣ Exploratory Data Analysis (EDA)

Visualizations using matplotlib & seaborn

![image](https://github.com/user-attachments/assets/49e9e293-b525-4fcf-aa17-18990250408f)

Correlation heatmaps

Histogram of temperature distribution

3️⃣ Model Training

We trained multiple ML models to predict motor temperature:

Linear Regression

Logistic Regression

SVM / SVC 

K-Means Clustring

4️⃣ Evaluation Metrics

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R-squared Score (R²)

🔥 Results

✅ Best Performing Model

Model: [Linear Regression, Logisitic Regression, SVC]

MAE: 2.1972 °C

R² Score: 0.9879 ~ 98%

📌 Predicted vs. Actual Temperatures | Confusion Matrix & Visualizations

![image](https://github.com/user-attachments/assets/8db13a03-b597-4f4d-a245-dee0274a88ae)
![image](https://github.com/user-attachments/assets/5e4b482f-e462-42f5-8e26-8515c4aba152)
![image](https://github.com/user-attachments/assets/05a698cf-f1ba-4af5-ad09-9ae783cfa0f8)
![image](https://github.com/user-attachments/assets/7c5190a1-ff68-44f0-b18b-add416985711)


🛠 Installation & Usage

🔧 Setup

Clone this repository:

git clone https://github.com/AmrKhamis1/Electric-Motor-Temperature-Prediction.git
cd Electric-Motor-Temperature-Prediction

Install dependencies:

pip install -r requirements.txt

Run the Jupyter Notebook:

jupyter notebook

📌 Dependencies

numpy

pandas

matplotlib

seaborn

sklearn

tensorflow (for deep learning models)


