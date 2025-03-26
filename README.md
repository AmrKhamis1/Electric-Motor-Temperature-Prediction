🚀 Electric Motor Temperature Prediction - Machine Learning Project

📌 Overview

This project uses Machine Learning (ML) models to predict electric motor temperature based on sensor data. The dataset includes various factors affecting motor temperature, such as ambient temperature, rotational speed, and voltage.

📊 Dataset

Source: [Specify dataset source, e.g., UCI repository]

Features: Voltage, Current, Rotational Speed, Torque, Ambient Temperature

Target: Motor Temperature (°C)

🔬 Machine Learning Workflow

1️⃣ Data Preprocessing

Load dataset using pandas

Handle missing values and outliers

Normalize numerical features using StandardScaler

Feature engineering (if applicable)

2️⃣ Exploratory Data Analysis (EDA)

Visualizations using matplotlib & seaborn

Correlation heatmaps

Histogram of temperature distribution

3️⃣ Model Training

We trained multiple ML models to predict motor temperature:

Linear Regression

Random Forest Regressor

Gradient Boosting (XGBoost, LightGBM)

Neural Networks (TensorFlow/Keras)

4️⃣ Evaluation Metrics

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R-squared Score (R²)

🔥 Results

✅ Best Performing Model

Model: [Specify best model, e.g., Random Forest]

MAE: X.XX °C

R² Score: X.XX

📌 Confusion Matrix & Visualizations

Confusion Matrix:


Predicted vs. Actual Temperatures:


🛠 Installation & Usage

🔧 Setup

Clone this repository:

git clone https://github.com/YOUR_USERNAME/motor-temperature-ml.git
cd motor-temperature-ml

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

📜 License

This project is MIT licensed.

