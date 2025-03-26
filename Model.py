# %%
# ================= IMPORTS =================
# Import essential libraries for data handling, visualization, and machine learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os
import tkinter as tk
from tkinter import Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Importing ML models and utilities from Scikit-Learn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

# %%
# Set plot style for consistent visualizations
plt.style.use('seaborn-v0_8')
sns.set(style="whitegrid")


# %%
# ================= DATA LOADING AND PREPROCESSING =================
try:
    df = pd.read_csv("data.csv")
    print("Successfully loaded data.csv")
except Exception as e:
    print(f"Error loading data.csv: {e}")
    exit(1)

# Display dataset info
print(f"Dataset Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nColumn Data Types:")
for col in df.columns:
    print(f"{col}: {df[col].dtype}")

# Handle missing values and drop irrelevant columns
df = df.dropna()
if "profile_id" in df.columns:
    df = df.drop(columns=["profile_id"])

# Convert numeric columns to float32 for efficiency
numeric_columns = ['u_q', 'coolant', 'stator_winding', 'u_d', 'stator_tooth', 
                   'motor_speed', 'i_d', 'i_q', 'pm', 'stator_yoke', 'ambient', 'torque']
for col in numeric_columns:
    if col in df.columns:
        try:
            df[col] = df[col].astype('float32')
        except Exception as e:
            print(f"Error converting {col} to float32: {e}")
            df[col] = df[col].fillna(0).astype('float32')

# Correlation matrix visualization
print("\nGenerating correlation matrix...")
corr_matrix = df[numeric_columns].corr()

# %%
# ================= FEATURE SELECTION AND DATA SPLITTING =================
features = [col for col in numeric_columns if col != 'stator_winding' and col in df.columns]
target_regression = 'stator_winding'
X = df[features]
y_regression = df[target_regression].astype('float32')

df['high_temperature'] = (df[target_regression] > df[target_regression].median()).astype('int32')
y_classification = df['high_temperature']

# Train-test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_classification, test_size=0.2, random_state=42)

# Feature scaling
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

scaler_cls = StandardScaler()
X_train_cls_scaled = scaler_cls.fit_transform(X_train_cls)
X_test_cls_scaled = scaler_cls.transform(X_test_cls)

print("Data scaling and splitting complete.")

# %%
# # ================= MACHINE LEARNING MODELS =================##

# ===== LINEAR REGRESSION MODEL =====
print("\n===== Training Linear Regression Model =====")
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train_reg_scaled, y_train_reg)

# Make predictions
y_pred_reg = lin_reg_model.predict(X_test_reg_scaled)

# Evaluate regression model
r2_test = r2_score(y_test_reg, y_pred_reg)
mae_test = mean_absolute_error(y_test_reg, y_pred_reg)
mse_test = mean_squared_error(y_test_reg, y_pred_reg)

print(f"Linear Regression - Test R² Score: {r2_test:.4f}")
print(f"Linear Regression - Test MAE: {mae_test:.4f}")
print(f"Linear Regression - Test MSE: {mse_test:.4f}")

# Check for overfitting
r2_train = r2_score(y_train_reg, lin_reg_model.predict(X_train_reg_scaled))
if r2_train - r2_test > 0.05:
    print("Warning: Potential Overfitting! Train score is much higher than test score.")
else:
    print("No significant overfitting detected.")

# ===== LOGISTIC REGRESSION MODEL =====
print("\n===== Training Logistic Regression Model =====")
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train_cls_scaled, y_train_cls)

# Make predictions
y_pred_log = log_reg_model.predict(X_test_cls_scaled)

# Evaluate logistic regression model
accuracy_log = accuracy_score(y_test_cls, y_pred_log)
print(f"Logistic Regression - Test Accuracy: {accuracy_log:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_cls, y_pred_log))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_cls, y_pred_log))

# %%
# ===== SVM MODEL =====
print("\n===== Training SVM Model =====")
svm_model = LinearSVC(penalty='l2', C=0.1, max_iter=1000, class_weight='balanced')
svm_model.fit(X_train_cls_scaled, y_train_cls)

# Make predictions
y_pred_svm = svm_model.predict(X_test_cls_scaled)

# Evaluate SVM model
accuracy_svm = accuracy_score(y_test_cls, y_pred_svm)
print(f"SVM - Test Accuracy: {accuracy_svm:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_cls, y_pred_svm))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_cls, y_pred_svm))


# %%
# ===== KMEANS CLUSTERING =====
("\n===== Training KMeans Clustering =====")

# Downsample training data (e.g., use only 20% of original data)
X_train_cls_sampled, _, y_train_cls_sampled, _ = train_test_split(
    X_train_cls_scaled, y_train_cls, test_size=0.8, random_state=42
)

kmeans_model = KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=100)
kmeans_model.fit(X_train_cls_scaled)

# Predict clusters for test data
y_pred_clusters = kmeans_model.predict(X_test_cls_scaled)

# Map clusters to actual class labels (fixed version)
def map_clusters_to_labels(y_true, y_pred):
    # Convert to numpy arrays to avoid pandas indexing issues
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    label_mapping = {}
    for cluster in np.unique(y_pred_np):
        # Get indices where predictions equal current cluster
        cluster_indices = np.where(y_pred_np == cluster)[0]
        # Get true labels for those indices
        actual_labels = y_true_np[cluster_indices]
        # Find most common label (mode)
        if len(actual_labels) > 0:
            most_common_label = np.bincount(actual_labels.astype(int)).argmax()
            label_mapping[cluster] = most_common_label
    
    # Apply the mapping
    return np.array([label_mapping.get(cluster, 0) for cluster in y_pred_np])

# Map predicted clusters to the closest actual class
y_pred_kmeans = map_clusters_to_labels(y_test_cls, y_pred_clusters)

# Evaluate KMeans model
accuracy_kmeans = accuracy_score(y_test_cls, y_pred_kmeans)
print(f"KMeans - Test Accuracy: {accuracy_kmeans:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_cls, y_pred_kmeans))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_cls, y_pred_kmeans))

# %%

# ================= GUI IMPLEMENTATION =================
def show_correlation_matrix():
    window = Toplevel(root)
    window.title("Correlation Matrix")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title('Correlation Matrix of Motor Data Variables')
    
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()

def show_linear_regression():
    window = Toplevel(root)
    window.title("Linear Regression Results")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Predicted vs Actual values
    ax1.scatter(y_test_reg, y_pred_reg, color="blue", alpha=0.5)
    ax1.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 
             color="red", linestyle="--")
    ax1.set_xlabel("Actual Motor Speed")
    ax1.set_ylabel("Predicted Motor Speed")
    ax1.set_title("Predicted vs Actual Values")
    
    # Plot 2: Residuals
    residuals = y_test_reg - y_pred_reg
    ax2.scatter(y_pred_reg, residuals, color="green", alpha=0.5)
    ax2.axhline(y=0, color="red", linestyle="--")
    ax2.set_xlabel("Predicted Values")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residual Plot")
    
    # Add metrics as text
    metrics_text = f"R² Score: {r2_test:.4f}\nMAE: {mae_test:.4f}\nMSE: {mse_test:.4f}"
    fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()

def show_logistic_regression():
    window = Toplevel(root)
    window.title("Logistic Regression Results")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test_cls, y_pred_log)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Confusion Matrix")
    
    # Plot 2: Feature Importance
    if hasattr(log_reg_model, 'coef_'):
        importance = np.abs(log_reg_model.coef_[0])
        feature_names = X.columns
        
        # Sort feature importances
        sorted_idx = np.argsort(importance)
        ax2.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
        ax2.set_yticks(range(len(sorted_idx)))
        ax2.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax2.set_title("Feature Importance")
    else:
        ax2.text(0.5, 0.5, "Feature importance not available", 
                ha='center', va='center', fontsize=12)
    
    # Add accuracy as text
    fig.text(0.5, 0.01, f"Accuracy: {accuracy_log:.4f}", ha='center', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.5))
    
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()

def show_svm():
    window = Toplevel(root)
    window.title("SVM Results")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test_cls, y_pred_svm)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Confusion Matrix")
    
    # Plot 2: Feature Importance
    if hasattr(svm_model, 'coef_'):
        importance = np.abs(svm_model.coef_[0])
        feature_names = X.columns
        
        # Sort feature importances
        sorted_idx = np.argsort(importance)
        ax2.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
        ax2.set_yticks(range(len(sorted_idx)))
        ax2.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax2.set_title("Feature Importance")
    else:
        ax2.text(0.5, 0.5, "Feature importance not available", 
                ha='center', va='center', fontsize=12)
    
    # Add accuracy as text
    fig.text(0.5, 0.01, f"Accuracy: {accuracy_svm:.4f}", ha='center', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.5))
    
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()

def show_kmeans():
    window = Toplevel(root)
    window.title("KMeans Clustering Results")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test_cls, y_pred_kmeans)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Confusion Matrix")
    
    # Plot 2: Cluster Visualization
   # Reduce number of points plotted by selecting a random subset (e.g., 10% of data)
    num_samples = int(0.0005 * len(X_test_cls_scaled))  # Take only 10% of the data
    sample_indices = np.random.choice(len(X_test_cls_scaled), size=num_samples, replace=False)

# Select only sampled points
    X_sampled = X_test_cls_scaled[sample_indices]
    y_pred_sampled = y_pred_clusters[sample_indices]

# Plot only the sampled points
    scatter = ax2.scatter(X_sampled[:, 0], X_sampled[:, 1], 
                      c=y_pred_sampled, cmap='viridis', alpha=0.8)


    ax2.set_xlabel(features[0])
    ax2.set_ylabel(features[1])
    ax2.set_title("Cluster Visualization")
    plt.colorbar(scatter, ax=ax2)
    
    # Add accuracy as text
    fig.text(0.5, 0.01, f"Accuracy: {accuracy_kmeans:.4f}", ha='center', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.5))
    
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()

# Create main GUI window
root = tk.Tk()
root.title("Motor Data Analysis - ML Model Comparison")
root.geometry("600x500")

# Add title and description
title_label = tk.Label(root, text="Motor Data Analysis Dashboard", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

description = """
This application analyzes motor data using various machine learning models.
Choose a model or visualization method below to see detailed results.
"""
desc_label = tk.Label(root, text=description, justify=tk.LEFT, padx=20)
desc_label.pack(pady=10)

# Create a frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=20)

# Create buttons with improved styling
button_style = {"width": 25, "height": 2, "font": ("Arial", 10), "bg": "#e0e0e0"}

btn_corr = tk.Button(button_frame, text="Correlation Matrix", command=show_correlation_matrix, **button_style)
btn_corr.grid(row=0, column=0, padx=10, pady=10)

btn_linreg = tk.Button(button_frame, text="Linear Regression", command=show_linear_regression, **button_style)
btn_linreg.grid(row=0, column=1, padx=10, pady=10)

btn_logreg = tk.Button(button_frame, text="Logistic Regression", command=show_logistic_regression, **button_style)
btn_logreg.grid(row=1, column=0, padx=10, pady=10)

btn_svm = tk.Button(button_frame, text="SVM Classification", command=show_svm, **button_style)
btn_svm.grid(row=1, column=1, padx=10, pady=10)

btn_kmeans = tk.Button(button_frame, text="KMeans Clustering", command=show_kmeans, **button_style)
btn_kmeans.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Add status information
status_text = f"Data loaded: {df.shape[0]} samples, {df.shape[1]} features"
status_label = tk.Label(root, text=status_text, bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_label.pack(side=tk.BOTTOM, fill=tk.X)

# Run the GUI main loop
root.mainloop()


