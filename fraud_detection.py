
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pyod.models.auto_encoder import AutoEncoder
import time

import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load the credit card transaction dataset
df = pd.read_csv("data/creditcard.csv")

# Display the first five transactions
print("First 5 rows:")
print(df.head())

# Display the number of rows and columns
print("\nDataset shape:", df.shape)

# Check the number of legitimate and fraudulent transactions
print("\nTransaction distribution:")
print(df["Class"].value_counts())

# Check for missing values
print("\nTotal missing values:", df.isnull().sum().sum())



# Separate input features from the fraud labels
X = df.drop(columns=["Class"])
y = df["Class"]

# Split into training (80%) and testing (20%) sets
# Stratify preserves the fraud ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train the AutoEncoder using legitimate transactions only
X_train_normal = X_train[y_train == 0]

print("\nTraining and Testing Data:")
print("Training samples:", len(X_train_normal))
print("Testing samples:", len(X_test))
print("Fraud cases in test set:", y_test.sum())


# Create a scaler for standardizing the features
scaler = StandardScaler()

# Learn scaling parameters from legitimate training data only
X_train_scaled = scaler.fit_transform(X_train_normal)

# Apply the same scaling to the test data
# Do not fit the scaler again on test data
X_test_scaled = scaler.transform(X_test)

# Display the resulting dimensions
print("\nScaled Data:")
print("Training shape:", X_train_scaled.shape)
print("Testing shape:", X_test_scaled.shape)

# Verify that training features are standardized
print("Mean of first feature:", round(X_train_scaled[:, 0].mean(), 4))
print("Standard deviation:", round(X_train_scaled[:, 0].std(), 4))


# Initialize the AutoEncoder model
model = AutoEncoder(
    hidden_neuron_list=[32, 16, 8, 16, 32],
    epoch_num=10,
    batch_size=256,
    lr=0.001,
    contamination=0.00173,
    preprocessing=False,
    verbose=1,
    random_state=42
)

# Train the model using legitimate transactions
print("\nTraining AutoEncoder...")
start_time = time.time()

model.fit(X_train_scaled)

training_time = time.time() - start_time

print("\nTraining completed!")
print(f"Training time: {training_time:.2f} seconds")


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Predict fraud labels for the test dataset
# 0 = legitimate transaction
# 1 = potentially fraudulent transaction
y_pred = model.predict(X_test_scaled)

# Get anomaly scores (higher means more anomalous)
anomaly_scores = model.decision_function(X_test_scaled)

# Evaluate classification performance
print("\nFraud Detection Results:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate ROC-AUC using anomaly scores
roc_auc = roc_auc_score(y_test, anomaly_scores)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Create a folder to store experiment screenshots
os.makedirs("screenshots", exist_ok=True)

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Legitimate", "Fraud"],
    cmap="Blues",
    values_format="d",
    ax=ax
)

ax.set_title("Credit Card Fraud Detection - AutoEncoder")

# Save the visualization for the assignment report
plt.tight_layout()
plt.savefig(
    "screenshots/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
