
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD ATTACKER DATA
# ============================================================

df = pd.read_csv("attacker_scouted_data.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. SELECT FEATURES AND VICTIM RESPONSES
# ============================================================

feature_columns = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s"
]

target_column = "Firewall_Response"


X = df[feature_columns].values
y = df[target_column].values


# Convert to float32 for PyTorch
X = X.astype(np.float32)
y = y.astype(np.float32)


print("\nNumber of samples:", len(X))
print("Number of features:", X.shape[1])


# ============================================================
# 3. SPLIT INTO TRAINING AND TEST DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
# Keep a copy of the original feature values
# before StandardScaler changes them
X_test_original = X_test.copy()

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))


# ============================================================
# 4. SCALE THE FEATURES
# ============================================================ bringing to same scale 2 and 20 is comparable 3000000 and 0.54678565443 is not

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# 5. CONVERT DATA TO PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


# ============================================================
# 6. CREATE PYTORCH DATASET AND DATALOADER
# ============================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True # randomly shuffle the training data
)


# ============================================================
# 7. DEFINE THE CLONE MLP
# ============================================================

class CloneMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(8, 32), #take 8 inputs and tranform them to 32 by assigning weights to them that is reaised during the trsining procedure
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


model = CloneMLP()

print("\nClone model:")
print(model)


# ============================================================
# 8. LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 9. TRAIN THE CLONE
# ============================================================

epochs = 100

print("\nTraining Clone model...")

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for batch_X, batch_y in train_loader:

        # Forward pass
        outputs = model(batch_X)

        # Calculate loss
        loss = criterion(outputs, batch_y)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:

        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {average_loss:.4f}"
        )


# ============================================================
# 10. TEST THE CLONE ON THE UNSEEN 20%
# ============================================================

model.eval()

with torch.no_grad():

    logits = model(X_test_tensor)

    probabilities = torch.sigmoid(logits)

    clone_predictions = (
        probabilities >= 0.5
    ).float()

# Save probabilities as normal NumPy values
clone_probabilities = probabilities.numpy().flatten()

# Convert predictions to NumPy
clone_predictions = (
    clone_predictions.numpy().flatten()
)

victim_responses = y_test.astype(int)


# ============================================================
# 11. FIND DISAGREEMENTS
# ============================================================

test_results = pd.DataFrame(
    X_test_original,
    columns=feature_columns
)

test_results["Victim_Response"] = victim_responses
test_results["Clone_Response"] = clone_predictions.astype(int)
test_results["Clone_Probability"] = clone_probabilities

test_results["Agreement"] = (
    test_results["Victim_Response"]
    == test_results["Clone_Response"]
)

# Keep only rows where Victim and Clone disagree
mismatched_rows = test_results[
    test_results["Agreement"] == False
].copy()

print("\n========================================")
print("        DISAGREEMENT ANALYSIS")
print("========================================")

print("\nNumber of disagreements:",
      len(mismatched_rows))

print("\nMismatched rows:")
print(mismatched_rows.to_string(index=False))

# ============================================================
# 12. FEATURE RANGE AND VARIANCE OF DISAGREEMENTS
# ============================================================

if len(mismatched_rows) > 0:

    variance_table = pd.DataFrame({
        "Variance":
            mismatched_rows[feature_columns].var(),

        "Minimum":
            mismatched_rows[feature_columns].min(),

        "Maximum":
            mismatched_rows[feature_columns].max(),

        "Range":
            mismatched_rows[feature_columns].max()
            - mismatched_rows[feature_columns].min()
    })

    variance_table = variance_table.sort_values(
        by="Variance",
        ascending=False
    )

    print("\n========================================")
    print("   DISAGREEMENT FEATURE ANALYSIS")
    print("========================================")

    print(variance_table)
    

    # Convert disagreement features to the same standardized scale
    mismatch_scaled = scaler.transform(
        mismatched_rows[feature_columns].values
    )

    standardized_variance = pd.Series(
        mismatch_scaled.var(axis=0),
        index=feature_columns
    ).sort_values(ascending=False)

    print("\nStandardized variance:")
    print(standardized_variance)
 
 # ============================================================
# 13. BOUNDARY ANALYSIS
# ============================================================

test_results["Distance_From_0.5"] = abs(
    test_results["Clone_Probability"] - 0.5
)

boundary_rows = test_results.sort_values(
    by="Distance_From_0.5"
)

print("\n========================================")
print("       CLOSEST TO CLONE BOUNDARY")
print("========================================")

print(
    boundary_rows[
        feature_columns
        + [
            "Victim_Response",
            "Clone_Response",
            "Clone_Probability",
            "Distance_From_0.5"
        ]
    ].head(20).to_string(index=False)
)

boundary_disagreements = mismatched_rows.copy()

boundary_disagreements["Distance_From_0.5"] = abs(
    boundary_disagreements["Clone_Probability"] - 0.5
)

boundary_disagreements = boundary_disagreements.sort_values(
    by="Distance_From_0.5"
)

print("\n========================================")
print("  BOUNDARY DISAGREEMENTS")
print("========================================")

print(
    boundary_disagreements[
        feature_columns
        + [
            "Victim_Response",
            "Clone_Response",
            "Clone_Probability",
            "Distance_From_0.5"
        ]
    ].head(20).to_string(index=False)
)
   
# ============================================================
# 11. CALCULATE FIDELITY
# ============================================================

agreements = (
    clone_predictions.astype(int)
    == victim_responses
)

number_of_agreements = agreements.sum()

total_test_samples = len(victim_responses)

fidelity = (
    number_of_agreements
    / total_test_samples
) * 100


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("        CLONE MODEL RESULTS")
print("========================================")

print(
    f"Total unseen test samples : {total_test_samples}"
)

print(
    f"Victim-Clone agreements   : {number_of_agreements}"
)

print(
    f"Disagreements             : "
    f"{total_test_samples - number_of_agreements}"
)

print(
    f"Fidelity                  : {fidelity:.2f}%"
)

print("========================================")

