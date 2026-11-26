import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD ATTACKER DATA
# ============================================================

df = pd.read_csv("attacker_scouted_data.csv")

print("Dataset shape:", df.shape)


# ============================================================
# 2. FEATURES AND VICTIM RESPONSE
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

X = df[feature_columns].values.astype(np.float32)
y = df[target_column].values.astype(np.float32)

print("Number of samples:", len(X))
print("Number of features:", X.shape[1])


# ============================================================
# 3. 80/20 SPLIT
# ============================================================

X_train_original, X_test_original, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train_original))
print("Test samples:", len(X_test_original))


# ============================================================
# 4. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_original)
X_test = scaler.transform(X_test_original)


# ============================================================
# 5. PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
).reshape(-1, 1)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
).reshape(-1, 1)


# ============================================================
# 6. CLONE MODEL
# ============================================================

class CloneMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(8, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# 7. FUNCTION TO TRAIN MODEL
# ============================================================

def train_model(
    X_data,
    y_data,
    sample_weights=None,
    epochs=100
):

    model = CloneMLP()

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    X_tensor = torch.tensor(
        X_data,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y_data,
        dtype=torch.float32
    ).reshape(-1, 1)

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    # --------------------------------------------------------
    # If weights are supplied, important samples are selected
    # more frequently during training.
    # --------------------------------------------------------

    if sample_weights is not None:

        weights_tensor = torch.tensor(
            sample_weights,
            dtype=torch.double
        )

        sampler = WeightedRandomSampler(
            weights=weights_tensor,
            num_samples=len(weights_tensor),
            replacement=True
        )

        loader = DataLoader(
            dataset,
            batch_size=32,
            sampler=sampler
        )

    else:

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for batch_X, batch_y in loader:

            optimizer.zero_grad()

            outputs = model(batch_X)

            loss = criterion(
                outputs,
                batch_y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

    return model


# ============================================================
# 8. BASELINE CLONE
# ============================================================

print("\n========================================")
print("TRAINING BASELINE CLONE")
print("========================================")

baseline_model = train_model(
    X_train,
    y_train,
    epochs=100
)


# ============================================================
# 9. BASELINE FIDELITY
# ============================================================

baseline_model.eval()

with torch.no_grad():

    baseline_logits = baseline_model(
        X_test_tensor
    )

    baseline_probabilities = torch.sigmoid(
        baseline_logits
    )

    baseline_predictions = (
        baseline_probabilities >= 0.5
    ).float()


baseline_predictions = (
    baseline_predictions
    .numpy()
    .flatten()
    .astype(int)
)

victim_responses = y_test.astype(int)

baseline_agreements = (
    baseline_predictions
    == victim_responses
)

baseline_fidelity = (
    baseline_agreements.sum()
    / len(victim_responses)
) * 100

print(
    f"\nBaseline Fidelity: "
    f"{baseline_fidelity:.2f}%"
)

print(
    f"Baseline agreements: "
    f"{baseline_agreements.sum()}/{len(victim_responses)}"
)


# ============================================================
# 10. CALCULATE CLONE JACOBIAN
# ============================================================
#
# The Jacobian tells us:
#
#     "How sensitive is the Clone output to each feature?"
#
# We calculate it ONLY on the 800 training samples.
#
# The 200 test samples remain completely untouched.
# ============================================================

print("\n========================================")
print("CALCULATING CLONE JACOBIAN")
print("========================================")

X_gradient = torch.tensor(
    X_train,
    dtype=torch.float32,
    requires_grad=True
)

baseline_model.eval()

logits = baseline_model(
    X_gradient
)

# Sum allows us to calculate gradients for all
# training samples in one backward pass.

gradient = torch.autograd.grad(
    outputs=logits.sum(),
    inputs=X_gradient
)[0]

gradient = gradient.detach().numpy()


# ============================================================
# 11. JACOBIAN MAGNITUDE
# ============================================================
#
# For each training sample:
#
# gradient = [
#     d(output)/d(feature1),
#     d(output)/d(feature2),
#     ...
# ]
#
# Large gradient magnitude means:
#
# "Small changes around this point can strongly affect
#  the Clone's output."
# ============================================================

gradient_magnitude = np.linalg.norm(
    gradient,
    axis=1
)

print(
    "Jacobian magnitude range:",
    gradient_magnitude.min(),
    "to",
    gradient_magnitude.max()
)


# ============================================================
# 12. CLONE PROBABILITIES ON TRAINING DATA
# ============================================================

with torch.no_grad():

    train_logits = baseline_model(
        torch.tensor(
            X_train,
            dtype=torch.float32
        )
    )

    train_probabilities = torch.sigmoid(
        train_logits
    ).numpy().flatten()


# ============================================================
# 13. DISTANCE FROM DECISION BOUNDARY
# ============================================================
#
# Decision boundary:
#
# probability = 0.5
#
# The closer the probability is to 0.5,
# the more uncertain the Clone is.
# ============================================================

boundary_distance = np.abs(
    train_probabilities - 0.5
)


# ============================================================
# 14. COMBINE JACOBIAN + BOUNDARY INFORMATION
# ============================================================
#
# Important samples are:
#
# 1. Close to the decision boundary
# 2. Sensitive according to the Jacobian
#
# These samples receive larger training weights.
# ============================================================

# Normalize Jacobian magnitude

gradient_min = gradient_magnitude.min()
gradient_max = gradient_magnitude.max()

if gradient_max > gradient_min:

    normalized_gradient = (
        (gradient_magnitude - gradient_min)
        / (gradient_max - gradient_min)
    )

else:

    normalized_gradient = np.zeros_like(
        gradient_magnitude
    )


# Normalize boundary importance

boundary_importance = 1.0 / (
    boundary_distance + 0.01
)

boundary_importance = (
    boundary_importance
    / boundary_importance.max()
)


# Combine both

importance_score = (
    0.5 * normalized_gradient
    +
    0.5 * boundary_importance
)


# ============================================================
# 15. CONVERT IMPORTANCE INTO SAMPLING WEIGHTS
# ============================================================
#
# Normal samples:
#     weight ≈ 1
#
# Important boundary samples:
#     weight > 1
#
# We are NOT creating new Victim queries.
# We are simply showing the Clone the existing informative
# samples more often during training.
# ============================================================

sample_weights = (
    1.0 + 4.0 * importance_score
)


# ============================================================
# 16. DISPLAY MOST IMPORTANT SAMPLES
# ============================================================

important_indices = np.argsort(
    importance_score
)[-10:][::-1]

print("\nTop 10 Jacobian/boundary samples:")

for rank, index in enumerate(
    important_indices,
    start=1
):

    print(
        f"{rank}. "
        f"Index={index}, "
        f"Probability={train_probabilities[index]:.4f}, "
        f"BoundaryDistance={boundary_distance[index]:.4f}, "
        f"Jacobian={gradient_magnitude[index]:.6f}, "
        f"Importance={importance_score[index]:.4f}"
    )


# ============================================================
# 17. TRAIN IMPROVED CLONE
# ============================================================

print("\n========================================")
print("TRAINING JACOBIAN-GUIDED CLONE")
print("========================================")

improved_model = train_model(
    X_train,
    y_train,
    sample_weights=sample_weights,
    epochs=100
)


# ============================================================
# 18. TEST IMPROVED CLONE
# ============================================================

improved_model.eval()

with torch.no_grad():

    improved_logits = improved_model(
        X_test_tensor
    )

    improved_probabilities = torch.sigmoid(
        improved_logits
    )

    improved_predictions = (
        improved_probabilities >= 0.5
    ).float()


improved_predictions = (
    improved_predictions
    .numpy()
    .flatten()
    .astype(int)
)


# ============================================================
# 19. CALCULATE IMPROVED FIDELITY
# ============================================================

improved_agreements = (
    improved_predictions
    == victim_responses
)

improved_fidelity = (
    improved_agreements.sum()
    / len(victim_responses)
) * 100


# ============================================================
# 20. FINAL RESULTS
# ============================================================

print("\n========================================")
print("           FINAL RESULTS")
print("========================================")

print(
    f"Unseen test samples : "
    f"{len(victim_responses)}"
)

print(
    f"Baseline agreements : "
    f"{baseline_agreements.sum()}"
)

print(
    f"Baseline fidelity   : "
    f"{baseline_fidelity:.2f}%"
)

print("----------------------------------------")

print(
    f"Improved agreements : "
    f"{improved_agreements.sum()}"
)

print(
    f"Improved fidelity   : "
    f"{improved_fidelity:.2f}%"
)

print("----------------------------------------")

print(
    f"Improvement        : "
    f"{improved_fidelity - baseline_fidelity:+.2f} percentage points"
)

print("========================================")
