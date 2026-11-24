import pandas as pd
import glob

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Select the 8 features
# --------------------------------------------------

features = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s"
]

columns_needed = features + ["Label"]


# --------------------------------------------------
# 2. Load the CIC-IDS2017 CSV files
# --------------------------------------------------

files = glob.glob("MachineLearningCSV/*.csv")

dataframes = []

for file in files:
    print("Loading:", file)

    data = pd.read_csv(
        file,
        usecols=lambda col: col.strip() in columns_needed
    )

    data.columns = data.columns.str.strip()

    dataframes.append(data)


df = pd.concat(dataframes, ignore_index=True)


# --------------------------------------------------
# 3. Clean the data
# --------------------------------------------------

df = df.replace([float("inf"), float("-inf")], float("nan"))
df = df.dropna()


# --------------------------------------------------
# 4. Keep original attack labels
# --------------------------------------------------

print("\nAttack types in dataset:")
print(df["Label"].value_counts())


# --------------------------------------------------
# 5. Convert labels to binary
#
# BENIGN  -> 0 (NORMAL)
# ATTACKS -> 1 (MALICIOUS)
# --------------------------------------------------

df["BinaryLabel"] = df["Label"].apply(
    lambda x: 0 if x == "BENIGN" else 1
)


# --------------------------------------------------
# 6. Separate features and target
# --------------------------------------------------

X = df[features]
y = df["BinaryLabel"]


# --------------------------------------------------
# 7. Split into training and testing data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 8. Create the Network Defender
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 9. Train
# --------------------------------------------------

print("\nTraining Network Defender...")

model.fit(X_train, y_train)


# --------------------------------------------------
# 10. Test
# --------------------------------------------------

predictions = model.predict(X_test)


accuracy = accuracy_score(y_test, predictions)

print("\n========== NETWORK DEFENDER ==========")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["NORMAL", "MALICIOUS"]
    )
)


# --------------------------------------------------
# 11. Show which features the defender uses most
# --------------------------------------------------

print("\n========== FEATURE IMPORTANCE ==========")

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance.to_string(index=False))


# ==========================================
# TASK 2: SIMULATED CLOUD FIREWALL API with 1000 random or slightly modified features
# ==========================================

import numpy as np


# 1. Hide the model behind a function firewall_api
def firewall_api(flow):
    prediction = model.predict(flow)[0]
    return int(prediction)


# 2. Select 1,000 test network flows from the x_test that we used to test containing all 8 columns
attacker_samples = X_test.sample(
    n=1000,
    random_state=42
).copy()


# 3. Create random generator
rng = np.random.default_rng(42)


# 4. Slightly modify the network flows
for feature in features:

    if feature == "Destination Port":

        noise = rng.integers(-2, 3, size=1000)

        attacker_samples[feature] = (
            attacker_samples[feature] + noise
        ).clip(1, 65535) ##port numbers should be from 1 to 65535 only hence the clipping

    else:

        noise = rng.normal(
            loc=0,
            scale=0.05,
            size=1000
        )

        attacker_samples[feature] = (
            attacker_samples[feature] * (1 + noise)
        ).clip(lower=0)


# 5. Send the 1,000 requests to the firewall
results = []

for index, flow in attacker_samples.iterrows():

    request = pd.DataFrame(
        [flow.values],
        columns=features
    )

    response = firewall_api(request)

    results.append(response)


# 6. Store firewall responses
attacker_samples["Firewall_Response"] = results


# 7. Save results
attacker_samples.to_csv(
    "attacker_scouted_data.csv",
    index=False
)


# 8. Display summary
print("\n========== TASK 2 ==========")
print("Attacker requests:", len(attacker_samples))

print("\nFirewall responses:")
print(
    attacker_samples["Firewall_Response"].value_counts()
)

print("\nSaved file: attacker_scouted_data.csv")
