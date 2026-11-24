import pandas as pd
import glob

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Features we want to use
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
# 2. Load all CSV files
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

print("\nTotal rows:", len(df))
print("Total columns loaded:", len(df.columns))


# --------------------------------------------------
# 3. Clean the data
# --------------------------------------------------

df = df.replace(
    [float("inf"), float("-inf")],
    float("nan")
)

df = df.dropna()


# --------------------------------------------------
# 4. Convert labels to NORMAL / MALICIOUS
# --------------------------------------------------

df["Label"] = df["Label"].apply(
    lambda x: 0 if x == "BENIGN" else 1
)

print("\nLabels:")
print("Normal:", (df["Label"] == 0).sum())
print("Malicious:", (df["Label"] == 1).sum())


# --------------------------------------------------
# 5. Separate features and target
# --------------------------------------------------

X = df[features]
y = df["Label"]


# --------------------------------------------------
# 6. Split: 80% training, 20% testing
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", len(X_train))
print("Testing data:", len(X_test))


# --------------------------------------------------
# 7. Train Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 8. Make predictions
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# 9. Calculate accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print(" NETWORK DEFENDER RESULTS")
print("==============================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print(
    f"\nOur network defender model is "
    f"{accuracy * 100:.2f}% accurate at distinguishing "
    f"normal and malicious network traffic."
)


# --------------------------------------------------
# 10. Detailed results
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["NORMAL", "MALICIOUS"]
    )
)
