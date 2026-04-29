"""
AI-Based College Admission Screening System
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# ─────────────────────────────────────────────
# BASE PATH
# ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATASET (150 rows)
# ─────────────────────────────────────────────
np.random.seed(42)
n = 150

def generate_data(n):
    records = []
    courses = [
        "Computer Science",
        "Artificial Intelligence and Machine Learning",
        "Mechanical Engineering",
        "Electronics",
        "Civil Engineering",
        "Biotechnology",
        "Business Administration"
    ]

    for i in range(n):
        marks_10 = round(np.random.uniform(45, 99), 1)
        marks_12 = round(np.random.uniform(45, 99), 1)
        entrance = round(np.random.uniform(20, 100), 1)
        age = int(np.random.randint(16, 22))
        gender = np.random.choice(["Male", "Female", "Other"])
        course_pref = np.random.choice(courses)

        # Admission rule
        avg_marks = (marks_10 + marks_12) / 2

        if avg_marks >= 65 and entrance >= 55:
            admitted = 1

            if entrance >= 85 and marks_12 >= 80:
                course = "Artificial Intelligence and Machine Learning"
            if entrance >= 80 and marks_12 >= 80:
                course = "Computer Science"
            elif marks_12 >= 75:
                course = "Electronics"
            elif marks_12 >= 70:
                course = "Mechanical Engineering"
            elif marks_12 >= 65:
                course = "Civil Engineering"
            elif marks_12 >= 60:
                course = "Biotechnology"
            else:
                course = "Business Administration"

        elif avg_marks >= 55 and entrance >= 45:
            admitted = 1
            course = "Business Administration"

        else:
            admitted = 0
            course = "Not Admitted"

        records.append({
            "age": age,
            "gender": gender,
            "marks_10": marks_10,
            "marks_12": marks_12,
            "entrance_score": entrance,
            "preferred_course": course_pref,
            "admitted": admitted,
            "recommended_course": course
        })

    return pd.DataFrame(records)

df = generate_data(n)

# Save dataset
df.to_csv(os.path.join(MODEL_DIR, "admission_dataset.csv"), index=False)

print(" Dataset saved: admission_dataset.csv")
print(df["admitted"].value_counts())

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
le_gender = LabelEncoder()
le_course = LabelEncoder()
le_rec = LabelEncoder()

df["gender_enc"] = le_gender.fit_transform(df["gender"])
df["course_enc"] = le_course.fit_transform(df["preferred_course"])
df["rec_enc"] = le_rec.fit_transform(df["recommended_course"])

features = [
    "age",
    "gender_enc",
    "marks_10",
    "marks_12",
    "entrance_score",
    "course_enc"
]

X = df[features]
y_admit = df["admitted"]
y_course = df["rec_enc"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_admit,
    test_size=0.2,
    random_state=42
)

# ─────────────────────────────────────────────
# 3. TRAIN ADMISSION MODEL
# ─────────────────────────────────────────────
clf_admit = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

clf_admit.fit(X_train, y_train)

y_pred = clf_admit.predict(X_test)

print(f"\n Admission Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────────
# 4. TRAIN COURSE RECOMMENDATION MODEL
# ─────────────────────────────────────────────
admitted_df = df[df["admitted"] == 1].copy()

X_c = admitted_df[features]
y_c = admitted_df["rec_enc"]

X_c_scaled = scaler.transform(X_c)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_c_scaled,
    y_c,
    test_size=0.2,
    random_state=42
)

clf_course = DecisionTreeClassifier(
    max_depth=6,
    random_state=42
)

clf_course.fit(Xc_train, yc_train)

yc_pred = clf_course.predict(Xc_test)

print(f"\n Course Model Accuracy: {accuracy_score(yc_test, yc_pred)*100:.2f}%")

# ─────────────────────────────────────────────
# 5. SAVE ALL ARTIFACTS 
# ─────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "admission_model.pkl"), "wb") as f:
    pickle.dump(clf_admit, f)

with open(os.path.join(MODEL_DIR, "course_model.pkl"), "wb") as f:
    pickle.dump(clf_course, f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "wb") as f:
    pickle.dump({
        "gender": le_gender,
        "preferred_course": le_course,
        "recommended_course": le_rec
    }, f)

print("\n All models and encoders saved successfully!")
print("   → admission_model.pkl")
print("   → course_model.pkl")
print("   → scaler.pkl")
print("   → label_encoders.pkl")
