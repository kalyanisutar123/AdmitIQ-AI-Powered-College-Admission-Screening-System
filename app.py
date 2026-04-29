"""
AI-Based College Admission Screening System
Flask Backend — app.py
Corrected / Clean / Efficient Version
"""

from flask import Flask, request, jsonify, render_template, redirect, session
import sqlite3
import pickle
import numpy as np
import pandas as pd
import os
import hashlib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==================================================
# APP CONFIG
# ==================================================
BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE, "frontend", "templates"),
    static_folder=os.path.join(BASE, "frontend", "static")
)

app.secret_key = "admission_secret_key_2024"

MODEL_DIR = os.path.join(BASE, "model")
DATA_DIR = os.path.join(BASE, "data")
DB_PATH = os.path.join(DATA_DIR, "admission.db")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ==================================================
# TRAIN MODEL
# ==================================================
def train_models():
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        "age": np.random.randint(17, 25, n),
        "gender": np.random.choice(["Male", "Female"], n),
        "marks_10": np.random.randint(40, 100, n),
        "marks_12": np.random.randint(40, 100, n),
        "entrance_score": np.random.randint(40, 100, n),
        "preferred_course": np.random.choice([
            "Computer Science",
            "Artificial Intelligence and Machine Learning",
            "Mechanical Engineering",
            "Civil Engineering",
            "Electronics",
            "Business Administration",
            "Biotechnology"
        ], n)
    })

    df["admitted"] = (
        (df["marks_10"] + df["marks_12"] + df["entrance_score"]) / 3 > 65
    ).astype(int)

    df["recommended_course"] = df["preferred_course"]
    df.loc[df["admitted"] == 0, "recommended_course"] = "Not Admitted"

    le_gender = LabelEncoder()
    le_course = LabelEncoder()
    le_rec = LabelEncoder()

    df["gender"] = le_gender.fit_transform(df["gender"])
    df["preferred_course"] = le_course.fit_transform(df["preferred_course"])
    df["recommended_course"] = le_rec.fit_transform(df["recommended_course"])

    encoders = {
        "gender": le_gender,
        "preferred_course": le_course,
        "recommended_course": le_rec
    }

    X = df[[
        "age",
        "gender",
        "marks_10",
        "marks_12",
        "entrance_score",
        "preferred_course"
    ]]

    y = df["admitted"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    with open(os.path.join(MODEL_DIR, "admission_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(MODEL_DIR, "course_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    print("Models trained & saved")


if not os.path.exists(os.path.join(MODEL_DIR, "admission_model.pkl")):
    train_models()

# ==================================================
# LOAD MODELS
# ==================================================
with open(os.path.join(MODEL_DIR, "admission_model.pkl"), "rb") as f:
    admission_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "course_model.pkl"), "rb") as f:
    course_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "rb") as f:
    label_encoders = pickle.load(f)

# ==================================================
# DATABASE
# ==================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            age INTEGER,
            gender TEXT,
            marks_10 REAL,
            marks_12 REAL,
            entrance_score REAL,
            preferred_course TEXT,
            admitted INTEGER,
            recommended_course TEXT,
            confidence REAL,
            submitted_at TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT
        )
    """)

    admin_pass = hashlib.sha256("admin123".encode()).hexdigest()

    c.execute("""
        INSERT OR IGNORE INTO users(username,password,role)
        VALUES(?,?,?)
    """, ("admin", admin_pass, "admin"))

    conn.commit()
    conn.close()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


init_db()

# ==================================================
# HELPER
# ==================================================
def encode_input(age, gender, marks_10, marks_12, entrance_score, course):
    le_gender = label_encoders["gender"]
    le_course = label_encoders["preferred_course"]

    if gender not in le_gender.classes_:
        gender = le_gender.classes_[0]

    if course not in le_course.classes_:
        course = le_course.classes_[0]

    g = le_gender.transform([gender])[0]
    c = le_course.transform([course])[0]

    X = np.array([[age, g, marks_10, marks_12, entrance_score, c]])
    return scaler.transform(X)

# ==================================================
# ROUTES
# ==================================================
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/form')
def form():
    return render_template("form.html")


@app.route('/result')
def result():
    return render_template("result.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = hashlib.sha256(
            request.form["password"].encode()
        ).hexdigest()

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        ).fetchone()
        conn.close()

        if user:
            session["user"] = username
            session["role"] = user["role"]
            return redirect("/admin")

        return render_template("login.html", error="Invalid Credentials")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


@app.route("/admin")
def admin():
    if session.get("role") != "admin":
        return redirect("/login")

    conn = get_db()

    students = conn.execute(
        "SELECT * FROM students ORDER BY id DESC"
    ).fetchall()

    stats = conn.execute("""
        SELECT
        COUNT(*) total,
        SUM(admitted) admitted,
        COUNT(*)-SUM(admitted) rejected
        FROM students
    """).fetchone()

    conn.close()

    return render_template(
        "admin.html",
        students=students,
        stats=stats
    )

# ==================================================
# API PREDICT
# ==================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        name = data.get("name")
        email = data.get("email")
        age = int(data.get("age"))
        gender = data.get("gender")
        marks_10 = float(data.get("marks_10"))
        marks_12 = float(data.get("marks_12"))
        entrance = float(data.get("entrance_score"))
        course = data.get("preferred_course")

        X = encode_input(age, gender, marks_10, marks_12, entrance, course)

        admitted = int(admission_model.predict(X)[0])

        confidence = round(
            float(admission_model.predict_proba(X)[0][admitted]) * 100, 1
        )

        if admitted == 1:
            recommended = course
        else:
            recommended = "Not Admitted"

        result = {
            "name": name,
            "admitted": admitted,
            "recommended_course": recommended,
            "confidence": confidence
        }

        conn = get_db()
        conn.execute("""
            INSERT INTO students(
                name,email,age,gender,
                marks_10,marks_12,entrance_score,
                preferred_course,admitted,
                recommended_course,confidence,
                submitted_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            name, email, age, gender,
            marks_10, marks_12, entrance,
            course, admitted,
            recommended, confidence,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        conn.close()

        session["result_data"] = result

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


@app.route("/api/delete/<int:id>", methods=["DELETE"])
def delete_student(id):
    conn = get_db()
    conn.execute("DELETE FROM students WHERE id=?", (id,))
    conn.commit()
    conn.close()

    return jsonify({"success": True})

# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    print("Running on http://127.0.0.1:5000")
    print("Admin Login: admin / admin123")
    app.run(debug=True)
