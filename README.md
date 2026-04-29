# AdmitIQ-AI-Powered-College-Admission-Screening-System
AI-Based College Admission Screening System automates admissions using Machine Learning and Web Technology. It speeds decisions, recommends courses, and improves efficiency. Future upgrades can include real datasets, cloud deployment, email alerts, biometric login, and advanced analytics.
# AI-Based College Admission Screening System

> An end-to-end machine learning project that automates college admission decisions and recommends courses using Python, Flask, scikit-learn, and SQLite.

---

##  1. Project Overview

This system uses machine learning to evaluate student applications and deliver instant admission eligibility decisions. A student fills in their academic details (10th %, 12th %, entrance exam score, etc.) and the system returns a Yes/No admission decision plus a personalised course recommendation — all within under one second.

**Real-world use cases:**
- College admission offices to pre-screen thousands of applications
- Student self-assessment before applying
- Automated shortlisting for entrance counselling

**Technologies Used:** Python · Flask · scikit-learn · SQLite · HTML/CSS/JavaScript · Chart.js

---

## 2. System Architecture

```
┌─────────────────────────────┐
│      Student Browser        │
│   index.html / form.html    │
└────────────┬────────────────┘
             │  fetch() POST /predict
             ▼
┌─────────────────────────────┐
│    Flask Backend (app.py)   │
│  Route: /predict            │
│  • Validate input           │
│  • Encode features          │
│  • Scale with StandardScaler│
└────────────┬────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌──────────┐   ┌──────────────┐
│Admission │   │   Course     │
│  Model   │   │  Recommender │
│(Random   │   │(Decision Tree│
│ Forest)  │   │    .pkl)     │
└────┬─────┘   └──────┬───────┘
     └───────┬────────┘
             ▼
┌─────────────────────────────┐
│   SQLite Database           │
│   students + users tables   │
└─────────────────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Result Page (result.html) │
│   ✓ Eligible / ✗ Rejected   │
│   Course Recommended     │
└─────────────────────────────┘
```

---

## 3. Project Structure

```
admission_system/
├── app.py                        ← Flask backend (main entry point)
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
│
├── model/
│   ├── train_model.py            ← ML training script
│   ├── admission_dataset.csv     ← Generated training data (150 rows)
│   ├── admission_model.pkl       ← Trained Random Forest model
│   ├── course_model.pkl          ← Trained Decision Tree recommender
│   ├── scaler.pkl                ← StandardScaler
│   └── label_encoders.pkl        ← LabelEncoders for gender/course
│
├── frontend/
│   ├── templates/
│   │   ├── index.html            ← Home page
│   │   ├── form.html             ← Student application form
│   │   ├── result.html           ← Prediction result page
│   │   ├── login.html            ← Admin login
│   │   └── admin.html            ← Admin dashboard
│   └── static/
│       ├── css/                  ← (optional custom CSS)
│       └── js/                   ← (optional custom JS)
│
└── database/
    ├── schema.sql                ← SQL table definitions
    └── admission.db              ← SQLite database (auto-created)
```

---

## 4. ML Model Details

| Component | Algorithm | Accuracy |
|-----------|-----------|----------|
| Admission Eligibility | Random Forest (100 trees) | **93.3%** |
| Course Recommendation | Decision Tree (depth=6) | **95.5%** |

**Features used:**
- Age
- Gender (encoded)
- 10th Grade Marks
- 12th Grade Marks
- Entrance Exam Score
- Preferred Course (encoded)

**Admission Rule (training logic):**
- avg(10th, 12th) ≥ 65% AND entrance ≥ 55 → Admitted (full course choice)
- avg ≥ 55% AND entrance ≥ 45 → Admitted (Business Admin)
- Otherwise → Rejected

---

## 5. API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Home page |
| GET | `/form` | Application form |
| POST | `/predict` | ML prediction (JSON) |
| GET | `/result` | Show prediction result |
| GET/POST | `/login` | Admin login |
| GET | `/admin` | Admin dashboard |
| DELETE | `/api/delete/<id>` | Delete application |
| GET | `/api/stats` | Raw stats JSON |
| GET | `/logout` | Clear session |

## 5. Setup & Running

### Step 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the ML models (first time only)
```bash
cd model
python train_model.py
cd ..
```

### Step 3 — Run the Flask server
```bash
python app.py
```

### Step 4 — Open in browser
```
http://127.0.0.1:5000
```

### Admin Dashboard
```
http://127.0.0.1:5000/login
Username: admin
Password: admin123
```

---

## 7. Database

SQLite database at `database/admission.db` — auto-created on first run.

**students table columns:**
`id, name, email, age, gender, marks_10, marks_12, entrance_score, preferred_course, admitted, recommended_course, confidence, submitted_at`

---

## 8. Sample Dataset

150-row CSV generated by `model/train_model.py` with columns:
`age, gender, marks_10, marks_12, entrance_score, preferred_course, admitted, recommended_course`

---

## 9. Admin Features

- View all applications in a searchable table
- Delete individual records
- Score distribution scatter plot (Chart.js)
- Course allocation doughnut chart
- Admission vs rejection bar chart
- Acceptance rate statistics

---

## 10. Changing Admin Password

Edit `app.py` line:
```python
admin_pw = hashlib.sha256("YOUR_NEW_PASSWORD".encode()).hexdigest()
```

Or run in Python:
```python
import hashlib, sqlite3
pw = hashlib.sha256("newpassword".encode()).hexdigest()
conn = sqlite3.connect("database/admission.db")
conn.execute("UPDATE users SET password=? WHERE username='admin'", (pw,))
conn.commit()
```
