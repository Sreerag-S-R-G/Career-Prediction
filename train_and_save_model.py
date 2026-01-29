import pickle
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data/mldata.csv"
MODEL_PATH = "career_rf_pipeline.pkl"
TARGET = "Suggested Job Role"

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

# -----------------------------
# Feature groups (MATCH CSV)
# -----------------------------

# Text features
text_features = [
    "Interested subjects",
    "Interested Type of Books",
    "certifications",
    "workshops",
    "Type of company want to settle in?",
    "interested career area "
]

# Categorical features (Yes/No & small categories)
cat_features = [
    "self-learning capability?",
    "Extra-courses did",
    "Taken inputs from seniors or elders",
    "worked in teams ever?",
    "Introvert",
    "reading and writing skills",
    "memory capability score",
    "Management or Technical",
    "hard/smart worker"
]

# Numerical features
num_features = [
    "Logical quotient rating",
    "coding skills rating",
    "hackathons",
    "public speaking points"
]

# -----------------------------
# Preprocessing
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("subjects", TfidfVectorizer(stop_words="english"), "Interested subjects"),
        ("books", TfidfVectorizer(stop_words="english"), "Interested Type of Books"),
        ("cert", TfidfVectorizer(stop_words="english"), "certifications"),
        ("work", TfidfVectorizer(stop_words="english"), "workshops"),
        ("company", TfidfVectorizer(stop_words="english"), "Type of company want to settle in?"),
        ("career", TfidfVectorizer(stop_words="english"), "interested career area "),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", "passthrough", num_features)
    ]
)

# -----------------------------
# Pipeline Model
# -----------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=10,
        n_jobs=-1
    ))
])

# -----------------------------
# Train
# -----------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

model.fit(X, y)

# -----------------------------
# Save model
# -----------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as career_rf_pipeline.pkl")
