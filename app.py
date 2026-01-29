import pickle
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Career Prediction",
    page_icon="🎯",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("career_rf_pipeline.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🎯 Career Prediction System</h1>
    <p style='text-align:center; font-size:16px;'>
    Discover career paths based on your skills, interests, and abilities
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Section: Skills & Abilities
# -----------------------------
st.subheader("📊 Skills & Abilities")

col1, col2 = st.columns(2)

with col1:
    logical = st.slider("Logical Quotient", 0, 10, 5)
    coding = st.slider("Coding Skills", 0, 10, 5)

with col2:
    hackathons = st.slider("Hackathons Participated", 0, 10, 0)
    speaking = st.slider("Public Speaking Skills", 0, 10, 5)

# -----------------------------
# Section: Learning & Personality
# -----------------------------
st.subheader("🧠 Learning & Personality")

col3, col4 = st.columns(2)

with col3:
    self_learning = st.selectbox("Self-learning Capability", ["Yes", "No"])
    teamwork = st.selectbox("Team Work Experience", ["Yes", "No"])
    introvert = st.selectbox("Introvert", ["Yes", "No"])

with col4:
    extra_courses = st.selectbox("Extra Courses Completed", ["Yes", "No"])
    rw_skill = st.selectbox("Reading & Writing Skills", ["poor", "medium", "excellent"])
    memory = st.selectbox("Memory Capability", ["poor", "medium", "excellent"])

# -----------------------------
# Section: Preferences
# -----------------------------
st.subheader("🎓 Interests & Preferences")

col5, col6 = st.columns(2)

with col5:
    domain = st.selectbox("Management or Technical", ["Management", "Technical"])
    work_style = st.selectbox("Work Style", ["Smart worker", "Hard worker"])
    subject = st.selectbox(
        "Interested Subject",
        ["programming", "Management", "data engineering", "networks",
         "Software Engineering", "cloud computing", "parallel computing",
         "IOT", "Computer Architecture", "hacking"]
    )

with col6:
    books = st.selectbox(
        "Preferred Book Type",
        ["Series", "Autobiographies", "Travel", "Guide", "Health", "Journals",
         "Anthology", "Dictionaries", "Prayer books", "Art", "Encyclopedias",
         "Religion-Spirituality", "Action and Adventure", "Comics", "Horror",
         "Satire", "Self help", "History", "Cookbooks", "Math", "Biographies",
         "Drama", "Diaries", "Science fiction", "Poetry", "Romance",
         "Science", "Trilogy", "Fantasy", "Childrens", "Mystery"]
    )

    certification = st.selectbox(
        "Certification",
        ["information security", "shell programming", "r programming",
         "distro making", "machine learning", "full stack",
         "hadoop", "app development", "python"]
    )

    workshop = st.selectbox(
        "Workshop Attended",
        ["Testing", "database security", "game development", "data science",
         "system designing", "hacking", "cloud computing", "web technologies"]
    )

# -----------------------------
# Section: Career Goals
# -----------------------------
st.subheader("🚀 Career Goals")

company = st.selectbox(
    "Preferred Company Type",
    ["BPA", "Cloud Services", "product development",
     "Testing and Maintainance Services", "SAaS services",
     "Web Services", "Finance", "Sales and Marketing",
     "Product based", "Service Based"]
)

career_area = st.selectbox(
    "Interested Career Area",
    ["testing", "system developer", "Business process analyst",
     "security", "developer", "cloud computing"]
)

# -----------------------------
# Prediction
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)

if st.button("🔍 Predict Career", use_container_width=True):

    input_df = pd.DataFrame([{
        "Logical quotient rating": logical,
        "coding skills rating": coding,
        "hackathons": hackathons,
        "public speaking points": speaking,
        "self-learning capability?": self_learning,
        "Extra-courses did": extra_courses,
        "Taken inputs from seniors or elders": "Yes",  # default
        "worked in teams ever?": teamwork,
        "Introvert": introvert,
        "reading and writing skills": rw_skill,
        "memory capability score": memory,
        "Management or Technical": domain,
        "hard/smart worker": work_style,
        "Interested subjects": subject,
        "Interested Type of Books": books,
        "certifications": certification,
        "workshops": workshop,
        "Type of company want to settle in?": company,
        "interested career area ": career_area
    }])

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    classes = model.classes_

    top3_idx = np.argsort(proba)[-3:][::-1]

    st.success(f"🎯 **Best Career Match:** {prediction}")

    st.markdown("### 🔝 Top Career Recommendations")
    for i in top3_idx:
        st.write(f"- **{classes[i]}** (Confidence: {proba[i]:.2f})")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:14px; color:gray;'>
    Built by <b>Sreerag K</b>
    </p>
    """,
    unsafe_allow_html=True
)
