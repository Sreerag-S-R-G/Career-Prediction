🎯 Career Prediction System using Machine Learning

A machine learning–based career prediction system that recommends suitable job roles for students based on their skills, interests, certifications, and behavioral attributes. The system uses a Random Forest classifier with a pipeline-based architecture to ensure reliable preprocessing and prediction.

📌 Project Overview

Choosing the right career is a challenging task for students due to the wide variety of options and lack of structured guidance. This project addresses the problem by using machine learning techniques to analyze student profiles and provide data-driven career recommendations.

The system predicts a primary career option and also suggests alternative career paths, making it more practical and flexible than traditional single-output systems.

🚀 Features

Predicts suitable career roles using machine learning

Supports multiple career recommendations (Top-3 predictions)

Handles numerical, categorical, and text-based inputs

Uses TF-IDF vectorization for text features

Pipeline-based preprocessing for consistency and robustness

Clean and user-friendly Streamlit web interface

Model persistence using Pickle

🧠 Machine Learning Models

The following models were evaluated:

Model	Accuracy
Decision Tree	Lower
Support Vector Machine (SVM)	83%
Random Forest (Final Model)	87%

Random Forest was selected due to its higher accuracy, reduced overfitting, and better generalization performance.



⚙️ Technologies Used

Programming Language: Python

Libraries:

Pandas

NumPy

Scikit-learn

Streamlit

ML Techniques:

Random Forest Classifier

One-Hot Encoding

TF-IDF Vectorization

Pipeline Architecture

🧩 Feature Engineering

Numerical features: Used directly (ratings and scores)

Categorical features: Encoded using One-Hot Encoding

Text features: Converted using TF-IDF vectorization

Pipeline: Ensures identical preprocessing during training and prediction

▶️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/Sreerag-S-R-G/Career-Prediction.git
cd Career-Prediction

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train and Save the Model
python train_and_save_model.py

4️⃣ Run the Streamlit App
streamlit run app.py

🖥️ Streamlit Interface

The Streamlit app allows users to:

Enter their skills, interests, and preferences

View the best career match

See Top-3 career recommendations with confidence scores

📈 Results

Achieved 87% accuracy using Random Forest

Predictions change dynamically with user inputs

Demonstrates real-world applicability of machine learning

🔮 Future Enhancements

Integration with real-time job market data

Salary prediction and skill gap analysis

Mobile and web deployment

Personalized learning and certification recommendations

Use of deep learning models for improved performance

⚠️ Limitations

Prediction quality depends on dataset size and balance

Cannot fully replace human career counseling

Limited to predefined career roles in the dataset

👨‍💻 Author

Sreerag K
Built as an academic and practical machine learning project for career guidance.

📜 License

This project is intended for educational and research purposes.
