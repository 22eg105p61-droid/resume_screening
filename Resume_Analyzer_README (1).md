# 📂 Resume Analyzer System

## ✅ 1. train_model.py --- Model Training Explained in Detail

### 🎯 Purpose:

This file creates and trains a simple machine learning model that
classifies resumes into job categories based on the skills mentioned.

------------------------------------------------------------------------

### ✅ Step-by-Step Explanation

#### 1️⃣ Import Libraries

-   pandas: For handling data in tables (DataFrame).
-   pickle: To save the trained model and label encoder for later use.
-   TfidfVectorizer: Converts resume text into numeric values by
    calculating word importance.
-   LogisticRegression: A machine learning algorithm that predicts
    categories.
-   LabelEncoder: Converts text labels (like "Data Science") into
    numbers.
-   train_test_split: Splits the data into training and test sets.
-   make_pipeline: Combines vectorizer and classifier in one pipeline.

#### 2️⃣ Define Category-Specific Skills

A predefined dictionary of skills required for each job category.

#### 3️⃣ Create a Synthetic Dataset

We simulate resume data by combining the skills into a string per
category.

#### 4️⃣ Convert Data to DataFrame

Converts the list of dictionaries into a table (DataFrame).

#### 5️⃣ Encode Labels

Converts category names into numeric labels (e.g., "Data Science" → 0).

#### 6️⃣ Split Data

80% data for training, 20% for testing.

#### 7️⃣ Create the Model Pipeline

Converts text → numerical features → logistic regression classifier.

#### 8️⃣ Train the Model

The model learns patterns from the training data.

#### 9️⃣ Save Model and Resources

Saves the trained model, encoder, and skill mapping for later use.

### ✅ Final Output

Model and resources saved successfully!

------------------------------------------------------------------------

## ✅ 2. app.py --- Streamlit Resume Analyzer App Detailed Explanation

### 🎯 Purpose:

User interface where users upload resumes and get analysis: - Name,
email, phone, age extraction. - Skill extraction. - Resume score. - Best
job category recommendation. - Learning resources and improvement
suggestions.

------------------------------------------------------------------------

### ✅ Step-by-Step Explanation

#### 1️⃣ Load Libraries

streamlit, pdfplumber, re, spacy, pickle, datetime

#### 2️⃣ Load SpaCy Model

NLP model helps extract names.

#### 3️⃣ Load Saved Model & Resources

Load model.pkl, encoder.pkl, and category_skills.pkl.

#### 4️⃣ Extend CATEGORY_SKILLS

Added roles like Frontend, Backend, Full Stack Developer.

#### 5️⃣ Define ALL_SKILLS

A large set of all known skills for easy matching.

#### 6️⃣ Learning Resources

Map skills to helpful learning links.

#### 7️⃣ Data Extraction Functions

-   extract_text_from_pdf()
-   extract_name()
-   extract_birth_year()
-   extract_email()
-   extract_phone_number()
-   extract_skills()
-   clean_resume()
-   predict_category()
-   calculate_resume_score()
-   find_best_category()
-   predict_selection_chance()

#### 8️⃣ Streamlit App Workflow

1.  Select job category (or Predict).
2.  Upload PDF resume.
3.  Extract: Name, Email, Phone, Age.
4.  Predict Category + Skills + Score.
5.  Predict Selection Chance.
6.  Show Results:
    -   Name, Email, Phone, Age, Category, Skills, Resume Score,
        Selection Chance.
7.  Show Improvement Recommendations.
8.  Show Learning Resources for missing skills.

### ✅ Final Summary

-   train_model.py: Trains and saves the ML model.
-   app.py: Provides the interactive resume analysis system.
