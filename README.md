# resume_screening

📚 Detailed Explanation of train_model.py
✅ Purpose of the File

The purpose of train_model.py is to train a machine learning model that can classify a candidate's resume into different job categories based on the skills written in the resume.
After training, the model and related resources are saved so they can be used later in a web app or service to predict the correct category of a resume.

✅ Step-by-Step Breakdown
🔧 1️⃣. Import Necessary Libraries
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


pandas: Used to store and manage data in a table structure called DataFrame.

pickle: Used to save Python objects (like trained models) to files, so they can be reused later.

TfidfVectorizer: Converts text into numbers by calculating how important each word is in a document compared to the whole dataset.

LogisticRegression: A simple and effective classification algorithm used to categorize the resume.

LabelEncoder: Converts category names (like "Data Science") into numbers (like 0, 1, 2) because ML models work with numbers.

train_test_split: Divides data into two parts: training data and testing data.

make_pipeline: Helps combine multiple steps into a clean workflow (e.g., converting text → classifying).

🗂️ 2️⃣. Define Skills for Each Job Category
CATEGORY_SKILLS = {
    "Data Science": ["python", "sql", "r", "pandas", ...],
    "Data Analyst": ["sql", "python", "pandas", "excel", ...],
    "Software Development": ["python", "javascript", "typescript", ...],
    "UI/UX Designer": ["figma", "adobe xd", "html", "css", ...],
    "AI/ML": ["python", "tensorflow", "pytorch", "keras", ...]
}


This dictionary holds important skills related to each job category.

Example:

If a resume mentions "python" and "tensorflow", it likely belongs to "AI/ML" or "Data Science".

These skills are used later to help explain the predicted category to the user.

📝 3️⃣. Prepare a Small Example Dataset
data = {
    'resume_text': [
        "Skilled in python pandas numpy sql tensorflow pytorch matplotlib seaborn tableau aws spark",
        "Expert in sql python pandas excel tableau power bi mysql snowflake looker studio",
        "Full-stack developer with python java spring boot postgresql redis kubernetes pytest",
        ...
    ],
    'category': [
        "Data Science", "Data Analyst", "Software Development", ...
    ]
}


This is a small dataset we created manually for demonstration.

Each entry consists of:

resume_text: A string that simulates what a resume might say.

category: The correct job category that this text belongs to.

✅ 4️⃣. Create a Pandas DataFrame
df = pd.DataFrame(data)


The data is converted into a table-like structure (DataFrame) for easy data manipulation.

🔢 5️⃣. Convert Categories into Numbers
encoder = LabelEncoder()
y = encoder.fit_transform(df['category'])


Converts textual categories into numeric form (because machine learning models need numbers).

Example mapping:

"Data Science" → 0

"Data Analyst" → 1

"Software Development" → 2

And so on...

✂️ 6️⃣. Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    df['resume_text'], y, test_size=0.2, random_state=42
)


X_train, y_train: Data used to train the model (80% of the dataset).

X_test, y_test: Data used to test how well the model performs (20% of the dataset).

random_state=42 ensures reproducibility of the split.

🧱 7️⃣. Create the ML Model Pipeline
model = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    LogisticRegression(max_iter=1000)
)


The pipeline does two things automatically in order:

TfidfVectorizer(stop_words='english'):

Converts raw resume text into a matrix of numbers.

Removes common words like "the", "and", etc., which don’t add value.

LogisticRegression(max_iter=1000):

Learns patterns from training data to classify a resume into a category.

max_iter=1000 ensures the model has enough iterations to converge.

🚀 8️⃣. Train the Model
model.fit(X_train, y_train)


The model learns the relationship between resume text and categories.

Internally:

TF-IDF calculates word importance.

Logistic Regression tries to find patterns based on those word counts.

💾 9️⃣. Save Model and Related Resources
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

with open('category_skills.pkl', 'wb') as f:
    pickle.dump(CATEGORY_SKILLS, f)


The following are saved into files:

Trained Model → model.pkl

Label Encoder (Category to Number Mapping) → encoder.pkl

Predefined Category Skills Dictionary → category_skills.pkl

👉 Why save them?
So later, you can load the trained model and use it directly without retraining.

✅ 10️⃣. Final Success Message
print("Model and resources saved successfully!")


✅ Detailed Explanation of Streamlit Resume Analyzer App
🎯 Purpose of the App

This app helps candidates analyze their resumes by:

Predicting the most suitable job category.

Extracting personal details (name, email, phone, age).

Extracting skills.

Scoring the resume based on skill match.

Recommending how to improve it.

Suggesting learning resources for missing skills.

🔧 Step-by-Step Breakdown
1️⃣ Import Required Libraries
import streamlit as st
import pdfplumber
import re
import spacy
import pickle
from datetime import datetime


streamlit (st): Library to create web apps easily.

pdfplumber: Extract text from PDF files.

re: For regular expressions (finding patterns like emails, phone numbers).

spacy: NLP tool to extract names or entities.

pickle: Load pre-trained model and other resources.

datetime: Calculate age from birth year.

2️⃣ Load SpaCy NLP Model (For Name Extraction)
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()


Loads the SpaCy small English model (for recognizing people’s names).

@st.cache_resource ensures it's loaded only once for performance.

3️⃣ Load Pre-trained Model & Resources
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('category_skills.pkl', 'rb') as f:
    CATEGORY_SKILLS = pickle.load(f)


Loads the machine learning model, label encoder, and predefined skills for categories.

4️⃣ Update CATEGORY_SKILLS

Adds new job categories and related skills into CATEGORY_SKILLS dictionary.

Example:

Frontend Developer → HTML, CSS, React, etc.

Backend Developer → Node.js, Flask, MongoDB, etc.

This helps the app predict more job categories.

5️⃣ Prepare All Skills Set
ALL_SKILLS = set()
for skills in CATEGORY_SKILLS.values():
    ALL_SKILLS.update(skill.lower() for skill in skills)
ALL_SKILLS.update([... academic terms, variations ...])


Converts all predefined skills into lowercase and collects them into ALL_SKILLS.

This helps when matching skills in the resume later.

6️⃣ Learning Resources Dictionary
LEARNING_RESOURCES = {
    "python": ["https://...", ...],
    ...
}


Contains links for recommended tutorials/courses for each skill.

7️⃣ Extract Text from Uploaded PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            text += page_text
    return text


Reads each page of the uploaded PDF and extracts the text.

If the PDF is scanned (images only), it won’t work.

8️⃣ Extract Name from Resume
def extract_name(text, skills_set=ALL_SKILLS):
    - Check first and second lines if they look like a name (all words starting with capital letters).
    - Use SpaCy to extract any PERSON entity.


Tries multiple ways to extract the person's name from the resume text.

Falls back to "Unknown" if no name is found.

9️⃣ Extract Age
def extract_birth_year(text):
    pattern = r'\b(19|20)\d{2}\b'
    - Search for any 4-digit year.
    - Calculate age by subtracting birth year from current year.


Returns estimated age or None if not found.

🔟 Extract Email and Phone
def extract_email(text):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    return match.group() or "None"

def extract_phone_number(text):
    pattern = r'\b(?:\+?1\s*?)?(?:\(\d{3}\)?|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b'
    return match.group() or "None"


Uses regex to extract first email and phone number found in the text.

1️⃣1️⃣ Extract Skills
def extract_skills(text, target_skills):
    - Loop through target_skills (like "python", "flask", etc.).
    - Search for variations in the resume text.
    - Add matched skills to a set and return them as a list.


Helps identify which of the target skills are present in the resume.

1️⃣2️⃣ Clean Resume Text
def clean_resume(resume_str):
    - Remove URLs, emails, and phone numbers from the text.
    - Helps prevent leakage of personal info during ML model prediction.

1️⃣3️⃣ Predict Job Category with Pre-trained Model
def predict_category(resume_text):
    - Clean the resume.
    - Use the trained ML model to predict the category.
    - Map numeric label back to category name.

1️⃣4️⃣ Calculate Resume Score
def calculate_resume_score(extracted_skills, target_skills):
    score = (number of matched skills / total target skills) * 100
    return score rounded to 2 decimal places


Simple percentage score of how well the resume matches the expected skills.

1️⃣5️⃣ Find Best Category (Auto-Prediction)
def find_best_category(resume_text, category_skills):
    For every category in CATEGORY_SKILLS:
        - Extract skills from resume.
        - Calculate score.
    Return the category with the highest score.
    
Fallback:
    If no category matches well, use the ML model to predict.

1️⃣6️⃣ Predict Selection Chance
def predict_selection_chance(score, ...):
    - High (if score > 80%)
    - Moderate (50–80%)
    - Low (<50%)
    
Provides:
    - Explanation of the chance.
    - Missing skills and recommendations.
    - Safer strategy (alternative roles).

✅ Streamlit Web App Flow

Select Category

User selects job category or leaves to "Predict".

Upload Resume

PDF file uploaded.

Extract Information

Extract name, email, phone, age, skills.

Determine Job Category

Based on selected option or model prediction.

Predict Resume Score and Selection Chance

Display Results

Name, email, age, phone number.

Predicted or selected category.

Extracted skills.

Resume score (%).

Selection chance description.

Recommendations

How to improve the resume.

Learning resources for missing skills.


This message confirms that everything (model, encoder, skills) is stored successfully and ready for future predictions.
