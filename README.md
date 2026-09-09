Employee Attrition Prediction AI

An AI-powered Machine Learning application designed to predict employee attrition based on demographic, job-related, compensation, satisfaction, and work-life balance factors.

Overview

Employee attrition is an important challenge for organizations because losing employees can increase recruitment costs, reduce productivity, and affect organizational performance.

This project uses Machine Learning to analyze employee-related features and estimate whether an employee is likely to leave the organization.

The project combines:

Data Analysis
Data Preprocessing
Feature Engineering
Machine Learning
Model Evaluation
Single Employee Prediction
Interactive Dashboard
Model Deployment
Project Objective

The main objective of this project is to build a Machine Learning system capable of predicting employee attrition using information such as:

Age
Job Level
Monthly Income
Distance From Home
Job Satisfaction
Environment Satisfaction
Work-Life Balance
Overtime
Business Travel
Department
Job Role
Marital Status
Education
Total Working Years
Years At Company
Years In Current Role
Years Since Last Promotion
Years With Current Manager
Features
Numerical Features

The model uses numerical employee attributes including:

Age
Daily Rate
Distance From Home
Education
Environment Satisfaction
Hourly Rate
Job Involvement
Job Level
Job Satisfaction
Monthly Income
Monthly Rate
Number of Companies Worked
Percent Salary Hike
Performance Rating
Relationship Satisfaction
Stock Option Level
Total Working Years
Training Times Last Year
Work-Life Balance
Years At Company
Years In Current Role
Years Since Last Promotion
Years With Current Manager
Categorical Features

The model also uses categorical employee information:

Business Travel
Department
Education Field
Gender
Job Role
Marital Status
OverTime
Machine Learning Pipeline

The project follows the following Machine Learning workflow:

Raw Employee Data
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Categorical Encoding
        ↓
Feature Scaling
        ↓
Train / Validation / Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
Model Export
        ↓
Prediction Application
Prediction System

The application allows users to enter information for a single employee and receive a prediction.

The system analyzes the provided employee information and produces an estimated attrition prediction.

Example:

Employee Information
        ↓
Preprocessing
        ↓
Trained ML Model
        ↓
Prediction
        ↓
Attrition Risk
Dashboard

The application provides an interactive interface where users can enter employee information.

The dashboard includes employee attributes such as:

Age
Daily Rate
Distance From Home
Education
Environment Satisfaction
Hourly Rate
Job Involvement
Job Level
Job Satisfaction
Monthly Income
Monthly Rate
Number of Companies Worked
Percent Salary Hike
Performance Rating
Relationship Satisfaction
Stock Option Level
Total Working Years
Training Times Last Year
Work-Life Balance
Years At Company
Years In Current Role
Years Since Last Promotion
Years With Current Manager

and categorical information such as:

Business Travel
Department
Education Field
Gender
Job Role
Marital Status
OverTime
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Streamlit
Joblib
Installation

Clone the repository:

git clone https://github.com/YOUR-USERNAME/employee-attrition-prediction.git

Move into the project directory:

cd employee-attrition-prediction

Install the required dependencies:

pip install -r requirements.txt
Running the Application

Start the application using:

streamlit run app.py

The application will then open in your browser.

Model Evaluation

The trained model is evaluated using appropriate classification metrics, including:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
ROC-AUC

The final model is selected based on its performance on unseen data rather than training performance alone.
