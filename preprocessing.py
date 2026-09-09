"""
Preprocessing logic that mirrors EXACTLY what was done in BUsiness_Intelligence.ipynb:

    categorical_columns -> OneHotEncoder(handle_unknown="ignore", drop="first")
    everything else     -> passthrough (numerical, unchanged, original order)

Because the notebook only saved the trained RandomForestClassifier
(joblib.dump(rf_smote, "random_forest_smote.pkl")) and NOT the fitted
ColumnTransformer/OneHotEncoder, we rebuild the encoding here using the
category values that exist in the standard IBM HR Employee Attrition
dataset (WA_Fn-UseC_-HR-Employee-Attrition.csv), in the same column order
the notebook produced.

sklearn's OneHotEncoder sorts each column's categories alphabetically and,
with drop="first", drops the alphabetically-first category. That behaviour
is reproduced by CATEGORY_LEVELS below (first entry in each list = the
dropped baseline).
"""

import numpy as np
import pandas as pd

# Raw feature column order AFTER dropping Attrition, EmployeeNumber,
# EmployeeCount, StandardHours, Over18 (exactly as in the notebook, cell 12/15)
RAW_COLUMNS = [
    "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
    "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]

CATEGORICAL_COLUMNS = [
    "BusinessTravel", "Department", "EducationField", "Gender",
    "JobRole", "MaritalStatus", "OverTime",
]

NUMERICAL_COLUMNS = [c for c in RAW_COLUMNS if c not in CATEGORICAL_COLUMNS]

# alphabetically sorted categories; index 0 = baseline dropped by drop="first"
CATEGORY_LEVELS = {
    "BusinessTravel": ["Non-Travel", "Travel_Frequently", "Travel_Rarely"],
    "Department": ["Human Resources", "Research & Development", "Sales"],
    "EducationField": ["Human Resources", "Life Sciences", "Marketing",
                        "Medical", "Other", "Technical Degree"],
    "Gender": ["Female", "Male"],
    "JobRole": ["Healthcare Representative", "Human Resources",
                "Laboratory Technician", "Manager", "Manufacturing Director",
                "Research Director", "Research Scientist", "Sales Executive",
                "Sales Representative"],
    "MaritalStatus": ["Divorced", "Married", "Single"],
    "OverTime": ["No", "Yes"],
}

# final encoded column order = [onehot(cat1), onehot(cat2), ... , numerical...]
def _encoded_column_names():
    names = []
    for col in CATEGORICAL_COLUMNS:
        levels = CATEGORY_LEVELS[col][1:]  # skip dropped baseline
        for lvl in levels:
            names.append(f"{col}_{lvl}")
    names.extend(NUMERICAL_COLUMNS)
    return names


ENCODED_COLUMNS = _encoded_column_names()  # should have length 44


def encode_employee(record: dict) -> np.ndarray:
    """
    record: dict with keys = RAW_COLUMNS, raw human-entered values.
    returns: 1 x 44 numpy array in the exact order the model expects.
    """
    row = []
    for col in CATEGORICAL_COLUMNS:
        levels = CATEGORY_LEVELS[col]
        baseline = levels[0]
        value = record.get(col, baseline)
        for lvl in levels[1:]:
            row.append(1.0 if value == lvl else 0.0)
    for col in NUMERICAL_COLUMNS:
        row.append(float(record.get(col, 0)))
    arr = np.array(row, dtype=float).reshape(1, -1)
    assert arr.shape[1] == 44, f"Expected 44 features, got {arr.shape[1]}"
    return arr


def encode_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Batch version: df has raw columns (subset/all of RAW_COLUMNS)."""
    records = df.to_dict(orient="records")
    rows = [encode_employee(r) for r in records]
    return np.vstack(rows)
