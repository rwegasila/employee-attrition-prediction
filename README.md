# HR Attrition Insight — App ya Kutabiri Uondokaji wa Wafanyakazi

App hii inatumia modeli yako halisi ya **Random Forest (SMOTE)** iliyotrainiwa
kwenye dataset ya IBM HR Employee Attrition, kuwasaidia HR:

- Kuingia kwenye mfumo (login)
- Kuona dashibodi (dashboard) na grafu za hatari ya uondokaji kwa idara/nafasi ya kazi
- Kutathmini mfanyakazi mmoja mmoja na kupata uwezekano (%) wa kuondoka
- Kupata **mapendekezo ya moja kwa moja kwa HR** kulingana na sababu za hatari
- Kupakia CSV ya wafanyakazi wengi na kupata utabiri wa wote kwa pamoja

## Muundo wa Faili

```
hr_app/
├── app.py              # App kuu ya Streamlit
├── preprocessing.py     # Ubadilishaji wa data (encoding) unaolingana na notebook yako
├── requirements.txt
├── model/
│   └── random_forest_smote.pkl   # Modeli yako uliyoipakia
└── README.md
```

## Jinsi ya Kuiendesha (How to Run)

1. Hakikisha una Python 3.9+ imewekwa kwenye kompyuta yako.
2. Fungua terminal kwenye folda ya `hr_app` kisha weka packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Endesha app:

   ```bash
   streamlit run app.py
   ```

4. Itafungua kwenye browser yako (kawaida `http://localhost:8501`).

### Akaunti za Kuingia (Demo Login)

| Username | Password  | Role         |
|----------|-----------|--------------|
| admin    | admin123  | HR Manager   |
| hr       | hr2024    | HR Officer   |

> ⚠️ Hizi ni akaunti za majaribio tu (hardcoded ndani ya `app.py`, sehemu ya
> `USERS`). Kwa matumizi halisi (production), unapaswa kuunganisha na
> database halisi (mf. PostgreSQL) na kutumia password hashing (mf. bcrypt),
> si maandishi wazi (plaintext) kama ilivyo hapa kwa mfano.

## CSV ya Batch Prediction (Dashibodi)

Kwa ukurasa wa "Dashibodi", pakia CSV yenye columns zifuatazo (kama za
dataset ya awali `WA_Fn-UseC_-HR-Employee-Attrition.csv`):

```
Age, BusinessTravel, DailyRate, Department, DistanceFromHome, Education,
EducationField, EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement,
JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate,
NumCompaniesWorked, OverTime, PercentSalaryHike, PerformanceRating,
RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears,
TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole,
YearsSinceLastPromotion, YearsWithCurrManager
```

## Kumbuka Muhimu Kuhusu Usahihi wa Encoding

Notebook yako ilihifadhi (`joblib.dump`) **modeli ya Random Forest tu**, si
`ColumnTransformer`/`OneHotEncoder` iliyotumika kubadilisha data. Faili
`preprocessing.py` imejengwa upya kutumia categories za kawaida (standard)
za dataset hii maarufu ya Kaggle, na imethibitishwa (verified) kutoa
features 44 sahihi zinazolingana na `model.n_features_in_ == 44`.

Ikiwa dataset yako halisi ina categories tofauti kidogo (mf. maadili mapya
ya JobRole), ni bora **kuhifadhi na kupakia `ColumnTransformer` yenyewe**
wakati ujao (`joblib.dump(preprocessor, "preprocessor.pkl")`) ili kuepuka
utegemezi wa kukisia.

## Kuboresha Baadaye

- Unganisha na database halisi ya wafanyakazi badala ya CSV upload
- Ongeza historia ya utabiri (prediction history) kwa kila mfanyakazi
- Weka authentication imara zaidi (JWT / OAuth)
- Deploy kwenye Streamlit Community Cloud, Render, au server yako
