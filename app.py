import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from preprocessing import (
    encode_employee, encode_dataframe, RAW_COLUMNS,
    CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, CATEGORY_LEVELS, ENCODED_COLUMNS,
)

st.set_page_config(page_title="HR Attrition Insight", page_icon="📊", layout="wide")

# ----------------------------------------------------------------------
# ------------------------------- LOGIN ---------------------------------
# ----------------------------------------------------------------------
# Demo user store. Kwa uzalishaji (production) tumia database + password
# hashing (mf. bcrypt) na si maandishi wazi (plaintext) kama hapa.
USERS = {
    "admin": {"password": "admin123", "role": "HR Manager", "name": "Admin"},
    "hr": {"password": "hr2024", "role": "HR Officer", "name": "HR Officer"},
}


def login_page():
    st.markdown(
        """
        <div style='text-align:center; padding-top:40px;'>
            <h1>📊 HR Attrition Insight</h1>
            <p style='color:gray;'>Ingia (Login) kuendelea</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Jina la mtumiaji (Username)")
            password = st.text_input("Nenosiri (Password)", type="password")
            submitted = st.form_submit_button("Ingia (Login)", use_container_width=True)
            if submitted:
                user = USERS.get(username)
                if user and user["password"] == password:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.session_state["role"] = user["role"]
                    st.session_state["name"] = user["name"]
                    st.rerun()
                else:
                    st.error("Jina la mtumiaji au nenosiri sio sahihi.")
        with st.expander("Akaunti za majaribio (demo accounts)"):
            st.code("admin / admin123\nhr / hr2024")


def logout_button():
    with st.sidebar:
        st.markdown(f"👤 **{st.session_state['name']}**  \n_{st.session_state['role']}_")
        if st.button("Toka (Logout)", use_container_width=True):
            for k in ["logged_in", "username", "role", "name"]:
                st.session_state.pop(k, None)
            st.rerun()
        st.divider()


# ----------------------------------------------------------------------
# ----------------------------- MODEL LOAD -------------------------------
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model/random_forest_smote.pkl")


model = load_model()


# ----------------------------------------------------------------------
# --------------------------- RECOMMENDATION -----------------------------
# ----------------------------------------------------------------------
def generate_recommendations(record: dict, risk_proba: float) -> list:
    """Rule-based HR recommendations derived from known attrition drivers
    in this dataset / HR literature, layered on top of the model's risk
    score. Returns a list of (priority, text) tuples."""
    recs = []

    if record.get("OverTime") == "Yes":
        recs.append(("Juu", "Mfanyakazi anafanya kazi za ziada (OverTime). Pima mzigo wa kazi (workload) na fikiria kupunguza saa za ziada au kutoa fidia/likizo ya ziada."))

    if record.get("JobSatisfaction", 3) <= 2:
        recs.append(("Juu", "Kiwango cha kuridhika na kazi (Job Satisfaction) ni cha chini. Panga mazungumzo ya 1-kwa-1 (one-on-one) kuelewa changamoto na fursa za ukuaji."))

    if record.get("EnvironmentSatisfaction", 3) <= 2:
        recs.append(("Wastani", "Kuridhika na mazingira ya kazi ni chini. Angalia mahusiano na wenzake/msimamizi na hali ya ofisi."))

    if record.get("WorkLifeBalance", 3) <= 2:
        recs.append(("Juu", "Uwiano wa kazi na maisha (Work-Life Balance) ni chini. Fikiria muundo wa kazi unaobadilika (flexible schedule) au kufanya kazi mbali (remote)."))

    monthly_income = record.get("MonthlyIncome", 0)
    if monthly_income and monthly_income < 3000:
        recs.append(("Juu", "Mshahara wa mwezi ni mdogo ukilinganisha na soko. Fanya tathmini ya malipo (salary benchmarking) kwa nafasi yake."))

    years_since_promo = record.get("YearsSinceLastPromotion", 0)
    if years_since_promo and years_since_promo >= 4:
        recs.append(("Wastani", "Hajapandishwa cheo kwa zaidi ya miaka 4. Jadili njia ya ukuaji wa kazi (career path) na fursa za kupandishwa cheo."))

    if record.get("DistanceFromHome", 0) and record.get("DistanceFromHome") >= 20:
        recs.append(("Wastani", "Umbali kutoka nyumbani kwenda kazini ni mkubwa. Fikiria uwezekano wa kufanya kazi mchanganyiko (hybrid) au usafiri wa kampuni."))

    if record.get("TrainingTimesLastYear", 0) is not None and record.get("TrainingTimesLastYear", 0) == 0:
        recs.append(("Ndogo", "Hajapata mafunzo mwaka jana. Mjumuishe kwenye mpango wa mafunzo (training plan) ujao."))

    if record.get("StockOptionLevel", 0) == 0 and risk_proba > 0.5:
        recs.append(("Ndogo", "Hana hisa/motisha ya muda mrefu (stock option). Fikiria vivutio vya ziada vya kubaki (retention incentives)."))

    if not recs:
        recs.append(("Chini", "Hakuna hatari kubwa iliyobainika. Endelea kufuatilia kuridhika kwake mara kwa mara (regular check-ins)."))

    # Sort: Juu > Wastani > Ndogo > Chini
    order = {"Juu": 0, "Wastani": 1, "Ndogo": 2, "Chini": 3}
    recs.sort(key=lambda r: order.get(r[0], 4))
    return recs


def risk_band(p):
    if p >= 0.66:
        return "Hatari Kubwa (High)", "🔴"
    elif p >= 0.33:
        return "Hatari ya Wastani (Medium)", "🟠"
    else:
        return "Hatari Ndogo (Low)", "🟢"


# ----------------------------------------------------------------------
# ------------------------------- PAGES ----------------------------------
# ----------------------------------------------------------------------
def page_dashboard():
    st.header("📊 Dashibodi (Dashboard)")
    st.caption("Pakia (upload) CSV yenye wafanyakazi wengi kuona muhtasari na grafu za hatari ya uondokaji (attrition).")

    uploaded = st.file_uploader("Pakia CSV ya wafanyakazi", type=["csv"], key="dash_csv")

    if uploaded is None:
        st.info("Bado hujapakia data. Unaweza kutumia CSV yenye columns kama za dataset ya awali (Age, Department, JobRole, n.k.).")
        return

    df = pd.read_csv(uploaded)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"CSV inakosa columns hizi: {missing}")
        return

    try:
        X = encode_dataframe(df[RAW_COLUMNS])
        proba = model.predict_proba(X)[:, 1]
        pred = model.predict(X)
    except Exception as e:
        st.error(f"Hitilafu wakati wa kutathmini data: {e}")
        return

    df_res = df.copy()
    df_res["Attrition_Risk_%"] = (proba * 100).round(1)
    df_res["Prediction"] = np.where(pred == 1, "Atakayeondoka (Yes)", "Atabaki (No)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumla ya Wafanyakazi", len(df_res))
    c2.metric("Wenye Hatari Kubwa (≥66%)", int((df_res["Attrition_Risk_%"] >= 66).sum()))
    c3.metric("Wastani wa Hatari", f"{df_res['Attrition_Risk_%'].mean():.1f}%")
    c4.metric("Idadi Itakayoondoka (predicted)", int((pred == 1).sum()))

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df_res, x="Attrition_Risk_%", nbins=20,
                            title="Mgawanyo wa Hatari ya Uondokaji (Attrition Risk Distribution)",
                            color_discrete_sequence=["#636EFA"])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "Department" in df_res.columns:
            dept_risk = df_res.groupby("Department")["Attrition_Risk_%"].mean().reset_index()
            fig2 = px.bar(dept_risk, x="Department", y="Attrition_Risk_%",
                          title="Wastani wa Hatari kwa Idara (Avg Risk by Department)",
                          color="Attrition_Risk_%", color_continuous_scale="Reds")
            st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if "JobRole" in df_res.columns:
            role_risk = df_res.groupby("JobRole")["Attrition_Risk_%"].mean().sort_values(ascending=False).reset_index()
            fig3 = px.bar(role_risk, x="Attrition_Risk_%", y="JobRole", orientation="h",
                          title="Wastani wa Hatari kwa Nafasi ya Kazi (by Job Role)",
                          color="Attrition_Risk_%", color_continuous_scale="Oranges")
            st.plotly_chart(fig3, use_container_width=True)
    with col4:
        importances = pd.DataFrame({
            "Feature": ENCODED_COLUMNS,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False).head(12)
        fig4 = px.bar(importances, x="Importance", y="Feature", orientation="h",
                      title="Vigezo Muhimu Zaidi kwa Modeli (Top Feature Importances)",
                      color="Importance", color_continuous_scale="Blues")
        fig4.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.subheader("Wafanyakazi wenye Hatari Kubwa Zaidi (Top At-Risk Employees)")
    top_risk = df_res.sort_values("Attrition_Risk_%", ascending=False).head(15)
    st.dataframe(top_risk, use_container_width=True)

    st.download_button(
        "⬇️ Pakua matokeo yote (Download full results CSV)",
        df_res.to_csv(index=False).encode("utf-8"),
        file_name="attrition_predictions.csv",
        mime="text/csv",
    )


def page_predict_single():
    st.header("🔍 Tathmini Mfanyakazi Mmoja (Single Employee Prediction)")

    with st.form("single_predict"):
        c1, c2, c3 = st.columns(3)
        with c1:
            Age = st.number_input("Umri (Age)", 18, 65, 30)
            DailyRate = st.number_input("Daily Rate", 100, 1500, 800)
            DistanceFromHome = st.number_input("Umbali kutoka nyumbani (km)", 0, 50, 5)
            Education = st.selectbox("Kiwango cha Elimu (1=Chini ... 5=Juu)", [1, 2, 3, 4, 5], index=2)
            EnvironmentSatisfaction = st.selectbox("Kuridhika na Mazingira (1-4)", [1, 2, 3, 4], index=2)
            HourlyRate = st.number_input("Hourly Rate", 20, 150, 60)
            JobInvolvement = st.selectbox("Kujihusisha na Kazi (1-4)", [1, 2, 3, 4], index=2)
            JobLevel = st.selectbox("Ngazi ya Kazi (1-5)", [1, 2, 3, 4, 5], index=1)
        with c2:
            JobSatisfaction = st.selectbox("Kuridhika na Kazi (1-4)", [1, 2, 3, 4], index=2)
            MonthlyIncome = st.number_input("Mshahara wa Mwezi", 1000, 50000, 5000, step=100)
            MonthlyRate = st.number_input("Monthly Rate", 1000, 30000, 15000, step=100)
            NumCompaniesWorked = st.number_input("Idadi ya Makampuni Aliyofanyia kazi", 0, 15, 2)
            PercentSalaryHike = st.number_input("Ongezeko la Mshahara (%)", 0, 50, 13)
            PerformanceRating = st.selectbox("Kiwango cha Utendaji (1-4)", [1, 2, 3, 4], index=2)
            RelationshipSatisfaction = st.selectbox("Kuridhika na Mahusiano Kazini (1-4)", [1, 2, 3, 4], index=2)
            StockOptionLevel = st.selectbox("Kiwango cha Hisa (Stock Option) (0-3)", [0, 1, 2, 3], index=0)
        with c3:
            TotalWorkingYears = st.number_input("Jumla ya Miaka ya Kufanya Kazi", 0, 45, 8)
            TrainingTimesLastYear = st.number_input("Mafunzo Mwaka Jana (idadi)", 0, 10, 2)
            WorkLifeBalance = st.selectbox("Uwiano wa Kazi na Maisha (1-4)", [1, 2, 3, 4], index=2)
            YearsAtCompany = st.number_input("Miaka Kampuni ni", 0, 40, 5)
            YearsInCurrentRole = st.number_input("Miaka Katika Nafasi ya Sasa", 0, 20, 3)
            YearsSinceLastPromotion = st.number_input("Miaka Tangu Kupandishwa Cheo", 0, 20, 1)
            YearsWithCurrManager = st.number_input("Miaka na Meneja wa Sasa", 0, 20, 3)

        st.markdown("**Taarifa za Kikategoria (Categorical)**")
        c4, c5, c6, c7 = st.columns(4)
        with c4:
            BusinessTravel = st.selectbox("Safari za Kikazi", CATEGORY_LEVELS["BusinessTravel"], index=2)
            Department = st.selectbox("Idara", CATEGORY_LEVELS["Department"], index=1)
        with c5:
            EducationField = st.selectbox("Fani ya Elimu", CATEGORY_LEVELS["EducationField"], index=1)
            Gender = st.selectbox("Jinsia", CATEGORY_LEVELS["Gender"], index=1)
        with c6:
            JobRole = st.selectbox("Nafasi ya Kazi", CATEGORY_LEVELS["JobRole"], index=5)
            MaritalStatus = st.selectbox("Hali ya Ndoa", CATEGORY_LEVELS["MaritalStatus"], index=1)
        with c7:
            OverTime = st.selectbox("Kazi za Ziada (OverTime)", CATEGORY_LEVELS["OverTime"], index=0)

        submitted = st.form_submit_button("Tathmini (Predict)", use_container_width=True)

    if submitted:
        record = dict(
            Age=Age, BusinessTravel=BusinessTravel, DailyRate=DailyRate,
            Department=Department, DistanceFromHome=DistanceFromHome,
            Education=Education, EducationField=EducationField,
            EnvironmentSatisfaction=EnvironmentSatisfaction, Gender=Gender,
            HourlyRate=HourlyRate, JobInvolvement=JobInvolvement,
            JobLevel=JobLevel, JobRole=JobRole, JobSatisfaction=JobSatisfaction,
            MaritalStatus=MaritalStatus, MonthlyIncome=MonthlyIncome,
            MonthlyRate=MonthlyRate, NumCompaniesWorked=NumCompaniesWorked,
            OverTime=OverTime, PercentSalaryHike=PercentSalaryHike,
            PerformanceRating=PerformanceRating,
            RelationshipSatisfaction=RelationshipSatisfaction,
            StockOptionLevel=StockOptionLevel, TotalWorkingYears=TotalWorkingYears,
            TrainingTimesLastYear=TrainingTimesLastYear,
            WorkLifeBalance=WorkLifeBalance, YearsAtCompany=YearsAtCompany,
            YearsInCurrentRole=YearsInCurrentRole,
            YearsSinceLastPromotion=YearsSinceLastPromotion,
            YearsWithCurrManager=YearsWithCurrManager,
        )

        X = encode_employee(record)
        proba = model.predict_proba(X)[0, 1]
        band_label, band_icon = risk_band(proba)

        st.divider()
        r1, r2 = st.columns([1, 2])
        with r1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={"text": "Uwezekano wa Kuondoka (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "black"},
                    "steps": [
                        {"range": [0, 33], "color": "#c7f0c2"},
                        {"range": [33, 66], "color": "#ffe0a3"},
                        {"range": [66, 100], "color": "#ffb3b3"},
                    ],
                },
            ))
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"### {band_icon} {band_label}")

        with r2:
            st.subheader("💡 Mapendekezo kwa HR (Recommendations)")
            recs = generate_recommendations(record, proba)
            for priority, text in recs:
                icon = {"Juu": "🔴", "Wastani": "🟠", "Ndogo": "🟡", "Chini": "🟢"}.get(priority, "⚪")
                st.markdown(f"{icon} **[{priority}]** {text}")


def page_about():
    st.header("ℹ️ Kuhusu Modeli (About the Model)")
    st.markdown(
        """
        - **Aina ya Modeli:** Random Forest Classifier (n_estimators=300)
        - **Data ya Mafunzo:** IBM HR Employee Attrition Dataset
        - **Ulinganifu wa Data (Class Balance):** SMOTE ilitumika kusawazisha
          makundi (Yes/No) kabla ya mafunzo
        - **Vigezo (Features):** 30 raw → 44 baada ya One-Hot Encoding
        - Muundo huu ni wa **msaada wa maamuzi (decision support)** — sio
          uamuzi wa mwisho. Daima changanya na hukumu ya kibinadamu na
          mazungumzo halisi na mfanyakazi.
        """
    )
    st.subheader("Vigezo Muhimu Zaidi (Top 15 Feature Importances)")
    importances = pd.DataFrame({
        "Feature": ENCODED_COLUMNS,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(15)
    fig = px.bar(importances, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Viridis")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------
# -------------------------------- MAIN -----------------------------------
# ----------------------------------------------------------------------
def main():
    if not st.session_state.get("logged_in"):
        login_page()
        return

    logout_button()
    with st.sidebar:
        st.markdown("### 📂 Menyu")
        page = st.radio(
            "Chagua ukurasa",
            ["Dashibodi", "Tathmini Mfanyakazi Mmoja", "Kuhusu Modeli"],
            label_visibility="collapsed",
        )

    if page == "Dashibodi":
        page_dashboard()
    elif page == "Tathmini Mfanyakazi Mmoja":
        page_predict_single()
    else:
        page_about()


if __name__ == "__main__":
    main()
