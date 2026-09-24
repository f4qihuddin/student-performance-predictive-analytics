import streamlit as st
import pandas as pd
import joblib
import base64
from preprocessing import preprocessing 
from prediction import prediction

data = pd.DataFrame()

with open("assets/certificate_8136898.png", "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode()

st.markdown("""
<style>
div[data-testid="stButton"] {
    display: flex;
    align-items: center !important;
    width: 100%;
}

div[data-testid="stButton"] button {
    background-color: #03045E;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    width: 300px;
    margin: 0 auto;
}

div[data-testid="stButton"] button:hover {
    background-color: #0077B6;
    color: white;
}

.top-bar {
    padding: 12px;
    display: flex;
    flex-direction: row;
    justify-content: center;
    gap: 20px;
    align-items: center;
    width: 100%;
    box-sizing: border-box;
}

.top-bar h3 {
    color: var(--text-color);
}
</style>

<div class="top-bar">
    <img src="data:image/png;base64,{image_base64}" alt="Certificate" width=50px height=50px>
    <h2>Student Performance Prediction</h2>
</div>
<br>
""".replace("{image_base64}", image_base64), unsafe_allow_html=True)


with st.container(border=True):
    tuition_fees_up_to_date = st.selectbox(
        label="Tuition Fees up to Date?",
        options=["yes", "no"],
        index=1
    )
    data['Tuition_fees_up_to_date'] = [tuition_fees_up_to_date]

    scholarship_holder = st.selectbox(
        label="Scholarship Holder?",
        options=["yes", "no"],
        index=1
    )
    data['Scholarship_holder'] = [scholarship_holder]

    debtor = st.selectbox(
        label="Debtor?",
        options=["yes", "no"],
        index=1
    )
    data['Debtor'] = [debtor]

    course = st.selectbox(
        label="Course",
        options=[
            "Biofuel Production Technologies",
            "Animation and Multimedia Design",
            "Social Service (evening attendance)",
            "Agronomy",
            "Communication Design",
            "Veterinary Nursing",
            "Informatics Engineering",
            "Equinculture",
            "Management",
            "Social Service",
            "Tourism",
            "Nursing",
            "Oral Hygiene",
            "Advertising and Marketing Management",
            "Journalism and Communication",
            "Basic Education",
            "Management (evening attendance)"
        ],
        index=1
    )
    data['Course'] = [course]

    application_mode = st.selectbox(
        label="Application Mode",
        options=[
            "1st phase - general contingent",
            "Ordinance No. 612/93",
            "1st phase - special contingent (Azores Island)",
            "Holders of other higher courses",
            "Ordinance No. 854-B/99",
            "International student (bachelor)",
            "1st phase - special contingent (Madeira Island)",
            "2nd phase - general contingent",
            "3rd phase - general contingent",
            "Ordinance No. 533-A/99, item b2) (Different Plan)"
            "Ordinance No. 533-A/99, item b3 (Other Institution)",
            "Over 23 years old",
            "Transfer",
            "Change of course",
            "Technological specialization diploma holders",
            "Change of institution/cou  rse",
            "Short cycle diploma holders",
            "Change of institution/course (International)",
        ],
        index=1
    )
    data['Application_mode'] = [application_mode]

with st.container(border=True):
    age = int(st.number_input(
        "Age at enrollment",
        min_value=17,
        max_value=70,
        value=20,
        step=1
    ))

    data['Age_at_enrollment'] = age

    st.text("Currricular Units")

    col1, col2 = st.columns(2)

    with col1:
        curricular_units_1st_sem_approved = int(st.number_input(
            "1st Semester Approved",
            value=0,
            step=1
        ))

        data['Curricular_units_1st_sem_approved'] = curricular_units_1st_sem_approved

        curricular_units_1st_sem_grade = float(st.number_input(
            "1st Semester Grade",
            value=0.0,
            step=0.1
        ))

        data['Curricular_units_1st_sem_grade'] = curricular_units_1st_sem_grade

        curricular_units_1st_sem_enrolled = int(st.number_input(
            "1st Semester Enrolled",
            value=0,
            step=1
        ))

        data['Curricular_units_1st_sem_enrolled'] = curricular_units_1st_sem_enrolled

        curricular_units_1st_sem_evaluations = int(st.number_input(
            "1st Semester Evaluations",
            value=0,
            step=1
        ))

        data['Curricular_units_1st_sem_evaluations'] = curricular_units_1st_sem_evaluations

with col2:
        curricular_units_2nd_sem_approved = int(st.number_input(
            "2nd Semester Approved",
            value=0,
            step=1
        ))

        data['Curricular_units_2nd_sem_approved'] = curricular_units_2nd_sem_approved

        curricular_units_2nd_sem_grade = float(st.number_input(
            "2nd Semester Grade",
            value=0.0,
            step=0.1
        ))

        data['Curricular_units_2nd_sem_grade'] = curricular_units_2nd_sem_grade

        curricular_units_2nd_sem_enrolled = int(st.number_input(
            "2nd Semester Enrolled",
            value=0,
            step=1
        ))

        data['Curricular_units_2nd_sem_enrolled'] = curricular_units_2nd_sem_enrolled

        curricular_units_2nd_sem_evaluations = int(st.number_input(
            "2nd Semester Evaluations",
            value=0,
            step=1
        ))

        data['Curricular_units_2nd_sem_evaluations'] = curricular_units_2nd_sem_evaluations

_, button_column, _ = st.columns([1, 1, 1])

with button_column:
    predict_clicked = st.button('Predict', use_container_width=True)

if predict_clicked:
    with st.container(border=True):
        new_data = preprocessing(data=data)
        status = prediction(new_data)
        
        if status == "Graduate":
            st.write("**Student Status: :green[Graduate]**")
        elif status == "Dropout":
            st.write("**Student Status: :red[Dropout]**")

        with st.expander("View the Preprocessed Data"):
            st.dataframe(data=new_data, width=800, height=65)