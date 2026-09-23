import numpy as np
import pandas as pd
import joblib

# Load Artifacts

# PCA
pca = joblib.load("model_artifacts/pca.joblib")

# Scalers
scaler_age_at_enrollment = joblib.load("model_artifacts/scalers/scaler_transform_Age_at_enrollment.joblib")
scaler_curricular_units_1st_sem_approved = joblib.load("model_artifacts/scalers/scaler_transform_Curricular_units_1st_sem_approved.joblib")
scaler_curricular_units_1st_sem_enrolled = joblib.load("model_artifacts/scalers/scaler_transform_Curricular_units_1st_sem_enrolled.joblib")
scaler_curricular_units_1st_sem_evaluations = joblib.load("model_artifacts/scalers/scaler_transform_Curricular_units_1st_sem_evaluations.joblib")
scaler_curricular_units_1st_sem_grade = joblib.load("model_artifacts/scalers/scaler_transform_Curricular_units_1st_sem_grade.joblib")
scaler_curricular_units_2nd_sem_approved = joblib.load("model_artifacts/scalers/scaler_transform_Curricular_units_2nd_sem_approved.joblib")
scaler_curricular_units_2nd_sem_enrolled = joblib.load("model_artifacts/scalers/scaler_transform_Curricular_units_2nd_sem_enrolled.joblib")
scaler_curricular_units_2nd_sem_evaluations = joblib.load("model_artifacts/scalers/scaler_transform_Curricular_units_2nd_sem_evaluations.joblib")
scaler_curricular_units_2nd_sem_grade = joblib.load("model_artifacts/scalers/scaler_transform_Curricular_units_2nd_sem_grade.joblib")

# Encoders
encoder_application_mode = joblib.load("model_artifacts/encoders/one_hot_encoder_Application_mode.joblib")
encoder_debtor = joblib.load("model_artifacts/encoders/one_hot_encoder_Debtor.joblib")
encoder_scholarship_holder = joblib.load("model_artifacts/encoders/one_hot_encoder_Scholarship_holder.joblib")
encoder_tuition_fee_up_to_date = joblib.load("model_artifacts/encoders/one_hot_encoder_Tuition_fees_up_to_date.joblib")
encoder_course = joblib.load("model_artifacts/encoders/one_hot_encoder_Course.joblib")

# Transformers
transformer_Age_at_enrollment = joblib.load("model_artifacts/transformers/transformer_Age_at_enrollment.joblib")
transformer_curricular_units_1st_sem_approved = joblib.load("model_artifacts/transformers/transformer_Curricular_units_1st_sem_approved.joblib")
transformer_curricular_units_1st_sem_enrolled = joblib.load("model_artifacts/transformers/transformer_Curricular_units_1st_sem_enrolled.joblib")
transformer_curricular_units_1st_sem_evaluations = joblib.load("model_artifacts/transformers/transformer_Curricular_units_1st_sem_evaluations.joblib")
transformer_curricular_units_1st_sem_grade = joblib.load("model_artifacts/transformers/transformer_Curricular_units_1st_sem_grade.joblib")
transformer_curricular_units_2nd_sem_approved = joblib.load("model_artifacts/transformers/transformer_Curricular_units_2nd_sem_approved.joblib")
transformer_curricular_units_2nd_sem_enrolled = joblib.load("model_artifacts/transformers/transformer_Curricular_units_2nd_sem_enrolled.joblib")
transformer_curricular_units_2nd_sem_evaluations = joblib.load("model_artifacts/transformers/transformer_Curricular_units_2nd_sem_evaluations.joblib")
transformer_curricular_units_2nd_sem_grade = joblib.load("model_artifacts/transformers/transformer_Curricular_units_2nd_sem_grade.joblib")

pca_columns = [
    'transform_Curricular_units_1st_sem_approved', 'transform_Curricular_units_2nd_sem_approved', 'transform_Curricular_units_2nd_sem_grade', 'transform_Curricular_units_2nd_sem_enrolled',
    'transform_Curricular_units_2nd_sem_evaluations', 'transform_Curricular_units_1st_sem_grade', 'transform_Curricular_units_1st_sem_enrolled','transform_Curricular_units_1st_sem_evaluations'
]

def preprocessing(data):
    data = data.copy()
    df = pd.DataFrame()

    # Power Transform
    df['transform_Age_at_enrollment'] = transformer_Age_at_enrollment.transform(np.asarray(data['Age_at_enrollment']).reshape(-1,1))[0]
    df['transform_Curricular_units_1st_sem_approved'] = transformer_curricular_units_1st_sem_approved.transform(np.asarray(data['Curricular_units_1st_sem_approved']).reshape(-1,1))[0]
    df['transform_Curricular_units_1st_sem_enrolled'] = transformer_curricular_units_1st_sem_enrolled.transform(np.asarray(data['Curricular_units_1st_sem_enrolled']).reshape(-1,1))[0]
    df['transform_Curricular_units_1st_sem_evaluations'] = transformer_curricular_units_1st_sem_evaluations.transform(np.asarray(data['Curricular_units_1st_sem_evaluations']).reshape(-1,1))[0]
    df['transform_Curricular_units_1st_sem_grade'] = transformer_curricular_units_1st_sem_grade.transform(np.asarray(data['Curricular_units_1st_sem_grade']).reshape(-1,1))[0]
    df['transform_Curricular_units_2nd_sem_approved'] = transformer_curricular_units_2nd_sem_approved.transform(np.asarray(data['Curricular_units_2nd_sem_approved']).reshape(-1,1))[0]
    df['transform_Curricular_units_2nd_sem_enrolled'] = transformer_curricular_units_2nd_sem_enrolled.transform(np.asarray(data['Curricular_units_2nd_sem_enrolled']).reshape(-1,1))[0]
    df['transform_Curricular_units_2nd_sem_evaluations'] = transformer_curricular_units_2nd_sem_evaluations.transform(np.asarray(data['Curricular_units_2nd_sem_evaluations']).reshape(-1,1))[0]
    df['transform_Curricular_units_2nd_sem_grade'] = transformer_curricular_units_2nd_sem_grade.transform(np.asarray(data['Curricular_units_2nd_sem_grade']).reshape(-1,1))[0]

    # Scaling
    df['transform_Age_at_enrollment'] = scaler_age_at_enrollment.transform(np.asarray(df['transform_Age_at_enrollment']).reshape(-1,1))[0]
    df['transform_Curricular_units_1st_sem_approved'] = scaler_curricular_units_1st_sem_approved.transform(np.asarray(df['transform_Curricular_units_1st_sem_approved']).reshape(-1,1))[0]
    df['transform_Curricular_units_1st_sem_enrolled'] = scaler_curricular_units_1st_sem_enrolled.transform(np.asarray(df['transform_Curricular_units_1st_sem_enrolled']).reshape(-1,1))[0]
    df['transform_Curricular_units_1st_sem_evaluations'] = scaler_curricular_units_1st_sem_evaluations.transform(np.asarray(df['transform_Curricular_units_1st_sem_evaluations']).reshape(-1,1))[0]
    df['transform_Curricular_units_1st_sem_grade'] = scaler_curricular_units_1st_sem_grade.transform(np.asarray(df['transform_Curricular_units_1st_sem_grade']).reshape(-1,1))[0]
    df['transform_Curricular_units_2nd_sem_approved'] = scaler_curricular_units_2nd_sem_approved.transform(np.asarray(df['transform_Curricular_units_2nd_sem_approved']).reshape(-1,1))[0]
    df['transform_Curricular_units_2nd_sem_enrolled'] = scaler_curricular_units_2nd_sem_enrolled.transform(np.asarray(df['transform_Curricular_units_2nd_sem_enrolled']).reshape(-1,1))[0]
    df['transform_Curricular_units_2nd_sem_evaluations'] = scaler_curricular_units_2nd_sem_evaluations.transform(np.asarray(df['transform_Curricular_units_2nd_sem_evaluations']).reshape(-1,1))[0]
    df['transform_Curricular_units_2nd_sem_grade'] = scaler_curricular_units_2nd_sem_grade.transform(np.asarray(df['transform_Curricular_units_2nd_sem_grade']).reshape(-1,1))[0]

    # Encoding
    tuition_fee_encoded_cols = encoder_tuition_fee_up_to_date.get_feature_names_out(['Tuition_fees_up_to_date'])
    df[tuition_fee_encoded_cols] = encoder_tuition_fee_up_to_date.transform(data[['Tuition_fees_up_to_date']])
    Application_mode_encoded_cols = encoder_application_mode.get_feature_names_out(['Application_mode'])
    df[Application_mode_encoded_cols] = encoder_application_mode.transform(data[['Application_mode']])
    course_encoded_cols = encoder_course.get_feature_names_out(['Course'])
    df[course_encoded_cols] = encoder_course.transform(data[['Course']])
    debtor_encoded_cols = encoder_debtor.get_feature_names_out(['Debtor'])
    df[debtor_encoded_cols] = encoder_debtor.transform(data[['Debtor']])
    scholarship_encoded_cols = encoder_scholarship_holder.get_feature_names_out(['Scholarship_holder'])
    df[scholarship_encoded_cols] = encoder_scholarship_holder.transform(data[['Scholarship_holder']])

    # PCA
    df[['pc_1', 'pc_2', 'pc_3', 'pc_4']] = pca.transform(df[pca_columns])

    df.drop(columns=pca_columns, inplace=True)

    return df