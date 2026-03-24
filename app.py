import streamlit as st
import joblib
import xgboost
import pandas as pd

st.title("Student Employability Predictor")
st.header("eXtreme Gradient Boosting Classifier model")

gen_appear=st.selectbox("GENERAL APPEARANCE",[2,3,4,5])
speaking= st.selectbox("MANNER OF SPEAKING",[2,3,4,5])
physical_cond= st.selectbox("PHYSICAL CONDITION",[2,3,4,5])
mental_alterness= st.selectbox("MENTAL ALERTNESS",[2,3,4,5])
confidence= st.selectbox("SELF CONFIDENCE",[2,3,4,5])
ideas= st.selectbox("ABILITY TO PRESENT IDEAS",[2,3,4,5])
com_skills= st.selectbox("COMMUNICATION SKILLS",[2,3,4,5])
performance= st.selectbox("STUDENT PERFORMANCE RATING",[3,4,5])


if st.button("Predict"):
    with open("student_employability_model.pkl", "rb") as f:
        model = joblib.load(f)
        columns = ['GENERAL APPEARANCE', 'MANNER OF SPEAKING', 'PHYSICAL CONDITION',
       'MENTAL ALERTNESS', 'SELF CONFIDENCE', 'ABILITY TO PRESENT IDEAS',
       'COMMUNICATION SKILLS', 'STUDENT PERFORMANCE RATING']
        sample_input = pd.DataFrame([[gen_appear, speaking, physical_cond,
       mental_alterness, confidence, ideas,
       com_skills, performance]], 
                            columns=columns)
        prediction = model.predict(sample_input)
        if prediction[0]==1:
            st.write("Employable")
        else:
            st.write("Less Employable")