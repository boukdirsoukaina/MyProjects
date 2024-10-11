
import pandas as pd 
import streamlit as st
import prediction



st.sidebar.header("Information about the patient")
st.write('''
# Heart Attack Prediction ''')


def input_user():
    sex=  ['F','M']
    generalHealth =   ['Poor','Good','Excellent','Very good','Fair']
    PhysicalHealthDays=  [30. , 5. , 0., 20.,  2., 14.,  7. , 8. ,25. ,15. , 1. ,10. , 6. , 3., 12.,  4., 16., 21.,
     9., 13. ,17. ,29. ,28. ,24. ,27. ,26. ,23. ,19. ,18. ,11. ,22.]
    physicalActivities=  ['No' ,'Yes']
    removedTeeth=  ['All', '6 or more, but not all' ,'None of them' ,'1 to 5']
    hadAngina=  ['No' ,'Yes']
    hadStroke=  ['No' ,'Yes']
    hadCOPD=  ['Yes' ,'No']
    hadKidneyDisease=  ['No' ,'Yes']
    hadArthritis=  ['No' ,'Yes']
    hadDiabetes=  ['No', 'Yes' ,'No, pre-diabetes or borderline diabetes',
     'Yes, but only during pregnancy (female)']
    deafOrHardOfHearing=  ['No' ,'Yes']
    blindOrVisionDifficulty=  ['No' ,'Yes']
    difficultyWalking=  ['No' ,'Yes']
    difficultyDressingBathing=  ['Yes', 'No']
    difficultyErrands=  ['Yes' ,'No']
    smokerStatus=  ['Former smoker' ,'Never smoked', 'Current smoker - now smokes some days',
     'Current smoker - now smokes every day']
    chestScan=  ['Yes' ,'No']
    ageCategory=  ['Age 80 or older' ,'Age 40 to 44', 'Age 50 to 54', 'Age 65 to 69',
     'Age 75 to 79' ,'Age 55 to 59' ,'Age 18 to 24', 'Age 70 to 74',
      'Age 60 to 64', 'Age 45 to 49' ,'Age 25 to 29', 'Age 35 to 39',
       'Age 30 to 34']
    alcoholDrinkers=  ['No', 'Yes']
    pneumoVaxEver=  ['Yes' ,'No']


    select_sex = st.sidebar.selectbox("sex",sex)
    select_generalHealth = st.sidebar.selectbox("generalHealth",generalHealth) 
    select_PhysicalHealthDays = st.sidebar.selectbox("PhysicalHealthDays",PhysicalHealthDays)
    select_physicalActivities = st.sidebar.selectbox("physicalActivities",physicalActivities)
    select_removedTeeth = st.sidebar.selectbox("removedTeeth",removedTeeth)
    select_hadAngina = st.sidebar.selectbox("hadAngina",hadAngina)
    select_hadStroke = st.sidebar.selectbox("hadStroke",hadStroke)
    select_hadCOPD = st.sidebar.selectbox("hadCOPD",hadCOPD)
    select_hadKidneyDisease = st.sidebar.selectbox("hadKidneyDisease",hadKidneyDisease)
    select_hadArthritis = st.sidebar.selectbox("hadArthritis",hadArthritis)
    select_hadDiabetes = st.sidebar.selectbox("hadDiabetes",hadDiabetes)
    select_deafOrHardOfHearing = st.sidebar.selectbox("deafOrHardOfHearing",deafOrHardOfHearing)
    select_blindOrVisionDifficulty = st.sidebar.selectbox("blindOrVisionDifficulty",blindOrVisionDifficulty)
    select_difficultyWalking = st.sidebar.selectbox("difficultyWalking",difficultyWalking)
    select_difficultyDressingBathing = st.sidebar.selectbox("difficultyDressingBathing",difficultyDressingBathing)
    select_difficultyErrands = st.sidebar.selectbox("difficultyErrands",difficultyErrands)
    select_smokerStatus = st.sidebar.selectbox("smokerStatus",smokerStatus)
    select_chestScan = st.sidebar.selectbox("chestScan",chestScan)
    select_ageCategory = st.sidebar.selectbox("ageCategory",ageCategory)
    select_alcoholDrinkers = st.sidebar.selectbox("alcoholDrinkers",alcoholDrinkers)
    select_pneumoVaxEver = st.sidebar.selectbox("pneumoVaxEver",pneumoVaxEver)


    data = [{
        "sex":select_sex,"generalHealth":select_generalHealth,"PhysicalHealthDays":select_PhysicalHealthDays,
        "physicalActivities":select_physicalActivities,"removedTeeth":select_removedTeeth,"hadAngina":select_hadAngina,
                "hadStroke":select_hadStroke,"hadCOPD":select_hadCOPD,"hadKidneyDisease":select_hadKidneyDisease,"hadArthritis":select_hadArthritis,
        "hadDiabetes":select_hadDiabetes,"deafOrHardOfHearing":select_deafOrHardOfHearing,"blindOrVisionDifficulty":select_blindOrVisionDifficulty,
        "difficultyWalking":select_difficultyWalking,"difficultyDressingBathing":select_difficultyDressingBathing,
        "difficultyErrands":select_difficultyErrands,
        "smokerStatus":select_smokerStatus,"chestScan":select_chestScan,"ageCategory":select_ageCategory,
        "alcoholDrinkers":select_alcoholDrinkers,"pneumoVaxEver":select_pneumoVaxEver
    
    }]
    
    
    return data



df_pred=input_user()
data_parameter = pd.DataFrame(df_pred,index=[0])
st.write(data_parameter )

st.header("Expectations")
prediction = prediction.fc(df_pred) 
st.write(prediction )
