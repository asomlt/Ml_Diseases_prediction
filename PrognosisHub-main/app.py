import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import openai
from streamlit_chat import message
import google.generativeai as genai
from IPython.display import Markdown
import pathlib
import textwrap


# from IPython.display import Markdown
# Only if you're using Google Colab
# from google.colab import userdata

# Set your OpenAI API key here

# Set page configuration
st.set_page_config(
    page_title="PrognosisHub",
    layout="wide",
    page_icon="🧑‍⚕️",
    initial_sidebar_state="expanded",
)

# Set background color
st.markdown(
    """
    <style>
        body {
            background-color: #f0f5f5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open(f'{working_dir}/Saved Models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open(f'{working_dir}/Saved Models/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open(f'{working_dir}/Saved Models/parkinsons_model.sav', 'rb'))

breast_cancer_model=pickle.load(open(f'{working_dir}/Saved Models/breast_cancer_model.sav','rb'))

# sidebar for navigation
with st.sidebar:
    # options = ['Home', 'Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Breast Cancer Prediction']

    options = ['Home','Heart Disease Prediction','Diabetes Prediction','chat']
    selected = option_menu('Multiple Disease Prediction System',
                           options,
                           menu_icon='hospital-fill',
                           icons=['house-door', 'activity', 'heart', 'person','file-earmark-medical'],
                           default_index=0)

# Home Page
if selected == 'Home':
    st.title('Welcome to Health Assistant')
    
    st.markdown("""
    ## PrognosisHub-Disease Prediction System
    
    Welcome to the PrognosisHub a Multiple Disease Prediction System, created by Kanchan Rai. 
    This application allows you to predict the likelihood of having different diseases: Diabetes, Heart Disease, Parkinson's Disease, and Breast Cancer.
    The prediction is based on 4 different machine learning models.
    Use the sidebar to navigate to specific prediction pages.

    ### Breast Cancer Prediction
    This model utilizes scikit-learn to implement a logistic regression model for breast cancer prediction.
    After loading the breast cancer dataset and organizing it into features and target variables, the data is split into training and testing sets.
    A logistic regression model is then trained using the training data.
    The accuracy of the model is evaluated on both the training and testing sets, providing insights into its performance.
    Subsequently, the trained model is used to make predictions on new data, showcasing its application for predicting whether a tumor is malignant or benign.
    Additionally, the code includes a mechanism to save the trained model using the pickle library, facilitating reuse without the need for retraining.

    ### Diabetes Prediction
    This Model utilizes the scikit-learn library to develop a Support Vector Machine (SVM) classifier for predicting diabetes based on the PIMA Diabetes dataset.
    It involves data preprocessing, including standard scaling, and the dataset is split into training and testing sets. A linear kernel SVM model is trained and evaluated for accuracy.
    The script also showcases making predictions on new data and uses the pickle library to save the trained model for future use, enhancing efficiency and reusability in diabetes prediction tasks.

    ### Heart Disease Prediction
    This model employs the scikit-learn library to construct a logistic regression model for predicting heart disease based on a provided dataset.
    The dataset is loaded and examined to understand its structure and content. Essential data preprocessing steps, such as handling missing values and splitting into training and testing sets, are executed.
    Subsequently, the logistic regression model is trained utilizing the training data, and its accuracy is assessed on both the training and test datasets.
    Furthermore, a predictive system is implemented to make predictions on new input data.
    The model concludes by saving the trained model using the pickle library for future deployment in heart disease prediction tasks.

    ### Parkinson's Disease Prediction
    This model utilizes the scikit-learn library to implement a Support Vector Machine (SVM) for predicting Parkinson's disease based on a provided dataset.
    The dataset is loaded and analyzed, including an examination of the first and last 10 rows, checking for missing values, and exploring basic statistics.
    The dataset is then split into training and testing sets, and the features are standardized using StandardScaler. A linear SVM model is trained on the standardized training data, and its accuracy is evaluated on both the training and test datasets.
    Finally, a predictive system is established to make predictions on new input data, and the trained SVM model is saved using the pickle library for future use in Parkinson's disease prediction tasks.
""", unsafe_allow_html=True)
    
    # GitHub icons and link on the left
    st.sidebar.subheader('Connect with the Creator:')
    st.sidebar.write("[Arpit]()")


if selected =='chat':

    

# Set your Google API key here
    genai.api_key = "sss"

    # Function to call the Google Gemini API
    def call_gemini(prompt):
        model  = genai.GenerativeModel('gemini-pro')
        completions= model.generate_content(prompt)

        return completions.choices[0].text.strip()

    # Title for the Streamlit app
    st.title("Gemini ChatBot With Streamlit")

    # Function to get user input
    def get_text():
        input_text = st.text_input("Write here")
        return input_text

    # Get user input
    user_input = get_text()

    # If user input is provided, call the Gemini API and display the response
    if user_input:
        output = call_gemini(user_input)
        
        # Display user input
        st.text("User: " + user_input)
        
        # Display Gemini response
        st.text("Gemini: " + output)


























































# if selected == 'Heart Disease Prediction':

#     # page title
#     st.title('Heart Disease Prediction using ML')

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         age = st.text_input('Age')

#     with col2:
#         sex = st.text_input('Sex')

#     with col3:
#         cp = st.text_input('Chest Pain types')

#     with col1:
#         trestbps = st.text_input('Resting Blood Pressure')

#     with col2:
#         chol = st.text_input('Serum Cholestoral in mg/dl')

#     with col3:
#         fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

#     with col1:
#         restecg = st.text_input('Resting Electrocardiographic results')

#     with col2:
#         thalach = st.text_input('Maximum Heart Rate achieved')

#     with col3:
#         exang = st.text_input('Exercise Induced Angina')

#     with col1:
#         oldpeak = st.text_input('ST depression induced by exercise')

#     with col2:
#         slope = st.text_input('Slope of the peak exercise ST segment')

#     with col3:
#         ca = st.text_input('Major vessels colored by flourosopy')

#     with col1:
#         thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
# '''if selected == 'Heart Disease Prediction':
#     # page title
#     st.title('Heart Disease Prediction using ML')

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         age = st.text_input('Age (years)', help='Enter the age in years')
#         trestbps = st.text_input('Resting Blood Pressure (mm Hg)', help='Enter the resting blood pressure in mm Hg')
#         restecg = st.text_input('Resting Electrocardiographic Results', help='Enter the resting ECG result')

#     with col2:
#         sex = st.selectbox('Sex', ['Male', 'Female'], help='Select the gender')
#         chol = st.text_input('Serum Cholesterol (mg/dl)', help='Enter the serum cholesterol in mg/dl')
#         thalach = st.text_input('Maximum Heart Rate Achieved', help='Enter the maximum heart rate achieved')

#     with col3:
#         cp = st.selectbox('Chest Pain types', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'], help='Select the type of chest pain')
#         fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'], help='Select if fasting blood sugar is > 120 mg/dl')
#         exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'], help='Select if exercise induced angina is present')

#     with col1:
#         oldpeak = st.text_input('ST Depression Induced by Exercise Relative to Rest', help='Enter the ST depression induced by exercise relative to rest')
#         slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'], help='Select the slope of the peak exercise ST segment')

#     with col2:
#         ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', ['0', '1', '2', '3'], help='Select the number of major vessels colored by fluoroscopy')
#         thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'], help='Select the thalassemia type')

#     # Button to trigger the prediction
#     if st.button('Predict Heart Disease'):
#         # Validate input values
#         if '' in [age, trestbps, chol, thalach, oldpeak]:
#             st.error('Please fill in all required fields.')
#         else:
#             # Convert input values to float
#             user_input = [float(x) if x != '' else 0 for x in [age, trestbps, chol, thalach, oldpeak]]
#             user_input.extend([sex, cp, fbs, restecg, exang, slope, ca, thal])

#             # Perform predictions and display results
#             heart_prediction = heart_disease_model.predict([user_input])
#             if heart_prediction[0] == 1:
#                 heart_diagnosis = 'The person is predicted to have heart disease.'
#             else:
#                 heart_diagnosis = 'The person is predicted to not have heart disease.'

#             st.success(heart_diagnosis)

#     # code for Prediction
#     heart_diagnosis = ''

#     # creating a button for Prediction

#     if st.button('Heart Disease Test Result'):

#         user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

#         user_input = [float(x) for x in user_input]

#         heart_prediction = heart_disease_model.predict([user_input])

#         if heart_prediction[0] == 1:
#             heart_diagnosis = 'The person is having heart disease'
#         else:
#             heart_diagnosis = 'The person does not have any heart disease'

#     st.success(heart_diagnosis)'''



# Assuming you have loaded your model earlier
# heart_disease_model = load_model()

# selected = st.sidebar.selectbox("Select Prediction Task", ["Heart Disease Prediction"])

if selected == 'Heart Disease Prediction':
    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age (years)', help='Enter the age in years')
        trestbps = st.text_input('Resting Blood Pressure (mm Hg)', help='Enter the resting blood pressure in mm Hg')
        restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T Wave Abnormality', 'Probable or Definite Left Ventricular Hypertrophy'], help='Select the resting ECG result')

    with col2:
        sex = st.selectbox('Sex', ['Male', 'Female'], help='Select the gender')
        chol = st.text_input('Serum Cholesterol (mg/dl)', help='Enter the serum cholesterol in mg/dl')
        thalach = st.text_input('Maximum Heart Rate Achieved', help='Enter the maximum heart rate achieved')

    with col3:
        cp = st.selectbox('Chest Pain types', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'], help='Select the type of chest pain')
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'], help='Select if fasting blood sugar is > 120 mg/dl')
        exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'], help='Select if exercise induced angina is present')

    with col1:
        oldpeak = st.text_input('ST Depression Induced by Exercise Relative to Rest', help='Enter the ST depression induced by exercise relative to rest')
        slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'], help='Select the slope of the peak exercise ST segment')

    with col2:
        ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', ['0', '1', '2', '3'], help='Select the number of major vessels colored by fluoroscopy')
        thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'], help='Select the thalassemia type')

    # Button to trigger the prediction
    if st.button('Predict Heart Disease'):
        # Validate input values
        if '' in [age, trestbps, chol, thalach, oldpeak]:
            st.error('Please fill in all required fields.')
        else:
            # Convert categorical variables to numeric representations if necessary
            sex_mapping = {'Male': 0, 'Female': 1}
            cp_mapping = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
            fbs_mapping = {'No': 0, 'Yes': 1}
            restecg_mapping = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Probable or Definite Left Ventricular Hypertrophy': 2}
            exang_mapping = {'No': 0, 'Yes': 1}
            slope_mapping = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
            ca_mapping = {'0': 0, '1': 1, '2': 2, '3': 3}
            thal_mapping = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}

            # Map categorical variables to numeric representations
            sex_numeric = sex_mapping.get(sex, 0)
            cp_numeric = cp_mapping.get(cp, 0)
            fbs_numeric = fbs_mapping.get(fbs, 0)
            restecg_numeric = restecg_mapping.get(restecg, 0)
            exang_numeric = exang_mapping.get(exang, 0)
            slope_numeric = slope_mapping.get(slope, 0)
            ca_numeric = ca_mapping.get(ca, 0)
            thal_numeric = thal_mapping.get(thal, 0)

            # Convert input values to numeric
            user_input_numeric = [float(age), sex_numeric, cp_numeric, float(trestbps), float(chol), fbs_numeric, restecg_numeric, float(thalach), exang_numeric, float(oldpeak), slope_numeric, ca_numeric, thal_numeric]

            # Reshape user_input_numeric to match model's expected input shape
            user_input_numeric = np.array(user_input_numeric).reshape(1, -1)

            # Perform predictions and display results
            heart_prediction = heart_disease_model.predict(user_input_numeric)
            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is predicted to have heart disease.'
            else:
                heart_diagnosis = 'The person is predicted to not have heart disease.'

            st.success(heart_diagnosis)
# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

