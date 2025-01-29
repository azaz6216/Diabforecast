import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from fpdf import FPDF



def dict_to_pdf(data, result, pdf_file):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_text_color(0, 102, 204)  # Set title color to dark blue (RGB: 0, 102, 204)
    pdf.set_font("Arial", size=20, style='B')
    pdf.cell(0, 10, txt="Dibforecast", ln=True, align='C')
    pdf.ln(1)  # Space below title

    # Subtitle
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(0, 10, txt="Diabetics Prediction Result", ln=True, align='C')
    pdf.ln(1)  # Space below subtitle

    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, txt="Website: www.diabforecast.com", ln=True, align='C')
    pdf.cell(0, 10, txt="Email: contact@diabforecast.com", ln=True, align='C')
    pdf.cell(0, 10, txt="Phone: +1 234 567 890", ln=True, align='C')
    pdf.ln(2)
    # Set font for the data
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)  # Black text

    # Add user data to the PDF
    for idx, (key, value) in enumerate(data.items()):
        if idx % 2 == 0:
            pdf.set_fill_color(230, 230, 230)  # Alternate row color
        else:
            pdf.set_fill_color(255, 255, 255)  # White background

        pdf.cell(180, 10, txt=f"{key}: {value}", border=0, ln=True, align='L', fill=True)

    pdf.ln(20)
    pdf.set_font("Arial", size=10, style='B')
    pdf.cell(0, 10, txt="About Diabforecast", ln=True, align='L')
    pdf.set_font("Arial", size=8)
    pdf.multi_cell(0, 10, txt=(
        "Diabforecast is a healthcare prediction platform that leverages machine learning to help "
        "individuals predict and manage their health conditions. With accurate models and an easy-to-use "
        "interface, we strive to provide a reliable and accessible tool for both patients and healthcare providers. "
        "Visit our website for more details and updates on our services."
    ))

    # Add space before footer
    pdf.ln(10)

    # Add website, email, and contact number

    # Footer with thank you note
    pdf.set_y(-40)
    pdf.set_font("Arial", size=10, style='I')
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, txt="Thank you for using Diseasecast. Stay healthy!", ln=True, align='C')

    pdf.output(pdf_file)


# Function to perform KNN prediction
def perform_prediction(X_train, y_train, sc_X, user_data):
    # Convert user data to array and scale it
    user_data_array = np.array([[
        user_data['Pregnancies'],
        user_data['Glucose'],
        user_data['Blood Pressure'],
        user_data['Skin Thickness'],
        user_data['Insulin'],
        user_data['BMI'],
        user_data['Diabetics Pedigree Function'],
        user_data['Age']
    ]])

    # Scale user data
    userdata_scaled = sc_X.transform(user_data_array)

    # KNN classifier
    classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(userdata_scaled)

    # Return prediction result
    return 'Negative' if prediction[0] == 0 else 'Positive'



st.image(r"https://github.com/azaz6216/Diabforecast/blob/main/project.jpg?raw=true", width=700)
st.markdown('<h1 style="text-align: center; font-size: 3.5rem;color:blue;">DiabForecast</h1>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Predict Diabetes</h3>", unsafe_allow_html=True)

data = pd.read_csv('https://raw.githubusercontent.com/azaz6216/Diabforecast/main/diabetes.csv')

st.header('Dataset Info')
st.write(data.describe())


st.header("Visualization of Dataset")
st.bar_chart(data)

nonzero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for feature in nonzero:
        data[feature] = data[feature].replace(0, np.NaN)
        mean = int(data[feature].mean(skipna=True))
        data[feature] = data[feature].replace(np.NaN, mean)



    
X = data.iloc[:, 0:8]
y = data.iloc[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

st.sidebar.title('Input all information')
name=st.sidebar.text_input('Name')
gender= st.sidebar.selectbox(label='Gender', options=['Male', 'Female','Others'])
age = st.sidebar.number_input('Age', step=1)
pregnancies = st.sidebar.number_input('Pregnancies', step=1)
glucose = st.sidebar.number_input('Glucose', step=1)
bp = st.sidebar.number_input('Blood Pressure', step=1)
skinthickness = st.sidebar.number_input('Skin Thickness', step=1)
insulin = st.sidebar.number_input('Insulin', step=1)
bmi = st.sidebar.number_input('BMI')
dpf = st.sidebar.number_input('Diabetics Pedigree Function')




st.sidebar.subheader("Input Values:")
st.sidebar.text(f"Name:{name}")
st.sidebar.text(f"Gender:{gender}")
st.sidebar.text(f"Age: {age}")
st.sidebar.text(f"Pregnancies: {pregnancies}")
st.sidebar.text(f"Glucose: {glucose}")
st.sidebar.text(f"Blood Pressure: {bp}")
st.sidebar.text(f"Skin Thickness: {skinthickness}")
st.sidebar.text(f"Insulin: {insulin}")
st.sidebar.text(f"BMI: {bmi}")
st.sidebar.text(f"Diabetics Pedigree Function: {dpf}")


        
user_data = {'Name': name,
            'Gender': gender,
            'Age': age,
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'Blood Pressure': bp,
            'Skin Thickness': skinthickness,
            'Insulin': insulin,
            'BMI': bmi,
            'Diabetics Pedigree Function': dpf,
        }

if st.sidebar.button("Submit"):      

        
  knn_result = perform_prediction(X_train, y_train, sc_X, user_data)

  user_data_result = {
    'Name': name,
    'Gender': gender,
    'Age': age,
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'Blood Pressure': bp,
    'Skin Thickness': skinthickness,
    'Insulin': insulin,
    'BMI': bmi,
    'Diabetics Pedigree Function': dpf,
    'Result': knn_result
        }

        
  pdf_file = f"{name}_diesesecast_report.pdf"
  dict_to_pdf(user_data_result, knn_result, pdf_file)

  output=knn_result
  container=st.container(border=True)
  container.write(f"Result: {knn_result}")
  st.markdown("---")
            

  with open(pdf_file, "rb") as f:
    pdf_data = f.read()
    st.download_button(
    label="Download Report",
    data=pdf_data,
    file_name=pdf_file,  
    mime="application/pdf")
