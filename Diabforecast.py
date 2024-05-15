import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import streamlit as st
from sklearn.linear_model import LogisticRegression
from fpdf import FPDF
import base64

data = pd.read_csv(r'https://raw.githubusercontent.com/azaz6216/Diabforecast/main/diabetes.csv')


nonzero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for x in nonzero:
    data[x] = data[x].replace(0, np.NaN)
    mean = int(data[x].mean(skipna=True))
    data[x] = data[x].replace(np.NaN, mean)


st.image(r"https://github.com/azaz6216/Diabforecast/blob/main/project.jpg?raw=true", width=700)
st.markdown('<h1 style="text-align: center; font-size: 3.5rem;color:blue;">DiabForecast</h1>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Predict Diabetes</h3>", unsafe_allow_html=True)
st.write("\n")

st.header('Dataset Info')
st.write(data.describe())


st.header("Visualization of Dataset")
st.bar_chart(data)


X = data.iloc[:,0:8]
y = data.iloc[:,8]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

st.sidebar.title('Choose Model')
model = st.sidebar.selectbox(label='Model', options=['K-NN', 'LGR'])




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











user_data = {
        'pregnancies': pregnancies,
        'glucose': glucose,  
        'bp': bp,
        'skinthickness': skinthickness,
        'insulin': insulin,
        'bmi': bmi,
        'dpf': dpf,
        'age': age
    }

    

values_list = list(user_data.values())
user_data_array = np.array([values_list])
userdata_scaled = sc_X.transform(user_data_array)





# KNN model
def knn(X_train, y_train, X_test, y_test, userdata_scaled):
    classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred) * 100
    st.subheader('Accuracy of K-NN Model')
    container=st.container(border=True)
    container.write(f"{accuracy:.2f}%")

    f1score = f1_score(y_test, y_pred)
    st.subheader('f1 Score of K-NN Model')
    container=st.container(border=True)
    container.write(f"{f1score:.2f}")

    user_result = classifier.predict(userdata_scaled)
    st.subheader('Result')
    output ='  '  
    if user_result[0] == 0:
        output='Negative'
        container=st.container(border=True)
        container.write(output)
    else:
        output='Possitive'
        container=st.container(border=True)
        container.write(output)
  



# Logistic Regression model
def lgr(X_train, y_train, X_test, y_test, userdata_scaled):
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) * 100
    st.subheader('Accuracy of Logistic Regression Model')
    container=st.container(border=True)
    container.write(f"{accuracy:.2f}%")

    f1score = f1_score(y_test, y_pred)
    st.subheader('f1 Score of Logistic Regression Model')
    container=st.container(border=True)
    container.write(f"{f1score:.2f}")

    user_result = classifier.predict(userdata_scaled)
    st.subheader('Result')
    output ='  '  
    if user_result[0] == 0:
        output='Negative'
        container=st.container(border=True)
        container.write(output)
    else:
        output='Possitive'
        container=st.container(border=True)
        container.write(output)
  



if model == 'K-NN':
    knn(X_train, y_train, X_test, y_test, userdata_scaled)

else:
    lgr(X_train, y_train, X_test, y_test, userdata_scaled)



def knnreport(X_train, y_train, userdata_scaled):
    classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
    classifier.fit(X_train, y_train)
    prediction= classifier.predict(userdata_scaled)

    output1  ='  '  
    if prediction[0] == 0:
        output1='Negative'
    
    else:
        output1='Possitive'
    return output1  


def lgrreport(X_train, y_train, userdata_scaled):
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    prediction1= classifier.predict(userdata_scaled)

    output1  ='  '  
    if prediction1[0] == 0:
        output1='Negative'
    
    else:
        output1='Possitive'
    return output1 

x=knnreport(X_train, y_train, userdata_scaled)
y=lgrreport(X_train, y_train, userdata_scaled)


user_report1 = {
        'Diabforecast':"Report",
        'Name':name,
        'Gender':gender,
        'Age': age,
        'Pregnancies': pregnancies,
        'Glucose': glucose,  
        'Bp': bp,
        'Skinthickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'Dpf': dpf,
        'Result(K-NN)':x,
        'Result(LGR)':y
    }



# Function to convert dictionary to PDF
def dict_to_pdf(data, pdf_file):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for key, value in data.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True, align='L')

    pdf.output(pdf_file)

# Create PDF and download link
pdf_file = "user_report.pdf"
dict_to_pdf(user_report1, pdf_file)

with open(pdf_file, "rb") as f:
    pdf_data = f.read()

st.subheader('Download Your Report Here')
if st.download_button(label="Click to Download", data=pdf_data, file_name=pdf_file, mime="application/pdf"):
    st.write("PDF file downloaded successfully.")
