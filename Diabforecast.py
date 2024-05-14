import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import streamlit as st
from sklearn.linear_model import LogisticRegression

data = pd.read_csv(r'C:\Users\User\Downloads\archive\diabetes.csv')


nonzero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for x in nonzero:
    data[x] = data[x].replace(0, np.NaN)
    mean = int(data[x].mean(skipna=True))
    data[x] = data[x].replace(np.NaN, mean)


st.image(r"C:\Users\User\Downloads\project.jpg", width=700)
st.markdown('<h1 style="text-align: center; font-size: 3.5rem;color:blue;">DiabForecast</h1>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Predict Diabetes</h3>", unsafe_allow_html=True)
st.write("\n")

st.header('Dataset Info:')
st.write(data.describe())


st.header("Visualization of Dataset:")
st.bar_chart(data)


X = data.iloc[:,0:8]
y = data.iloc[:,8]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

st.sidebar.title('Choose Model')
model = st.sidebar.selectbox(label='Select a Model', options=['K-NN', 'LGR'])

def user_report():
    st.sidebar.title('Input all information')
    pregnancies = st.sidebar.number_input('Pregnancies', step=1)
    glucose = st.sidebar.number_input('Glucose', step=1)
    bp = st.sidebar.number_input('Blood Pressure', step=1)
    skinthickness = st.sidebar.number_input('Skin Thickness', step=1)
    insulin = st.sidebar.number_input('Insulin', step=1)
    bmi = st.sidebar.number_input('BMI')
    dpf = st.sidebar.number_input('Diabetics Pedigree Function')
    age = st.sidebar.number_input('Age', step=1)

    st.sidebar.subheader("Input Values:")
    st.sidebar.text(f"Pregnancies: {pregnancies}")
    st.sidebar.text(f"Glucose: {glucose}")
    st.sidebar.text(f"Blood Pressure: {bp}")
    st.sidebar.text(f"Skin Thickness: {skinthickness}")
    st.sidebar.text(f"Insulin: {insulin}")
    st.sidebar.text(f"BMI: {bmi}")
    st.sidebar.text(f"Diabetics Pedigree Function: {dpf}")
    st.sidebar.text(f"Age: {age}")

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

    return user_data
    
user_data=user_report()
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
    st.subheader('Accuracy of K-NN Model:')
    st.write(f"{accuracy:.2f}%")

    f1score = f1_score(y_test, y_pred)
    st.subheader('F1 Score of K-NN Model:')
    st.write(f"{f1score:.2f}")

    user_result = classifier.predict(userdata_scaled)
    st.header('Your Report:')
    output = 'You have no Diabetes' if user_result[0] == 0 else 'You have Diabetes'
    st.write(output)

# Logistic Regression model
def lgr(X_train, y_train, X_test, y_test, userdata_scaled):
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) * 100
    st.subheader('Accuracy of Logistic Regression Model:')
    st.write(f"{accuracy:.2f}%")

    f1score = f1_score(y_test, y_pred)
    st.subheader('F1 Score of Logistic Regression Model:')
    st.write(f"{f1score:.2f}")

    user_result = classifier.predict(userdata_scaled)
    st.header('Your Report:')
    output = 'You have no Diabetes' if user_result[0] == 0 else 'You have Diabetes'
    st.write(output)




if model == 'K-NN':
    knn(X_train, y_train, X_test, y_test, userdata_scaled)

else:
    lgr(X_train, y_train, X_test, y_test, userdata_scaled)
