import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st

model = joblib.load("liveModelV1.pk1")

data= pd.read_csv('mobile_price_range_data.csv')
X=data.iloc[:,:-1]
y=data.iloc[:,:-1]

X_train ,X_test, y_train ,y_test = train_test_split (X,y,test_size=0.2,random_state=42)
#Make predictions for X_test set
y_pred =model.predict(X_test)

#Calculate Accuray
accuracy= accuray_score (y_test,y_pred)

#Page title
st.title("Model Accuracy and Real-Time Prediction")

#Display Accuarcy
st.write(f"Model{accuracy}")
#Real time predictions based on user inputs
st.header("Real-Time Prediction")
input_data =[]
for col in X_test.columns:
     input_value =st.number_input( f'Input for feature'{col}' , value='')
     input_data.append(input_value)
input_df=pd.DataFrame([input_data],columns=X_test.columns)     
#Make predictions
if st.button("Predict"):
    prediction =model.predict(input_df)
    st.write(f'prediction:{prediction[0]}')





