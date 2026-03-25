import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title("Ames Housing Price Prediction App")
st.write("Choose a model to predict the house prices in Ames city limits")

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Admin\Downloads\Ames_Housing_Subset.csv")
    return df

df1 = load_data()

st.subheader("Dataset preview")
st.dataframe(df1.head(10))

df1 = df1.select_dtypes(include=[np.number])

x = df1.drop(["SalePrice"], axis =1)
y = df1["SalePrice"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model_name = st.selectbox("Select the model",
                          ["Linear Regression","Decision Tree","Random Forest"])
if model_name == "LinearRegression":
    model = LinearRegression()
elif model_name == "DecisionTreeRegressor":
    model = DecisionTreeRegressor()
else:
    model = RandomForestRegressor()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)
st.subheader("MSE")
st.write(mse)
st.subheader("R2_Score")
st.write(r2)

# input values
input_values = { }
st.subheader("Please enter the input values")
for col in x.columns:
    input_values[col] = st.number_input(f"Enter the {col} :",value = x[col].median())

input_data=pd.DataFrame([input_values])
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success("The prediction of the sales price is: {}".format(prediction))