#importing all the modules required for forecasting the data
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date
from prophet import Prophet
import yfinance as yf
import tensorflow as tf
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import plotly.tools as plotly_tools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor



START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Price Prediction Website')

stocks = st.text_input('Please Enter Stock Ticker')

n_years = st.slider("Years of Prediction:", 1, 8)
periods = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data
data_load_state = st.text("Load data....")
data = load_data(stocks)
data_load_state.text("data loading.... done!")

st.subheader('Raw Data')
st.write(data.tail(45))

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Open Price", mode='lines', line=dict(color='blue', width=2), hoverinfo="x+y+name"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price", mode='lines', line=dict(color='red', width=2), hoverinfo="x+y+name"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

#Letting user select what to predict
choice = st.selectbox("Choose what you want to predict", ['Open', 'Close'])

#Choose the model with which you want to predict the data


#getting data for predictions
def what_to_predict(column):
    if column == 'Open':
        Y = data['Open'].astype('float64')  # Convert to float64
        train = data.drop(columns=['Open'])
    else:
        Y = data['Close'].astype('float64')  # Convert to float64
        train = data.drop(columns=['Close'])
    return train, Y

# Assuming 'choice' is defined elsewhere
train, Y = what_to_predict(choice)
train.set_index('Date', inplace=True)
st.write(train.tail(30)) 
#Splitting the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(train, Y, train_size=0.5)
#standardizing the data
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

#define hyperparameter for each model
params = {
    "Linear Regression":{
        "normalize":[True, False]
    },
    "Support Vector Machine":{
        "C":[0.1, 1, 10],
        "kernel":["linear","rbf"]
    },
    "Decision Tree Regression":{
        "max depth":[None, 10, 20, 30]
    },
    "Random Forest":{
        "n_estimators":[50,100,150],
        "max_depth":[None, 10, 20, 30]  
    },
    "Random Forest": {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 10, 20, 30]
    },
    "Logistic Regression":{
        "C":[0.1, 1, 10],
        "penalty":["11","12"]
    },
    "Support Vector Machine Classifier":{
        "C":[0.1, 1, 10],
        "kernel":["linear", "rbf"]
    },
    "Gradient Booster Classifier":{
        "n_estimators":[50,100,150],
        "max depth":[3,4,5]
    }

}

st.title("Machine Learning Model Selection")

# Select the model
selected_model = st.selectbox("Select a model", ["Linear Regression", "Support Vector Machine", "Decision Tree Regressor", "Random Forest Regressor", "KNN Regression", "Gradient Booster Regressor", "Niave Baise Model"])


def train_and_evaluate_model(selected_model, X_train, Y_train, X_test, Y_test):
    if selected_model == "Linear Regression":
        if st.button("Train Model"):
            model = LinearRegression()
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)
            r2_linear = r2_score(Y_test, y_pred)
            accuracy = r2_linear*100
            st.write("Accuracy of Linear Regression is :", accuracy)

    elif selected_model == "Support Vector Machine":
        C = st.slider("Regularization paramete(C)", min_value=0.1, max_value=10.0, step=0.1)
        epsilon = st.slider("Epsilon", min_value=0.01, max_value=1.0, step=0.01)
        if st.button("Train Model"):
            model = SVR(C=C, epsilon=epsilon)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)
            r2_svm = r2_score(Y_test, y_pred)
            accuracy = r2_svm*100
            st.write("Mean Absolute error in Support Vector Machine is :", accuracy)

    elif selected_model == "Decision Tree Regressor":
        max_depth = st.selectbox("Max depth",[None, 10, 20, 30])
        if st.button("Train Model"):
            model = DecisionTreeRegressor(max_depth=max_depth)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)
            r2_DT = r2_score(Y_test, y_pred)
            accuracy = r2_DT*100
            st.write("Decision Tree Regressor accuracy is :", accuracy)

    elif selected_model == "Random Forest Regressor":
        n_estimators = st.slider("Number of estimators", min_value=50, max_value=150, step=10)
        max_depth = st.selectbox("Max depth", [None, 10,20,30])
        if st.button("Train Model"):
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)
            r2_RF = r2_score(Y_test, y_pred)
            accuracy = r2_RF*100
            st.write("Accuracy of Random forest Regrssion is: ", accuracy)

    elif selected_model == "KNN Regression":
        n_neighbors = st.slider("Number of Neighbors (k)", min_value=1, max_value=10)
        weights = st.selectbox("Weight Function", ["uniform", "distance"])
        if st.button("Train Model"):
            model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)
            r2_KNN = r2_score(Y_test, y_pred)
            accuracy = r2_KNN*100
            st.write("Accuracy for Quantile Regression is: ", accuracy)

    elif selected_model == "Gradient Booster Regressor":
        n_estimator = st.slider("Number of Estimators", min_value=50, max_value=150, step=10)
        max_depth = st.selectbox("Maximum depth for model", [3,4,5])
        if st.button("Train Model"):
            model = GradientBoostingRegressor(n_estimators=n_estimator, max_depth=max_depth)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)
            r2_GB = r2_score(Y_test, y_pred)
            accuracy = r2_GB*100
            st.write("Gradient Boost Regrssor accuracy is: ", accuracy)

    elif selected_model == "Niave Baise Model":
        alpha = st.number_input("Smoothing paramter", value=1, min_value=0)
        fit_prior = st.checkbox("Fit Class Prior probabilities")
        if st.button("Train Model"):
            model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)
            r2_NB = r2_score(Y_test, y_pred)
            accuracy = r2_NB*100
            st.write("Naive Baise Model accuracy is: ", accuracy)

train_and_evaluate_model(selected_model, X_train, Y_train, X_test, Y_test)

#Getting various trends in market data

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date":"ds", "Close":"y" })
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=periods)
forecast = m.predict(future)
st.subheader('Stock Trend Data')
st.write(forecast.tail(5))
figure = m.plot_components(forecast)
st.write(figure)









