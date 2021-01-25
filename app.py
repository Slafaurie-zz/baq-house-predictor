from predict import make_prediction
import numpy as np
import streamlit as st


def main():
    st.title('Baq House Predictor')
    description()
    params = get_params()
    yhat = int(make_prediction(params)[0])
    st.subheader(f"Estimated Rent Price: ${yhat:,}")
    credit()

def description():
    st.text("""
    It's an End-to-End project to develop a regression model to 
    estimate flat rent prices in my home City: Barranquilla.
    The development process is divided in 4 steps: 

    1. Get Data
    2. Analyze Data
    3. Build Model
    4. Deployment

    In the first step, The data is obtained through Web Scraping of an 
    online marketplace for real estate using requests and pandas libraries. 
    The code is deployed in AWS as a Lambda function and runs once a week.
    
    In the second step, The goal is to know the data, to perform descriptive statistics, 
    check variables distribution and relationship with target variable, check cardinality
    and percentage of null values. 
    
    In the third step, the goal is to build the regression model using the variables selected
    from the second step. Multiple models area tested and the best fit is chosen. 
    After that, the chosen model is slightly improved using Hyperparameter Tuning.

    In the last step, the code to build the model is refactored 
    to a more production style format to easily integrate 
    with either an API with Flask or this Web App. 
    The code include a pipeline to retrain the model if needed. 
    This step also include the front-end in Streamlit 
    and configuration files to deploy the model in Heroku.

    In summary, these are the technologies utilized: 
    - Python: Requests, Numpy, Pandas, Matplotlib, Scipy, 
              Seaborn, Scikit-learn, Streamlit, boto3.
    - AWS: Lambda, IAM, Event-Bridge, S3.
    - Heroku
    """)

def credit():
    st.text("""
    Author: Sebastian Lafaurie
    LinkedIn: https://www.linkedin.com/in/sebastianlafaurie/
    """)

def get_params():
    params = {}
    params['mareac'] = st.sidebar.slider('Built area', 20,300)
    params['mnrocuartos'] = st.sidebar.slider('Room number', 1,10)
    params['mnrobanos'] = st.sidebar.slider('Bathroom number', 1,10)
    params['mnrogarajes'] = st.sidebar.slider('Garage number', 0,10)
    params['mzona_nombre'] = st.sidebar.selectbox('Select house zone', 
                            ('Oriente', 'Noroccidente', 'Otros' , 'Suroccidente',
                            'Industrial y Alrededores', 'Villa Campestre')
                            )
    return params

if __name__ == '__main__':
    main()