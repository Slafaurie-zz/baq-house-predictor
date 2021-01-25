import pandas as pd
import numpy as np
import joblib
import config

def make_prediction(data):
    data_df = pd.DataFrame([data])
    model_pipe = joblib.load(config.MODEL_NAME)
    yhat = model_pipe.predict(data_df).tolist()
    yhat = np.round(yhat, -4).tolist()
    return yhat

