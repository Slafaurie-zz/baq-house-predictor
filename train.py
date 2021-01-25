import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics

from pipeline import model_pipe
import joblib
import config


def train_pipeline(est_pipe, X_train, y_train, X_test, y_test, performCV=True, cv_folds=5):
    est_pipe.fit(X_train, y_train)
    
    if performCV:
        cv_score = cross_val_score(est_pipe, X_train, y_train, cv=cv_folds, scoring='r2')
    
    yhat, yhat_test = est_pipe.predict(X_train), est_pipe.predict(X_test)
    mae_train, mae_test = metrics.mean_absolute_error(y_train, yhat), metrics.mean_absolute_error(y_test, yhat_test)
    r2_train, r2_test = metrics.r2_score(y_train, yhat), metrics.r2_score(y_test, yhat_test)
        
    print('======== Train metrics =========')
    print('MAE:{}'.format(mae_train))
    print('R2:{}'.format(r2_train))
    print('CV score: mean - {} | std - {}'.format(np.mean(cv_score), np.std(cv_score)))
    print()
    print('======== Test metrics =========')
    print('MAE:{}'.format(mae_test))
    print('R2:{}'.format(r2_test))
    print()

    return est_pipe

def train():
    data = pd.read_csv(config.TRAINING_DATA_FILE)[config.FEATURES].dropna()
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=config.TARGET),
        data[config.TARGET],
        test_size = 0.2,
        random_state = config.SEED
    )

    model_fit = train_pipeline(model_pipe, X_train, y_train, X_test, y_test)
    return model_fit


if __name__ == '__main__':
    model = train()
    joblib.dump(model, config.MODEL_NAME)


