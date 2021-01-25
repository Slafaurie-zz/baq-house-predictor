from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.ensemble import GradientBoostingRegressor

import config

ohe = OneHotEncoder()
scaler = StandardScaler()
ct = make_column_transformer(
    (ohe, config.CATEGORICAL_VARS),
    remainder = 'passthrough'
)

model = GradientBoostingRegressor(
    learning_rate = config.LEARNING_RATE,
    max_depth = config.MAX_DEPTH,
    loss = config.LOSS,
    n_estimators = config.N_ESTIMATORS,
    max_features = config.MAX_FEATURES,
    min_samples_leaf = config.MIN_SAMPLES_LEAF,
    min_samples_split = config.MIN_SAMPLES_SPLIT,
    random_state = config.SEED
)

model_pipe = make_pipeline(
    ct,
    scaler,
    model
)

