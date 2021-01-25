TRAINING_DATA_FILE = 'baseline-data.csv'
SEED = 0
MODEL_NAME = 'gbregressor_house.joblib'


TARGET = 'mvalorarriendo'

FEATURES = [
    'mvalorarriendo',
    'mareac',
    'mnrocuartos',
    'mnrobanos',
    'mnrogarajes',
    'mzona_nombre'
]

CATEGORICAL_VARS = ['mzona_nombre']

############# MODEL PARAMETERS ##############
LEARNING_RATE = 0.05
LOSS = 'ls'
MAX_DEPTH = 3
MAX_FEATURES = 'auto'
MIN_SAMPLES_LEAF = 20
MIN_SAMPLES_SPLIT = 2
N_ESTIMATORS = 100


