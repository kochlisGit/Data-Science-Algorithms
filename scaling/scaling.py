from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
import numpy as np

__scalers_dict = {
    'Standard': StandardScaler(),
    'MinMax': MinMaxScaler(),
    'MaxAbs': MaxAbsScaler(),
    'Robust': RobustScaler()
}

# Transforms data by the given scaling method.
def transform(scaling_method, data):
    scaler = __scalers_dict[scaling_method]
    return np.array(scaler.fit_transform(data))
