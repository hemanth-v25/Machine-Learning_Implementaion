import numpy as np

def bce(y_true, y_pred):
    #assuming shape as (batch_size, )
    loss = -np.mean(y_true * np.log(y_pred + 1e-7) + (1 - y_true) * np.log(1 - y_pred + 1e-7))
    return loss

def rmse(y_true, y_pred):
    #assuming shape as (batch_size,..,..)
    loss = -np.sqrt(np.mean(np.square(y_true - y_pred), axis=0))
    return loss

