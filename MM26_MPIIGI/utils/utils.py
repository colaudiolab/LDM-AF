import numpy as np
import pandas as pd


def concordance_correlation_coefficient(y_true, y_pred):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).dropna()
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    cor = np.corrcoef(y_true, y_pred)[0, 1]
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    sd_true, sd_pred = np.std(y_true), np.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2 + 1e-12
    return numerator / denominator


def logmsg(logfile, msg):
    print(msg)
    with open(logfile, 'a') as f:
        f.write(msg + '\n')