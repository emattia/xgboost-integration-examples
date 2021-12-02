# data processing
from sklearn.datasets import make_swiss_roll, make_moons
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost

def get_data():
  # Generate sythethic data
  df = pd.DataFrame()
  # Swiss Roll Synthetic Datasets
  num_rolls = 4
  roll_samples = 20
  roll_noise = 1
  roll_stretch = 1
  for n in range(num_rolls):
    X, y = make_swiss_roll(n_samples=roll_samples, noise=roll_noise)
    df_sw = pd.DataFrame()
    df_sw['dimension-0'] = X[:,0]*(1.0 + roll_stretch*n)
    df_sw['dimension-1'] = X[:,1]*(1.0 + roll_stretch*n)
    df_sw['dimension-2'] = X[:,2]*(1.0 + roll_stretch*n)
    df_sw['y'] = n
    df = pd.concat([df, df_sw])
  # Moons Sythetic Datasets
  num_moons = 1
  moon_samples = 20
  moon_noise = 0.715
  moon_stretch = 0.122
  for m in range(num_moons):
    X, y = make_moons(n_samples=moon_samples, noise=moon_noise)
    df_m = pd.DataFrame()
    df_m['dimension-0'] = X[:,0]*10*(1.0 + moon_stretch*m)
    df_m['dimension-1'] = X[:,1]*10*(1.0 + moon_stretch*m)
    df_m['dimension-2'] = (X[:,0] + X[:,1])*10*(1.0 + moon_stretch*m)
    df_m['y'] = y + num_rolls + m*2
    df = pd.concat([df, df_m])
  
  # split data
  y = df.pop('y')
  X = df
  Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4)
  
  dtrain = xgboost.DMatrix(data=Xtrain, label=ytrain) 
  dtest = xgboost.DMatrix(data=Xtest, label=ytest) 
  return (dtrain, dtest, y)