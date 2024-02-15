import pandas as pd
import pickle

data = pd.read_pickle('Data/Data_3Days.pkl')
print('Data loaded')
pickle.HIGHEST_PROTOCOL = 4
data.to_hdf('Data/Data_3Days.hdf', 'df')
print('Data saved')
