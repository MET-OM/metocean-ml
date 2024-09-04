import matplotlib.pyplot as plt
import sklearn
from metocean_stats.stats import ml
import numpy as np
import xarray as xr
import pandas as pd
from metocean_stats.stats.aux_funcs import *
import warnings
warnings.filterwarnings('ignore')

ds = xr.open_dataset('/home/konstantinosc/github/dnora/examples/output/ww3_spec_NORA3_Sula_20180101T0000-20191231T2300.nc')

# Define training and validation period:
start_training = '2018-01-01'
end_training   = '2018-12-31'
start_valid    = '2019-01-01'
end_valid      = '2019-12-31'

# Select method and variables for ML model:
model='GBR' # 'SVR_RBF', 'LSTM', GBR
var_origin = ['efth']
var_train  = ['efth']
station_origin = 2 # A location 62.427002, 6.044994
station_train = 1 # B location 62.404, 6.079001

df_spec_origin = convert_spec_to_dataframe(ds=ds.isel(station=station_origin), spec_name = 'efth', dir_name='direction', freq_name='frequency' )
df_spec_train= convert_spec_to_dataframe(ds=ds.isel(station=station_train), spec_name = 'efth', dir_name='direction', freq_name='frequency' )
df_spec_ml  = create_zeros_dataframe_like(df_spec_train)

# Run ML model:
#ds_ml = xr.full_like(ds[var_origin[0]][:,station_train,:,:], fill_value=np.nan)
for i in range(len(df_spec_ml.columns)):
    print(i) 
    df_spec_ml[:,i] = ml.predict_ts(ts_origin = df_spec_origin,var_origin=df_spec_origin.columns[:],ts_train  = df_spec_train.loc[start_training:end_training],var_train=df_spec_train.columns[i], model=model)
    


#ds_ml.to_netcdf(model+'_ml_spec.nc')
ds_ml = xr.open_dataset(model+'_ml_spec.nc')
# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# Plot data on the first subplot
c1 = axs[1].pcolormesh(ds_ml[var_train[0]][-3000, :, :])
axs[1].set_title('ML predicted spectrum')
# Plot data on the second subplot
c0 = axs[0].pcolormesh(ds[var_train[0]][-3000, station_train, :, :])
axs[0].set_title('spectum')
# Optionally, add colorbars to each subplot
fig.colorbar(c0, ax=axs[0])
fig.colorbar(c1, ax=axs[1])
# Show the plot
plt.tight_layout()
plt.show()
