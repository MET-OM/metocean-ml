import matplotlib.pyplot as plt
import sklearn
from metocean_ml import ml
import numpy as np
import xarray as xr
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def convert_dataframe_to_spec(df, spec_name='efth', dir_name='direction', freq_name='frequency'):
    # Extract direction and frequency indices from column names
    dir_freq_indices = [col.split('_')[1:] for col in df.columns]
    dir_indices = [int(indices[0][1:]) for indices in dir_freq_indices]
    freq_indices = [int(indices[1][1:]) for indices in dir_freq_indices]

    # Determine the unique direction and frequency indices
    unique_dir_indices = np.unique(dir_indices)
    unique_freq_indices = np.unique(freq_indices)

    # Create empty xarray with the correct dimensions
    ds = xr.DataArray(
        np.empty((len(df), len(unique_dir_indices), len(unique_freq_indices))),
        coords={
            'time': df.index,
            dir_name: unique_dir_indices,
            freq_name: unique_freq_indices
        },
        dims=['time', dir_name, freq_name]
    )

    # Fill the xarray with data from the DataFrame
    for col, dir_idx, freq_idx in zip(df.columns, dir_indices, freq_indices):
        ds.loc[:, dir_idx, freq_idx] = df[col]

    return ds



def convert_spec_to_dataframe(ds , spec_name = 'efth', dir_name='direction', freq_name='frequency'):
    
    # Reshape the data so that each combination of direction and frequency has its own column
    df_spec = (
        ds[spec_name].stack(combination=[dir_name, freq_name])  # Flatten direction and frequency dimensions
        .to_pandas()  # Convert to a pandas DataFrame
    )
    # Rename columns to reflect direction and frequency combinations
    df_spec.columns = [f"{spec_name}_{dir_name[0]}{dir_idx}_{freq_name[0]}{freq_idx}" 
                         for dir_idx, freq_idx in zip(*df_spec.columns.codes)]

    df_spec = df_spec.loc[:,~df_spec.columns.duplicated()].copy()
    return df_spec

def create_zeros_dataframe_like(df):
    empty_df = 0*df #pd.DataFrame(columns=df.columns)
    
    return empty_df


ds = xr.open_dataset('ww3_spec_NORA3_Sula_20180101T0000-20191231T2300.nc')

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
    df_spec_ml[df_spec_ml.columns[i]] = ml.predict_ts(ts_origin = df_spec_origin,var_origin=[df_spec_origin.columns[i]],
                                        ts_train  = df_spec_train.loc[start_training:end_training],
                                        var_train=df_spec_train.columns[i], 
                                        model=model)
    #else:
    #    df_spec_ml[df_spec_ml.columns[i]] = ml.predict_ts(ts_origin = df_spec_origin,var_origin=df_spec_origin.columns[i-1:i],
    #                                    ts_train  = df_spec_train.loc[start_training:end_training],
    #                                    var_train=df_spec_train.columns[i], 
    #                                    model=model)


df_spec_ml.to_csv('data.csv')
breakpoint()

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
