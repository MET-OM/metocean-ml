import matplotlib.pyplot as plt
import sklearn
from metocean_api import ts
from metocean_stats.stats import ml
import numpy as np
import pandas as pd


# når du endrer koordinater må du huske å bytte fra load til import
lat_b= 62.976446
lon_b= 6.586304
distance = 0.05

#punktet vi skal prediktere
df_norac_b = ts.TimeSeries(lon=lon_b, lat= lat_b,start_time='2017-01-01', end_time='2019-12-31' ,variable=['hs','dir','tp'],product='NORAC_wave')
#df_norac_b.import_data(save_csv=True)
df_norac_b.load_data(local_file=df_norac_b.datafile)
df_norac_b.data.columns = [str(col) + '_C_b' for col in df_norac_b.data.columns]


# Import nora data
#prøv å endre +- for å se endringene
df_nora3_a = ts.TimeSeries(lon=lon_b-distance, lat=lat_b-distance,start_time='2017-01-01', end_time='2019-12-31' ,product='NORA3_wave_sub')
#df_nora3_a.import_data(save_csv=True)
df_nora3_a.load_data(local_file=df_nora3_a.datafile)
df_nora3_a.data.columns = [str(col) + '_a' for col in df_nora3_a.data.columns]



df_nora3_c = ts.TimeSeries(lon=lon_b+distance, lat=lat_b+distance,start_time='2017-01-01', end_time='2019-12-31' ,product='NORA3_wave_sub')
#df_nora3_c.import_data(save_csv=True)
df_nora3_c.load_data(local_file=df_nora3_c.datafile)
df_nora3_c.data.columns = [str(col) + '_c' for col in df_nora3_c.data.columns]



df_nora3 = pd.concat([df_nora3_a.data, df_norac_b.data, df_nora3_c.data], axis=1)

# Define training and validation period:
start_training = '2017-01-01'
end_training = '2018-12-31'
start_valid = '2019-01-01'
end_valid = '2019-12-31'

# Select method and variables for ML model:
model= 'GBR' #'LSTM' # 'SVR_RBF', 
var_origin = ['hs_a','hs_c']
var_train = ['hs_C_b']
# Run ML model:
ts_pred = ml.predict_ts(ts_origin=df_nora3,var_origin=var_origin,ts_train=df_nora3.loc[start_training:end_training],var_train=var_train, model=model)
# Plotting a month of data:
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12, 6),gridspec_kw={'top': 0.95,'bottom': 0.150,'left': 0.05,'right': 0.990,'hspace': 0.2,'wspace': 0.2})
plt.title('Model: '+model+',Training Variables: '+','.join(var_origin))
#plt.plot(df_nora3['hs_a'].loc['2019-01-01':'2019-01-15'],'o',label='NORA3_a')
#plt.plot(df_nora3['hs_c'].loc['2019-01-01':'2019-01-15'],'o',label='NORA3_c')
plt.plot(df_nora3['hs_C_b'].loc['2019-01-01':'2019-01-15'],'o',label='NORAC_b')
plt.plot(ts_pred.loc['2019-01-01':'2019-01-15'],'x',label='NORA3_pred_b')
plt.ylabel('Hs[m]',fontsize=20)
#plt.plot(df_nora3['hs_b'].loc['2017-12-30':'2018-01-30'].asfreq('h'),'.',label='NORA3_b')
plt.grid()
plt.legend()
plt.savefig(model+'-'+'_'.join(var_origin)+'ts.png')
plt.show()

#plt.close()



#Plot all the data:
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12, 6),gridspec_kw={'top': 0.95,'bottom': 0.150,'left': 0.05,'right': 0.990,'hspace': 0.2,'wspace': 0.2})
plt.title('Model: '+model+',Training Variables: '+','.join(var_origin))
plt.plot(df_nora3['hs_a'].loc['2019-01-01':'2019-01-15'],'o',label='NORA3_a')
plt.plot(df_nora3['hs_c'].loc['2019-01-01':'2019-01-15'],'o',label='NORA3_c')
plt.plot(ts_pred.loc['2019-01-01':'2019-01-15'],'x',label='NORA3_b_pred')
plt.ylabel('Hs[m]',fontsize=20)
plt.plot(df_nora3['hs_C_b'].loc['2019-01-01':'2019-01-15'],'.',label='NORAC_b')
plt.grid()
plt.legend()
plt.savefig(model+'-'+'_'.join(var_origin)+'ts_all.png')
plt.show()
#plt.close()

#breakpoint()

# Scatter plot and metrics:
plt.scatter(df_nora3['hs_C_b'].loc[start_valid:end_valid], ts_pred.loc[start_valid:end_valid], color='black')
plt.title('scatter:'+model+'-'+'_'.join(var_origin))
plt.text(0, 1.0,'ΜΑΕ:'+str(np.round(sklearn.metrics.mean_absolute_error(df_nora3['hs_C_b'].loc[start_valid:end_valid], ts_pred.loc[start_valid:end_valid]),3)))
plt.text(0, 0.8,'$R²$:'+str(np.round(sklearn.metrics.r2_score(df_nora3['hs_C_b'].loc[start_valid:end_valid], ts_pred.loc[start_valid:end_valid]),3)))
plt.text(0, 0.6,'RMSE:'+str(np.round(sklearn.metrics.mean_squared_error(df_nora3['hs_C_b'].loc[start_valid:end_valid], ts_pred.loc[start_valid:end_valid])**0.5,3)))
plt.xlabel('Hs from NORAC')
plt.ylabel('Hs from NORAC_pred')
plt.savefig(model+'-'+'_'.join(var_origin)+'scatter.png')
plt.show()
#plt.close()





