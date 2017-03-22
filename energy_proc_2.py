import pandas as pd
import matplotlib as pl

def parser(x):
    return pd.to_datetime(x)

series = pd.read_csv('new_data/EBU3B_Main_A_mod.csv', header=0, parse_dates=[0],index_col = 0, squeeze = True)
upsampled = series.resample('10T',how = 'mean')
#interpolated = upsampled.interpolate(method = 'mean')
#print (upsampled)

series2 = pd.read_csv('new_data/EBU3B_Main_B_mod.csv', header=0, parse_dates=[0],index_col = 0, squeeze = True)
upsampled2 = series2.resample('10T',how = 'mean')
sum = upsampled+upsampled2

#incorporate traffic data
gilman1 = pd.read_csv('new_data/gilman1_1_mod.csv', header=0, parse_dates=[0],index_col = 0, squeeze = True)

gilman1 = gilman1.resample('10T', how = 'mean')

gilman2 = pd.read_csv('new_data/gilman1_2_mod.csv', header=0, parse_dates=[0],index_col = 0, squeeze = True)
gilman2 = gilman2.resample('10T', how = 'mean')

hopkins1 = pd.read_csv('new_data/hopkins1_1_mod.csv', header=0, parse_dates=[0],index_col = 0, squeeze = True)
hopkins1 = hopkins1.resample('10T', how = 'mean')

hopkins2 = pd.read_csv('new_data/hopkins1_2_mod.csv', header=0, parse_dates=[0],index_col = 0, squeeze = True)
hopkins2 = hopkins2.resample('10T', how = 'mean')

scholars1 = pd.read_csv('new_data/scholars1_1_mod.csv', header=0, parse_dates=[0],index_col = 0, squeeze = True)
scholars1 = scholars1.resample('10T', how = 'mean')

scholars2 = pd.read_csv('new_data/scholars1_2_mod.csv', header=0, parse_dates=[0],index_col = 0, squeeze = True)
scholars2 = scholars2.resample('10T', how = 'mean')

voigt1 = pd.read_csv('new_data/voigt1_1_mod.csv', header=0, parse_dates=[0],index_col = 0, squeeze = True)
voigt1 = voigt1.resample('10T', how = 'mean')

voigt2 = pd.read_csv('new_data/voigt1_2_mod.csv', header=0, parse_dates=[0],index_col = 0, squeeze = True)
voigt2 = voigt2.resample('10T', how = 'mean')

voigt3 = pd.read_csv('new_data/voigt2_1_mod.csv', header=0, parse_dates=[0],index_col = 0, squeeze = True)
voigt3 = voigt3.resample('10T', how = 'mean')

voigt4 = pd.read_csv('new_data/voigt2_2_mod.csv', header=0, parse_dates=[0],index_col = 0, squeeze = True)
voigt4 = voigt4.resample('10T', how = 'mean')


frames = [sum, gilman1 ,gilman2 ,hopkins1,hopkins2,scholars1,scholars2,voigt1,voigt2,voigt3,voigt4]
result = pd.concat(frames,axis = 1,join_axes= [sum.index])



#print(sum)
#sum.plot()
#pl.pyplot.show()
#upsampled.add(upsampled2,fill_value = 0)
#print (upsampled)
#upsampled2.plot()
#print(sum.dropna())
#print(sum[647])
result.dropna().to_csv('new1.csv')
#pl.pyplot.show()
