import pandas as pd
import csv

minute_array = []
with open('new1.csv', 'rb') as f:
    reader = csv.reader(f)
    my_list = list(reader)

for line in my_list:
    minute = int(line[0][11:13])*60+int(line[0][14:16])
    minute_array.append(minute)
print minute_array

with open('minute.csv', 'wb') as myfile:
    wr = csv.writer(myfile)
    for i in minute_array:
        wr.writerow([i,''])





#train = pd.Series(data = data_array, index = minute_array)
#train.columns = ['time','energy','gilman1',	'gilman2','hopkins1','hopkins2',	'scholars1','scholars2','voigt1','voigt2',	'voigt3',	'voigt4'
#]
#minute_array.to_csv('minute.csv')
