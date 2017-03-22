import pandas as pd
import csv
import matplotlib as pl


def parser(x):
    return pd.to_datetime(x)

series = pd.read_csv('new_data/EBU3B_Main_A_mod.csv', header=0, parse_dates=[0],index_col = 0, squeeze = True)

series.plot()
pl.pyplot.show()



'''
with open ('data/EBU3B_Main_A.csv', 'rb') as energy1:

    

    
    reader = csv.reader(energy1, delimiter=',')
    data = []
    time = []
    i = 0
    for row in reader:
        row[0] = row[0].split('+')[0]
        
        row[0] = row[0][:10]+' '+row[0][11:]
        
        time.append(''.join(row[0]))
        data.append(''.join(row[1]))
    #print data

    Dtimes = pd.to_datetime(time)
    df = pd.DataFrame(Dtimes)
    series = pd.Series( index = df,data = data)
    #print series

    series.resample('5T').sum()



        
        s = row[0].split(',')
        date = s[0].split('T')[0]
        print date
        remainder = s[0].split('T')[1]
        
        time = remainder[:-6]
#        print time

        times = time.split(':')
#        print times

        minute =int(times[0])*60+int(times[1])
        print minute
#        print row[1]
'''
