#   --------- FIRST CLASS EXERCISES -----------
#            Probabilistic forecast
#             
#           ---------------------
#              SEBASTIAN SUWADA 
#           ---------------------
#

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# /Users/sebastiansuwada/Desktop/Predictive Analytics/Naive_probabilistic_forecasts_List_3

def import_data(filename):
    df_ = pd.DataFrame()
    d = np.loadtxt('/Users/sebastiansuwada/Desktop/Predictive Analytics/Naive_probabilistic_forecasts_List_3/GEFCOM.txt')
    df = pd.DataFrame(d, columns=['YYYYMMDD', 'HH', 'Zonal price', 'System load', 'Zonal load', 'Day of the week'])
    df_['price'] = df['Zonal price']
    #df = pd.read_csv(filename, sep=';')
    return df_

def naive1(df, setName):

    # Thats create shift down one day
    df['naive1'] = df[setName].shift(24) 
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def calculateErrors(df, setName):
    coverage = 0.5
    coverages = [] 
    tempErr = []
    real = []
    onehourPrice = []
    onehourNaive = []
    onehourNaiveContinue = []

    df364 = pd.DataFrame()
    df_real = pd.DataFrame()
    print(df['naive1'][540*24:])
    df364['price'] = df[setName][:365*24]
    df364['naive1'] = df['naive1'][:365*24]
    df_real['naive2'] = df['naive1'][365*24:]
    df_real['priceInterval'] = df[setName][365*24:]
    df_real.reset_index(drop=True, inplace=True)

    # Create DataFrame for each hour at every day:

    for j in range(0,24):
        for i in range(0,len(df364),24):
            tempErr.append(df364['price'][i+j] - df364['naive1'][i+j])
            real.append(df_real['priceInterval'][i+j]) 
            onehourPrice.append(df364['price'][i+j])
            onehourNaive.append(df364['naive1'][i+j])
            onehourNaiveContinue.append(df_real['naive2'][i+j])

        error = [price - naive for price, naive in zip(onehourPrice, onehourNaive)]

        quants = np.quantile(error, [(1-coverage)/2, (1+coverage)/2])
        low_q1 = [ forecast + quants[0] for forecast in onehourNaiveContinue]
        up_q2 = [ forecast + quants[1] for forecast in onehourNaiveContinue]
        hits = []

        for day in range(len(low_q1)-1):
            if real[day] < up_q2[day] and real[day] > low_q1[day]:
                hits.append(1)
            else:
                hits.append(0)
        print(hits)
        coverages.append(np.mean(hits))

        tempErr.clear()
        real.clear()
        onehourPrice.clear()
        onehourNaive.clear()
        onehourNaiveContinue.clear()

        print('Calculating for hour '+str(j+1)+'/24')
    
    del df364

    plt.plot(coverages)
    plt.show()


def plotHourDay(df, setNameStart, setNameForecast, upQuantile, downQuantile, dayNumber):
    print(downQuantile)
    # downQuantile_abs = [abs(x) for x in downQuantile]
    valuesForecast = df[setNameForecast][dayNumber*24:(dayNumber+1)*24].tolist()
    valuesSimple = df[setNameStart][dayNumber*24:(dayNumber+1)*24].tolist()
    x = list(range(24))
    print(len(x))
    print(len(upQuantile))
    print(len(valuesForecast))

    # Create a scatter plot for the points
    plt.scatter(x, valuesForecast, color='red', label='Forecast points')
    plt.errorbar(x, valuesSimple, yerr=[downQuantile, upQuantile], fmt='o', color='black', label='Asymmetric Error Bars Based on real data')
    plt.legend()
    plt.show()

def empiricalCoverage(df, upQuantile, downQuantile, setNameStart, setNameForecast, startDay):
    tempdf = pd.DataFrame()
    #listStartValues = df[setNameStart][startDay*24:].tolist()
    tempdf[setNameStart] = df[setNameStart][startDay*24:]
    tempdf[setNameForecast] = df[setNameForecast][startDay*24:]
    tempdf.dropna(inplace=True)
    tempdf.reset_index(drop=True, inplace=True)
    print(tempdf)

    counter = []

    for j in range(0,23):
        for i in range(0,len(tempdf),24):
            if tempdf[setNameForecast][i+j] > tempdf[setNameStart][i+j]+upQuantile[j] or tempdf[setNameForecast][i+j] < tempdf[setNameStart][i+j] - downQuantile[j]:
                counter.append(0)
            else: 
                counter.append(1)

    print(counter)
    print(counter.count(1)/len(counter))


# ------- === MAIN === --------


# Import data from data frame (csv) -> 
file_path = '/Users/sebastiansuwada/Desktop/Predictive Analytics/Naive_probabilistic_forecasts_List_3/GEFCOM.csv'
df = import_data(file_path)
df = naive1(df, 'price')
calculateErrors(df, 'price')

# dayNumber the day that we want use the hours to check 
# plotHourDay(df, 'price', 'naive1', upQuantile, downQuantile, 1000)

# empiricalCoverage(df, upQuantile, downQuantile, 'price', 'naive1', 365)

