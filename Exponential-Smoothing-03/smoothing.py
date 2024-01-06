#   --------- SECOND CLASS EXERCISES ----------
#                 Smoothing
#               till 11.10.23
#           ---------------------
#             SEBASTIAN SUWADA 
#           ---------------------
#

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


def import_data(filename):
    df = pd.read_csv(filename, sep=';')
    return df

def estimate_alfa(df, setName):

    # Initialize variables and lists
    alfa = []
    Mse = []

    # Create array of 100 elements of alfa -> from 0.01 to 0.99
    [alfa.append(x/100) for x in range(0,100)]
    Ytmp = np.zeros(len(df[setName]))

    # Formula
    # Mse = (Yf - Y)**2
    # Yt+1^ = alfa * Yt + (1-alfa)*Yt^
    # Yt - its df
    # Yt^ - is new array created based on df.

    for i in range(len(alfa)):
        
        # Initialize first value of array
        Ytmp[0] = df.iloc[0][setName]
        
        for j in range(1,len(df[setName])):

            # Based on formula calculate first part using data from df
            a = alfa[i]*(df.iloc[j-1][setName])
            # Based on formula calculate second part using previous value from new list Yt^ = (Ytmp)
            b = (1-alfa[i])* Ytmp[j-1] 
            Ytmp[j] = a + b 

        # MSE list append values 
        Mse.append(np.sqrt(np.mean((df[setName] - Ytmp)**2)))
        #print(str(alfa[i]*100)+'%')
    
    # print(Mse)
    #print(Mse, len(Mse), len(alfa))

    return Mse

def lowMSE(Mse):
    mseReturn = np.argmin(Mse)
    return mseReturn/100

def createForecast(dfF, setName, alfa):

    Ytmp = np.zeros(len(df[setName]))
    Ytmp[0] = df.iloc[0][setName]

    for j in range(1,len(df[setName])):
        # Based on formula calculate first part using data from df
        a = alfa*(df.iloc[j-1][setName])
        # Based on formula calculate second part using previous value from new list Yt^ = (Ytmp)
        b = (1-alfa)* Ytmp[j-1] 
        Ytmp[j] = a + b
    
    dfForecast = pd.DataFrame(Ytmp, columns=['price'])

    # file_path = '/Users/sebastiansuwada/Desktop/Predictive Analytics/Naive/forecast.csv'
    # #Save the DataFrame to Excel
    # dfForecast.to_csv(file_path, index=True)

    return dfForecast

def createForecastHourly(dfF, alfa):
    Ytmp = np.zeros(len(dfF))
    Ytmp[0] = dfF[0]

    for j in range(1,len(dfF)):
        # Based on formula calculate first part using data from df
        a = alfa*(dfF[j-1])
        # Based on formula calculate second part using previous value from new list Yt^ = (Ytmp)
        b = (1-alfa)* Ytmp[j-1] 
        Ytmp[j] = a + b

    
    
    return Ytmp

def averageDays(dfav, setName):

    temp = []

    for i in range(0,len(dfav[setName]), 24):
        temp.append(dfav.iloc[i:i + 24][setName].mean())

    df2_ = pd.DataFrame(temp, columns=['average'])

    return df2_

def maermseCalc(dfMAEinit, dfMAEforecast, setName):
    mae = np.zeros(len(dfMAEinit[setName]))
    rmse = np.zeros(len(dfMAEinit[setName]))

    for i in range(0,len(dfMAEinit[setName])):

        # Calculate MAE       
        mae[i] = np.mean(np.abs(dfMAEforecast.iloc[i][setName] - dfMAEinit.iloc[i][setName]))

        # Calculate RMSE
        rmse[i] = np.sqrt(np.mean((dfMAEforecast.iloc[i][setName] - dfMAEinit.iloc[i][setName])**2))
    
    # print(mae)
    # print(rmse)

    return mae, rmse

def hourlyCalc(dfMAEinit, dfMAEforecast, setName):

    maeTMP = []
    meanMAE = []
    rmseTMP = []
    meanrmse = []

    for j in range(0,23):
        for i in range(0,len(dfMAEinit[setName])-24,24):
            maeTMP.append(np.abs(dfMAEforecast.iloc[j+i][setName] - dfMAEinit.iloc[j+i][setName]))
            rmseTMP.append((dfMAEforecast.iloc[j+i][setName] - dfMAEinit.iloc[j+i][setName])**2)
        meanMAE.append(sum(maeTMP)/len(maeTMP))
        meanrmse.append(np.sqrt(sum(rmseTMP)/len(rmseTMP)))
        maeTMP.clear()
        rmseTMP.clear()
 
    print(meanMAE)
    print(meanrmse)

    return meanMAE, meanrmse

def estimateAlfa24H(df, setName):

    # Variables:
    hourTMP = []
    tempDF = pd.DataFrame()
    alphas = []

    # Create DataFrame for each hour at every day:
    for j in range(0,23):
        for i in range(0,len(df[setName])-24,24):
            hourTMP.append(df.iloc[j+i][setName])
        tempDF['hourVal'] = hourTMP
        hourTMP.clear()
        alphas.append(lowMSE(estimate_alfa(tempDF,'hourVal')))
        # Clear (reset) the DataFrame
        tempDF = pd.DataFrame()
        print('Alphas of hour '+str(j+1)+' Calculated and the value is '+str(alphas[j])+'')

    print(alphas)
    return alphas

def hourlyForecast(df, setName, alphas):

    # Variables
    hourTMP = []
    Ytmp = np.zeros(len(df))
    tmpList = []
    # Iterations for each hour
     
    for j in range(0,23):
        for i in range(0,len(df[setName])-24,24):
            hourTMP.append(df.iloc[j+i][setName])
        tmp = createForecastHourly(hourTMP,alphas[j])
        tmpList = tmp.tolist()
        hourTMP.clear()
        counter = 0
        for i in range(0,len(df[setName])-24,24):
            print("Already making positions for hour: "+str(j+1))
            Ytmp[i+j] = tmpList[counter]
            counter = counter + 1

        tmpList.clear()
    print(len(Ytmp))
    return Ytmp



# ------- === MAIN === --------


# === Exercise 1 ===

# Import data from data frame (csv) -> 
file_path = '/Users/sebastiansuwada/Desktop/Predictive Analytics/Naive/Algeria_Exports.csv'
df = import_data(file_path)

# Calculate/Estimate array of MSE and return alfas related to this values. 
Mse = estimate_alfa(df,'Exports')
alfaLowest = lowMSE(Mse)

print('Lowest alfa for this data set is: '+str(alfaLowest))

# === Exercise 2 === Exponential Smoothing ===

file_path = '/Users/sebastiansuwada/Desktop/Predictive Analytics/Naive/price_.csv'
df2 = import_data(file_path)

df2_360 = pd.DataFrame()
df2_360['price'] = df2['price'][:360*24]

file_path = '/Users/sebastiansuwada/Desktop/Predictive Analytics/Naive/first360.csv'
#Save the DataFrame to Excel
df2_360.to_csv(file_path, index=True)

# Calculations for alpha for every hour:
alphas = estimateAlfa24H(df2_360, 'price')



#Mse2 = estimate_alfa(df2_360, 'price')

# Find the smaller value from MSE array.
#alfaLowest2 = lowMSE(Mse2)
#print('Lowest alfa for this data set is: '+str(alfaLowest2))

# ======= CALUCLATE FORECAST FOR AVERAGE OF DAY =======

# Create new data frame consisting rest values on what will be based forecast
df2_rest_forecast = pd.DataFrame()
df2_rest_forecast['price'] = df2['price'][360*24:]
df2_rest_forecast.dropna(inplace=True)
df2_rest_forecast.reset_index(drop=True, inplace=True)

# Create DataFrame for forecast 
# forecastDF = createForecast(df2_rest_forecast, 'price', 0.99) # Musze miec 24 alpha na podstawie 360 dni
# daysForecast = averageDays(forecastDF,'price')

# Create DataFrame of average for initial data.
df2_rest_initial = pd.DataFrame()
df2_rest_initial['price'] = df2['price'][360*24:]   # Take next values after first year
df2_rest_initial.dropna(inplace=True)
df2_rest_initial.reset_index(drop=True, inplace=True)


# Hourly Forecast
hourlyFor = pd.DataFrame()
hourlyFor['price'] = hourlyForecast(df2_rest_initial,'price', alphas)
listMaeForcastHourly, listRmseForcastHourly  = hourlyCalc(df2_rest_initial,hourlyFor,'price')


# Plot MAE
plt.plot(listMaeForcastHourly, label='MAE', color='blue')

# Plot RMSE
plt.plot(listRmseForcastHourly, label='RMSE', color='red')

# Set titles and labels
plt.title('MAE and RMSE Over Hours')
plt.xlabel('Hour')
plt.ylabel('Error')
plt.legend()


plt.show()


# Create DataFrame for initial average values of days 
# daysInitial = averageDays(df2_rest_initial, 'price')






# Calculate MAE and RMSE for daily average values:

# aveMAE, aveRMSE = maermseCalc(daysInitial,daysForecast,'average')
# meanMAE = hourlyCalc(daysInitial,daysForecast,'average')

# plt.plot(meanMAE)
# plt.show()

# plt.plot(aveMAE)
# plt.title('MAE')
# plt.xlabel('day')
# plt.ylabel('MAE')

# plt.show()

# plt.plot(aveRMSE)
# plt.title('RMSE')
# plt.xlabel('day')
# plt.ylabel('RMSE')

# plt.show()

# aveMAE, aveRMSE = maermseCalc(df2_rest_initial,forecastDF,'price')

# plt.plot(aveMAE)
# plt.title('MAE')
# plt.xlabel('hour')
# plt.ylabel('MAE')

# plt.show()

# plt.plot(aveRMSE)
# plt.title('RMSE')
# plt.xlabel('hour')
# plt.ylabel('RMSE')

# plt.show()


