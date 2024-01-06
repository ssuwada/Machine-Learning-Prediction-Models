#   --------- FIRST CLASS EXERCISES -----------
#               Report task 1
#             till 11.10.23
#           ---------------------
#              SEBASTIAN SUWADA 
#           ---------------------
#

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

def create_df():
    df = pd.read_csv(r'/Users/sebastiansuwada/Desktop/Predictive Analytics/Naive/data.csv', sep=";", parse_dates=True, dayfirst=True)
    missing_values = df['load'].isna()  #save to list where is nan from load column
    missing_values_list = df[missing_values].index.to_list()   #show index numbers of position where is nan and save it to list

    for i in range(0,len(missing_values_list)):
        back = df.iloc[missing_values_list[-1]+1]['load']
        front = df.iloc[missing_values_list[i]-1]['load']
        ave = (front + back)/2
        df.loc[missing_values_list[i], 'load'] = ave


    # Convert the 'time' column to a datetime object
    # Set 'time' as the index
    df['time'] = pd.to_datetime(df['time'], format='%d.%m.%Y %H:%M')
    df.set_index('time', inplace=True)

    # Resample the data to hourly resolution and calculate the mean
    # The argument you pass to resample() is a string representing the new frequency 
    # or time interval you want for your data. Common frequency strings include 'D' 
    # for daily, 'H' for hourly, 'T' for minutes, and more. It can be also choosed for days or even a year.
    hourly_df = df.resample('H').mean()

    # Reset the index to have a regular datetime column
    hourly_df.reset_index(inplace=True)

    hourly_df.info()

    # last_index = df.index[-1]
    # df.loc[missing_values_list, 'load'] = 100
    # df['load'].fillna(0, inplace=True)  #replace every NaN as other defined in first argument 
    # print(df.iloc[8168]['load'])  #print value of this index
    # df.loc[8168, 'load'] = 100
    # print(df.iloc[8168]['load'])  #print value of this index
    # df.index = pd.to_datetime(df.index, format="%d.%m.%Y %H:%M")
    # d_range = pd.date_range(start='2022-01-01 00:00', end='2022-10-04 23:45', freq="15min")
    # df = df.reindex(d_range, fill_value="NAN")
    # df = df.reindex(pd.date_range(start='01.03.2022 00:00', end='01.04.2022 00:00', freq='D'), fill_value= pd.NA)
    # df.index = pd.DatetimeIndex(df.index)
    # df = df.reindex(pd.datarange("01.03.2022"), freq="15min", fill_value= pd.NA)
    file_path = '/Users/sebastiansuwada/Desktop/Predictive Analytics/Naive/sample_first_data.csv'
    #Save the DataFrame to Excel
    hourly_df.to_csv(file_path, index=True)  # Set index=False to exclude the index column in the Excel file
    #print(df)

    return hourly_df

def naive1(df1):

    df_naive1 = df1['load'].shift(168) #delete last seven days - thats create shift down
    # print(df_naive1.iloc[168])
    df_naive1.dropna(inplace=True)
    df_naive1.reset_index(drop=True, inplace=True)
    file_path = '/Users/sebastiansuwada/Desktop/Predictive Analytics/Naive/naive1.csv'
    
    #Save the DataFrame to Excel
    df_naive1.to_csv(file_path, index=True)  # Set index=False to exclude the index column in the Excel file

    

    return df_naive1

def naive2(df2):
# Convert the 'Timestamp' column to a datetime object
    #df['time'] = pd.to_datetime(df['time'])

# Extract the day of the week using .day_name()
    df2['Day_of_Week'] = df2['time'].dt.dayofweek

    # Create a new column 'Naive2'
    df2['Naive2'] = df2['load']

    # for index in range(168, len(df2)):
    #     if df2.loc[index, 'Day_of_Week'] == 0 or df2.loc[index, 'Day_of_Week'] == 5 or df2.loc[index, 'Day_of_Week'] == 6:
    #         df2.loc[index , 'Naive2'] = df2.loc[index - 168, 'load']
    #     else:
    #         df2.loc[index , 'Naive2'] = df2.loc[index - 24, 'load']

    df2['Naive_new'] = np.where(
        (df2['Day_of_Week'] == 0) | (df2['Day_of_Week'] == 5) | (df2['Day_of_Week'] == 6),
        df2['Naive2'].shift(168),
        df2['Naive2'].shift(24)
    )
    # Trzeba dropnac te z gory, a pozniej przezucic to na nowy dataframe bez czasu, zrobic nowy dataframe bez czasu dla kazdego tylko index i load i zrobic rmae i mae !!

    file_path = '/Users/sebastiansuwada/Desktop/Predictive Analytics/Naive/sample_data.csv'
    #Save the DataFrame to Excel
    df2.to_csv(file_path, index=True)  # Set index=False to exclude the index column in the Excel file
    #print(df)

    #df2.loc[:167, 'Naive2'] = np.nan

    naive2 = df2["Naive_new"]
    naive2.dropna(inplace=True)
    naive2.reset_index(drop=True, inplace=True)
    return naive2

def naive3(df3):

    df_naive3 = df3['load'].shift(24) #delete last seven days - thats create shift down
    df_naive3.dropna(inplace=True)
    df_naive3.reset_index(drop=True, inplace=True)

    return df_naive3

def calculate_mae(ndf1, ndf2, ndf3, df_):

    df_.loc[:167] = np.nan
    df_.dropna(inplace=True)
    df_.reset_index(drop=True, inplace=True)

    dataframes = [ndf1, ndf2, ndf3]

    # Find the length of the shortest DataFrame
    min_length = min(len(df_calc) for df_calc in dataframes)
    print(df_)

    for i, dfo in enumerate(dataframes):
        if len(dfo) > min_length:
            dfo.loc[:len(dfo) - min_length-1] = np.nan
            dfo.dropna(inplace=True)
            dfo.reset_index(drop=True, inplace=True)
        
        #print(dfo.iloc[0])

        # Calculate MAE

        mae = np.mean(np.abs(dfo - df_))
        print('MAE for '+str(i+1)+' naive forecast '+str(mae)+'\n')

        # Calculate RMSE

        rmse = np.sqrt(np.mean((dfo - df_)**2))
        print('RMSE for '+str(i+1)+' naive forecast '+str(rmse)+'\n')

# ------- MAIN --------


hourly_df = create_df()

hdf_ = hourly_df.copy()['load']

df_naive1 = naive1(hourly_df)
df_naive2 = naive2(hourly_df)
df_naive3 = naive3(hourly_df)

calculate_mae(df_naive1, df_naive2, df_naive3, hdf_)


# Specify the file path where you want to save the Excel file
# file_path = '/Users/sebastiansuwada/Desktop/Predictive Analytics/Naive/sample_data.csv'
# Save the DataFrame to Excel
# hourly_df.to_csv(file_path, index=True)  # Set index=False to exclude the index column in the Excel file

