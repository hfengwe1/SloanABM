# Generate Supply Curve and Price Prediction

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

Data_dir = "c:/School/SloanProject/ABM_Model/Data/"
# os.chdir(Data_dir)
Demand_dir = Data_dir + "Hist_Demand/"
Price_dir = Data_dir + "Hist_Price/"

# two zones (LCRA and RAYB) are scattered and small and are excluded from the discussion
# this function reads demand/price data from the range of "years" and output a dataframe
def combine_data(folder, years):
# years: the years for compiling monthly price and demand data 
    LZ = ['LZ_AEN', 'LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST' ]  # load zones
#    LZ = ['LZ_AEN', 'LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST', 'LZ_LCRA', 'LZ_RAYBN']  
    LZ.insert(0,'Year')  # also add Year to the output data
    appended = []
    if folder == Demand_dir:
        text = "demand_"
    elif folder == Price_dir:
        text = "price_"

    for y in years:
        file = folder + text + str(y) +'.csv'
        data = pd.read_csv(file, index_col= 0, thousands=',').T  # read data
        data.insert(0,'Year',y) # add year in the dataframe
        appended.append(data[LZ][0:12])

    appended_data = pd.concat(appended) # monthly price / demand
    return appended_data

# this function read one year data (monthly demand or price)
def read_one_year_data(text, years):
    # years: the years for compiling monthly price and demand data 
    LZ = ['LZ_AEN', 'LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST' ]  
#    LZ = ['LZ_AEN', 'LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST', 'LZ_LCRA', 'LZ_RAYBN']  
    if text == "demand":
        folder = Demand_dir

    elif text == "price":
        folder = Price_dir

    file = folder + text + "_" + str(years) +'.csv'
    data = pd.read_csv(file, index_col= 0, thousands=',').T  # read data
    data = data[LZ][0:12] # monthly price / demand
    data.insert(0,'Year',years) # add year in the dataframe
    return data

# this function uses monthly demand and supply data to generate a linear regression model ("linear"). 
def hist_price(years):
    LZ = ['LZ_AEN', 'LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST' ] 
# demand data 
    demand_data = read_one_year_data("demand",years)   
# price data
    price_data = read_one_year_data("price",years)


    linear = []  # the linear function: [coefficient, interception], price = a*demand + b
    for i in range(len(LZ)):
        lz = LZ[i]
        X = demand_data[lz].values
        x_train = X.reshape(-1, 1)
        y = price_data[lz].values
        # correct anomaly 
        for j in range(len(y)): # for the prices in y
            p = y[j]  # price in month j
            if p >= 50:  # the price is set to 0-50 $/MW
                y[j] = 50  # set the price to the upper bound $50/MWh
            elif p <= 0:
                y[j] = 0  # set negative price to $0/MWh
    
        reg = LinearRegression().fit(x_train,y)
        linear.append([reg.coef_[0],reg.intercept_])  # [slope, intersection] of the linear supply curves
    return linear


# this function read one year data (monthly demand or price)
# For convenience and future simulation, I created this function and the only difference between this function and previour one is 
# the folder paths.
def read_one_year_data_future(text, year):
    # years: the years for compiling monthly price and demand data 
    LZ = ['LZ_AEN', 'LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST' ]  
#    LZ = ['LZ_AEN', 'LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST', 'LZ_LCRA', 'LZ_RAYBN']  
    if text == "demand":
        folder = "C:/School/SloanProject/ABM_Model/Data/Future_Data/Demand/"

    elif text == "price":
        folder = "C:/School/SloanProject/ABM_Model/Data/Future_Data/Price/"

    file = folder + text + "_" + str(year) +'.pkl'
    data = pd.read_pickle(file).T  # read data
    data = data[LZ][0:12] # monthly price / demand
    data.insert(0,'Year',year) # add year in the dataframe
    return data

# This function uses monthly demand and supply data to generate a linear regression model ("linear"). 
# Again,this can be generalized and combined with "hist_price"
def future_price(year):  # we have demand and price data from 2011- 2021
    # is it for future or historical simulation 
    if year >= 2021:
        future = True
    else :
        future = False 
        print("The year is not in the future:", future)

    ## demand
    LZ = ['LZ_AEN', 'LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST' ] 
    # d_percent = [0.04,0.06,0.27,0.33,0.13,0.12]    # the percentage to total demand based on 2020 data
    # D_zone_percent = pd.DataFrame(d_percent, index = LZ, columns = ['D_perc']) # demand percentage dataframe
    demand_data = read_one_year_data_future("demand",year)   # demand data for th
    # file = os.path.join(Demand_dir,'demand_data_agg.csv' )
    # demand_data.to_csv(file)   # save data to csv
    ## price
    price_data = read_one_year_data_future("price",year)
    # file = os.path.join(Price_dir,'price_data_agg.csv' )
    # price_data.to_csv(file)    # save data to csv

    linear = []  # the linear function: [coefficient, interception], price = a*demand + b
    for i in range(len(LZ)):
        lz = LZ[i]
        X = demand_data[lz].values
        x_train = X.reshape(-1, 1)
        y = price_data[lz].values
        # correct anomaly 
        for j in range(len(y)): # for the prices in y
            p = y[j]  # price in month j
            if p >= 50:  # the price is set to 0-50 $/MW
                y[j] = 50  # set the price to the upper bound $50/MWh
            elif p <= 0:
                y[j] = 0  # set negative price to $0/MWh
    
        reg = LinearRegression().fit(x_train,y)
        linear.append([reg.coef_[0],reg.intercept_])  # [slope, intersection] of the linear supply curves
    return linear
