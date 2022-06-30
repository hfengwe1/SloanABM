#%% Main script
import os
from datetime import date
from random import sample
from turtle import ycor
import numpy as np
import pandas as pd
# from MainScript_Future_Scenario_0507 import LZ
from ABM_function import  Cost_F, Cost_Forecast
from Market_Class import Read_Future_Data, Merge
from Price_predict import read_one_year_data


today = date.today()
print("Today's date:", today)
# Month abbreviation, day and year	
date_now = today.strftime("%b-%d")  # date string

cwd = 'c:\\2021\\Sloan Project\\ABM_Model\\SloanABM'  # current folder: 'c:\\2021\\Sloan Project\\ABM_Model\\SloanABM'
os.chdir(cwd)
Demand_dir = "C:/2021/Sloan Project/ABM_Model/Data/Future_Data/Demand/"
Price_dir = "C:/2021/Sloan Project/ABM_Model/Data/Future_Data/Price/"
Cost_dir = "C:/2021/Sloan Project/ABM_Model/Data/Future_Data/Cost/"
CF_dir = "C:/2021/Sloan Project/ABM_Model/Data/Future_Data/CF/"
Retire_dir = "C:/2021/Sloan Project/ABM_Model/Data/Future_Data/Retirement/"
# load zones
LZ = ['LZ_AEN', 'LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST' ]  # load zones
T = 11 # data lenth is 11 years: 2021- 2031

#%%
###### Demand (MWh)#####
d_file = "Monthly_Demand_Forecast_2022_2031.csv"
y = Read_Future_Data(d_file)  # read .csv (comma separated)
Demand = y.read_data()     # demand [Year, Total Demand (MWh)]
Annual_Demand = Demand.groupby("Year").sum()  # sum of the monthly demands
# Annual_Demand = Annual_Demand.set_index('Year')  # set index = 'Year' column
Annual_Demand.loc[2021, "Energy (MWh)"] = 391579000  # MWh
Annual_Demand.to_csv(os.path.join(Demand_dir, "Future_annual_demand.csv"))
Annual_Demand.to_pickle(os.path.join(Demand_dir, "Future_annual_demand.pkl"))  # save to pickle format
# Annual_Demand = pd.read_pickle(os.path.join(Demand_dir, "Future_annual_demand.pkl"))  # read pickle file 
# reserve is 23.5% for summer 2022
for y in np.arange(2021,2032,1):
       text = "demand"
       data = read_one_year_data(text, 2020)  # read 2020 data
       data = data.iloc[:,1:]  # remove the "year" column
       demand_f = data / data.sum().sum()  # demand as fraction based on historical data
       demand_tot = Annual_Demand.loc[y, "Energy (MWh)"] * (1-0.04) # the read the demand prediction of year y 
                                                                    # 4% demand are consumed by two Load Zones (LCRA and RAYBN) 
                                                                    # that are excluded in the simulation
       newdf = demand_tot * demand_f  # distributted the total demand to the LZs and months
       df2 = newdf.T   # transpose the dataframe to match the format of the historical data.
       file = text + "_" + str(y)  # filename
       df2.to_csv(os.path.join(Demand_dir, file + ".csv"))
       df2.to_pickle(os.path.join(Demand_dir, file + ".pkl"))  # save to pickle format

###### Price ($/MWh)#####
# There is a decreasing trend in electricty price: y = -0.9264x +37.482
for y in np.arange(2011,2022,1):  # recycle historical data from 2011 to 2021 for the future 2021 - 2031
       if y < 2021:
              text = 'price'
              data = read_one_year_data(text, y)  # read 2011-2020 data
              price_reduction = 0.1
      
       elif y == 2021:
              text = 'price'
              data = read_one_year_data(text, 2020)  # read 2020 data              
              price_reduction = 0.2
       data = data.iloc[:,1:]*(1- price_reduction)  # remove the "year" column
       df2 = data.T  # transpose the dataframe to match the format of the historical data
       file = text + "_" + str(y+10)  # filename
       df2.to_csv(os.path.join(Price_dir, file + ".csv"))
       df2.to_pickle(os.path.join(Price_dir, file + ".pkl"))  # save to pickle format                


# df = pd.DataFrame(columns = LZ)
# price_mean = []
# for y in np.arange(2011,2021,1):
#        text = "price"
#        data = read_one_year_data(text, y)  # read 2020 data
#        data = data.iloc[:,1:]  # remove the "year" column
#        price_mean.append(data.mean().mean())  # distributted the total demand to the LZs and months       
#        price_lz_mean = data.mean()
#        df.loc[y] = price_lz_mean
# file = text + "_LZ_mean"  # filename
# df.to_csv(os.path.join(Price_dir, file + ".csv"))
# df.to_pickle(os.path.join(Price_dir, file + ".pkl"))  # save to pickle format
# df1 = pd.DataFrame(price_mean, index= np.arange(2011,2021,1), columns = ["Average_Price"])
# df1.to_csv(os.path.join(Price_dir, text + "_annul.csv"))
     

#%%
##### Installation Cost ($/MW) #####
G_hist_cost_file = "historical_cost_data.csv"
G_cost = Read_Future_Data(G_hist_cost_file)  
G_hist_cost = G_cost.read_data() # total cost data
G_cost_2020 = list(G_hist_cost.iloc[20,1:])  # get the cost in 2020 (the most recent cost information)
       ##### no further cost reduction
cost_no_change = []
for i in range(T):
    cost_no_change.append(G_cost_2020)
    
newdf = pd.DataFrame(cost_no_change, columns= G_hist_cost.columns[1:])
newdf['Year'] = np.arange(2021,2032,1)
df2 = newdf.set_index('Year')
df2.to_csv(os.path.join(Cost_dir, "cost_no_change.csv"))
df2.to_pickle(os.path.join(Cost_dir, "cost_no_change.pkl"))  # save to pickle format

       ##### 2% decrease in Solar and Wind 
reduction_rate = 0.98
df2["Wind"] = [G_cost_2020[2] * reduction_rate**(i+1) for i in range(T)]  # Wind cost
df2["Solar"] = [G_cost_2020[1] * reduction_rate**(i+1) for i in range(T)]  # Solar cost
df2.to_csv(os.path.join(Cost_dir, "cost_2pect_decrease.csv"))
df2.to_pickle(os.path.join(Cost_dir, "cost_2pect_decrease.pkl"))  # save to pickle format

#%%
###### Capacity Factor #####
cf_file = "historical_capacity_factor_ABM.csv"
z = Read_Future_Data(cf_file)
CF = z.read_data()    # capacity factor (%)
CF_2020 = list(CF.iloc[12,1:4])  # get the cost in 2020 (the most recent cost information)
mu_CF = CF.iloc[3:,1:4].mean()        # the mean capacity factor in the last 10 years
std_CF = CF.iloc[3:,1:4].std(axis=0)  # variance of the capacity factor in the last 10 years
G_name =['NG','Wind','Solar'] # generation types
       ##### no capacity facter improvement 
cf_no_change = []
for i in range(T):
    cf_no_change.append(CF_2020)

newdf = pd.DataFrame(cf_no_change, columns= CF.columns[1:4])
newdf['Year'] = np.arange(2021,2021+T,1)
df2 = newdf.set_index('Year')
df2.to_csv(os.path.join(CF_dir, "CF_no_chnage.csv"))
df2.to_pickle(os.path.join(CF_dir, "CF_no_chnage.pkl"))  # save to pickle format

       ##### 1% increase in Solar and Wind 
increase_rate = 1.01
newdf["Wind"] = [CF_2020[1] * increase_rate**(i+1) for i in range(T)]  # Wind cost
newdf["Solar"] = [CF_2020[2] * increase_rate**(i+1) for i in range(T)]  # Solar cost
df2 = newdf.set_index('Year')
df2.to_csv(os.path.join(CF_dir, "CF_1pect_increase.csv"))

df2.to_pickle(os.path.join(CF_dir, "CF_1pect_increase.pkl"))  # save to pickle format

#%% Capacity Retirement 
ca_re_file = "Retirement_Total_capacity.csv"
za = Read_Future_Data(ca_re_file)  
CA = za.read_data()   # total capacity data
CA = CA.set_index('Year(MW)') # set index = 'Year' column
CA['Retire(%)'] = CA.iloc[:,1]/CA.iloc[:,2]  # fractions of capacity retirement to the total capacity

mu_retire = CA["Retirement"][12:].mean()        # the mean capacity retirement for the last 10 years (2011 - 2020)                                             
var_retire = CA["Retirement"][12:].std(axis=0)  # variance of the capacity retirement for the last 10 years (2011 - 2020)
mu_retire_perct = CA["Retire(%)"][12:].mean()   # the mean retirement is about 1.5% of the total capacity (2011 - 2020)
       ##### constant capacity retirement 
retire_const = zip(np.arange(2021,2021+T,1),np.repeat(mu_retire, T)) # create a list of constant retirement
newdf = pd.DataFrame(data = retire_const, columns = ["Year", "Retirement"]) # create a dataframe for the retirement
df2 = newdf.set_index('Year')
df2.to_csv(os.path.join(Retire_dir, "Retire_const.csv"))
df2.to_pickle(os.path.join(Retire_dir, "Retire_const.pkl"))  # save to pickle format



# %%
