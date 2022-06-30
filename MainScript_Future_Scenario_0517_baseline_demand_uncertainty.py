#%% Main script
import os
cwd = 'c:/School/SloanProject/ABM_Model/SloanABM'  # current folder: 'c:\\2021\\Sloan Project\\ABM_Model\\SloanABM'
os.chdir(cwd)

from random import sample
# from sqlite3 import DatabaseError
import numpy as np
import pandas as pd
# from scipy import interpolate
# import matplotlib.pyplot as plt
from Market_Class import Market,Read_Future_Data, Merge
from Price_predict import future_price
from ABM_function import ZoneInvest, Agent_Investment, aggregation_fuel_future,create_dir 
from ABM_function import  KGE_stat, generating_ABM_samples
# import scipy.stats
from datetime import date
#%%
# Future result directory 
future_dir = "Future0517_baseline_Uncertainty"  # the future results directory name

today = date.today()
print("Today's date:", today)
# Month abbreviation, day and year	
date_now = today.strftime("%b-%d")  # date strin

future_data_dir = "c:/School/SloanProject/ABM_Model/Data/Future_Data/"
Demand_dir = future_data_dir + "Demand/"
Price_dir = future_data_dir + "Price/"
Cost_dir = future_data_dir + "Cost/"
CF_dir = future_data_dir + "CF/"
Retire_dir = future_data_dir + "Retirement/"

#%% ############## read data files #############
    #%% read market data
    # cost, demand, capacity factor, price 
    # EIA-860: 3_1_Generator_Y2020.xlsx, which includes three tabs - Operable, Proposed, Retired and Canceled  
###### Demand (MWh)#####
d_file = "Future_annual_demand.pkl"
Annual_Demand = pd.read_pickle(os.path.join(Demand_dir, d_file))  # read annual demand prediction (2022- 2031)

Cost = pd.read_pickle(os.path.join(Cost_dir, "cost_no_change.pkl"))  # constant cost
CF = pd.read_pickle(os.path.join(CF_dir, "CF_no_chnage.pkl"))       # constant capacity factor 
Retire = pd.read_pickle(os.path.join(Retire_dir, "Retire_const.pkl"))  # constant retirement

#%% 
# Cost, life span, and capacity factor 
# wind generator ($/MW), Source: Land-Based Wind Market Report: 2021 Edition
# Solar generator ($/MW), Source: Utility-Scale Solar, 2021 Edition

life_span = {'NG':30,'Wind':30, 'Solar':25}  # NG, solar, wind (Ziegler et. al, 2018)
Cost_NG = Cost.iloc[0,0]; Cost_w = Cost.iloc[0,2]; Cost_s = Cost.iloc[0,1]
cost_t = {'NG':Cost_NG, 'Wind':Cost_w, 'Solar':Cost_s}  # initial installation cost

new_CF = {'NG':56.6, 'Wind': 35.4, 'Solar':24.9}             # 2020 capacity factor
data = {'NG':[life_span['NG'],Cost_NG, new_CF['NG']],'Wind':[life_span['Wind'],Cost_w, new_CF['Wind']],
'Solar':[life_span['Solar'],Cost_s, new_CF['Solar']]}
df_G = pd.DataFrame.from_dict(data) # data for generators
df_G.index = ['LS','Cost','CF']     # add indices

#%% Agent-based model 
# If IRR > 0.06 - > invest, otherwise don't invest
# Invest = Normal (mu,sigma) of existing capacity 
# Investment Threshold 6% return 
# Generally speaking, a typical solar system in the U.S. can produce 
# electricity at the cost of $0.06 to $0.08 per kilowatt-hour. 
# a payback period of 5.5 years in New Jersey, Massachusetts, and California,
#  your ROI is 18.2% per year. In some other states, the payoff period can be 
#  up to 12 years, which translates to an 8.5% annual return. 
# G_index: index of the generation tech. 0:NG, 1:Solar, 2: Wind
# calcualte NPV and IRR
T = 11 # simulation time - From 2021 to 2031
LZ = ['LZ_AEN', 'LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST'] 

n_agt = 161 # number of agents; 74/161 are wind/solar, 8 solar only, which makes 82/161 ~= 50%
df = pd.read_csv(os.path.join(future_data_dir, "agt_size_risk_dict_May-04_2.csv"))  # read the distributions of agents size, risk, preceived incentive (in terms of REC value)
# the numbers are the best results from calibration
agt_risk_f = np.array(df["risk"])    # agents are risk-adverse
agt_size_dist = np.array(df["size"]) # agent sizes
rec_dist = np.array(df["Rec_dist"])  # agents' perception about REC incentive
# small company recieved more financial incentive; the numbers are the % of REC received

df_cost_dist = pd.read_csv(os.path.join(future_data_dir, "Cost_dist_May-04_2.csv")) # read cost preception data
wind_cost_dist = df_cost_dist["wind"]   # wind cost adjustment factor distribution 
solar_cost_dist = df_cost_dist["solar"]   # solar cost adjustment factor distribution 
ng_cost_dist =  df_cost_dist["NG"]   # NG cost adjustment factor distribution 
                               
#%% Market: Load Zones
#  Future simulation
# The scenario folder

d_percent = [0.04,0.06,0.27,0.33,0.13,0.12]    # the percentage to total demand based on 2020 data
D_zone_percent = pd.DataFrame(d_percent, index = LZ, columns = ['D_perc']) # demand percentage dataframe
IRR_threshold = 0.06   # investment threshold
Demand = Annual_Demand['Energy (MWh)']  # future demand from 2021 to 2031
# cap_retired = pd.read_pickle(os.path.join(Retire_dir, "Retire_const.pkl"))   # the capacity retired from 2012 - 2020 and the total capacity

# initialization
frame = []  # investment of the simuation period
row_list = []  # investment records
tot_ca = 128947  # 2020 ERCOT total capacity (MW)
tot_new_ca = []  # new total capacity installed each year
REC_p = 15       # renewable energy credit price ($/MW)

S = 100  # number of simulation runs for uncertainty analysis
create_dir(os.path.join(cwd,"00 Results",future_dir))  # create a directory to stroe the simulation results
d_err = 0.05  # add a randomness (standard deviation of a random normal error) to the demand forecast 
df_future =pd.DataFrame(index = np.arange(2020,2032,1))  # total capacity of the simulations

for s in range(S):
    frame = []  # investment of the simuation period
    row_list = []  # investment records
    tot_ca = 128947  # 2020 ERCOT total capacity (MW)
    # tot_new_ca = []  # new total capacity installed each year

    f_tot =[128947]  # total capacity in future runs
    f_wind = [25123] # total wind capacity in future runs; Wind capacity in 2020: 25123 MW (new_capacity_edited.csv)
    f_solar = [3975] # total solar capacity in future runs; Solar capacity in 2020: 3975 MW
        
    for t in range(T): # only zonal demand data only available from 2021 to 2030
        agt_rec = REC_p*rec_dist   # the financial incentive the agents received

        y = 2021 + t # year
        new_D = Annual_Demand.loc[y]['Energy (MWh)']*np.random.normal(1, d_err) # received new annual demand info.
        # demand_increase = 0.05  # annual demand increase rate
        # new_D = D*(1+ demand_increase)
        new_cost = Cost.loc[y]          # new cost estimates 

        # linear function Totoal capacity = total demand*0.0003 - 12410
        # Capacity_prediction = new_D *0.00029 + 5638  # capacity prediction using demand forecast data (actually historical demand)
        Capacity_prediction = new_D *0.00032 + 5638  # capacity prediction using demand forecast data (actually historical demand)

        capacity_deficit = Capacity_prediction - (tot_ca - Retire.loc[y].values[0])  # the agent's perception about capacity deficit

        if capacity_deficit < 0:  # if no demand for capacity, no investment
            capacity_deficit = 0

        # Agent's Supply Curve: linear model with an error term
        # from Price_predict import future_price
        # years = np.arange(y,y+1)  # historical data for generating the supply curves (1 year only, 12 monthly data)
        supply_curves = future_price(y)   # the supply curves [x_coeff, interception] of the load zone markets

        # new_P = {} # create an empty distionary
        capacity_deficit_agt = capacity_deficit*agt_size_dist # the quantity agents need to invest
        new_capacity = 0  # new capacity 
        for agt_i in range(n_agt):
            # invest_t = []
            IRR_t = []     # create a list
            G_tech_lz = [] # create a list of tech. to invest in the load zones  
            tech_row = {'Year':y, 'Agt_I':str(agt_i)}  # a row that records technology invested in LZ
            new_cost_dict = {"NG":new_cost["NG"], "Wind":new_cost["Wind"]*wind_cost_dist[agt_i], 
            "Solar":new_cost["Solar"]*solar_cost_dist[agt_i]}
            df_G.loc['Cost'] = new_cost_dict  # update cost
            # print(df_G)

            for i in range(len(LZ)):
                c1,c2 = supply_curves[i]  # the slope and intersection
                c3 = 0                    # scaler for agent's prediction error   
                new_P = c1*new_D/12*D_zone_percent.loc[LZ[i]].values[0] + c2   # predicted average price of a load zone
                rec_p = agt_rec[agt_i]
                G_tech, max_IRR = ZoneInvest(new_P, rec_p, df_G)   # determine the tech. and the corrsponding IRR
                new_row = {'Year':y, 'Agt_I':str(agt_i), 'LZ':LZ[i],'Tech':G_tech, 'IRR':max_IRR} 
                tech_row[LZ[i]+"_fuel"] = G_tech  # add the feul type invested in the LZ to the tech_row dictionary
                IRR_t.append(max_IRR)  # a list of IRR at Zone LZ and time t
                row_list.append(new_row)
            IRR_t_array = np.array(IRR_t)  # convert to array
            agt_capacity_invest = capacity_deficit_agt[agt_i]*agt_risk_f[agt_i]  # the amount of investment is discounted because of risk aversion
            lz_invest = Agent_Investment(IRR_t_array,agt_capacity_invest,IRR_threshold)
            lz_invest_agt_sum = np.sum([*lz_invest.values()])  # sum the capacity invested in the Load Zones by an agent
            new_capacity += lz_invest_agt_sum  # add the agent's capacity invested to the total new capacity
            Merge(tech_row, lz_invest)  # merge the two dictionaries into "lz_invest" using the update function. 

            frame.append(lz_invest)  # new invetments
        
        tot_ca = tot_ca - Retire.loc[y][0] + new_capacity # add new capacity and sbtract retirement
        # tot_new_ca.append(tot_ca)  # a list of new capacity installed for the simulation period
        f_tot.append(tot_ca)  # a list of capacity installed for the simulation period
        # f_wind.append()
    df_future["run_"+str(s)] = f_tot


    # save results
    result = pd.DataFrame(frame)
    result_file = os.path.join(cwd,"00 Results",future_dir, "Result_0517_"+str(s)+".xlsx")
    result.to_excel(result_file)

    # organize investment by agent and fuel type
    fuel_agt, fuel_yr, LZ_yr = aggregation_fuel_future(n_agt, result, Retire)  # fuel_agt: agents' investment in fuel types
                                                                # fuel_yr: total investment in fuel types from 2012-2020
    fuel_agt_file = os.path.join(cwd,"00 Results",future_dir, "ABM_fuel_agt_0517_"+str(s)+".xlsx")
    fuel_agt.to_excel(fuel_agt_file)
    fuel_yr_file = os.path.join(cwd,"00 Results",future_dir, "ABM_fuel_year_0517_"+str(s)+".xlsx")
    fuel_yr.to_excel(fuel_yr_file)
    LZ_yr_file = os.path.join(cwd,"00 Results",future_dir, "ABM_fuel_LZ_0517_"+str(s)+".xlsx")
    LZ_yr.to_excel(LZ_yr_file)


## Figures
import matplotlib.pyplot as plt
plt.plot(df_future.index, df_future, linewidth=2.0)
years =  np.arange(2020,2032,1)
df_tot = pd.DataFrame(index = years)
df_wind = pd.DataFrame(index = years)
df_solar = pd.DataFrame(index = years)
df_NG = pd.DataFrame(index = years)
retire = list(Retire["Retirement"].values)  # a list of the capacity retirement time series
retire.insert(0,0)  # in 2000, the capacity is already the total capaity. In the other years, we need to subtract the retirement

for s in range(S):
    fuel_yr_file = os.path.join(cwd,"00 Results",future_dir, "ABM_fuel_year_0517_"+str(s)+".xlsx")
    df = pd.read_excel(fuel_yr_file, index_col = 0)  # read excel file, index column is year
    for i in range(len(df['NG'])):
        y = df['NG'].index[i]
        df.iloc[i,0] = df.iloc[i,0]- Retire.values[i]
        df.iloc[i,3] = df.iloc[i,3]- Retire.values[i]
    # df is a dataframe of the new capacity for the simulated period
    new_row =  [99849,25123,3975,128947]   # add the 2020 capacity to the dataframe
    new_df = pd.DataFrame([new_row], columns=df.columns)  # make the new row a dataframe so that it can be combined with df
    df = pd.concat([new_df, df], ignore_index = True) # the add the new row to the first row of the dataframe 
    # df['Year'] = np.arange(2021,2032,1)   # create a colume of "Year"
    # df.set_index("Year")


    df_all = df.cumsum()  # calculate the cumulative sum of the dataframe
    df_tot["run_"+str(s)] = df_all["Total"].values  # total capacity in future runs
    df_wind["run_"+str(s)] = df_all["Wind"].values  # wind capacity in future runs
    df_solar["run_"+str(s)] = df_all["Solar"].values  # solar capacity in future runs 
    df_NG["run_"+str(s)] = df_all["NG"].values # non-renewable capacity in future runs     
 #   df_NG["run_"+str(s)] = df_all["NG"].values  - np.cumsum(retire) # non-renewable capacity in future runs     

df_tot_file = os.path.join(cwd,"00 Results",future_dir, "df_tot_file.xlsx")
df_tot.to_excel(df_tot_file)
df_wind_file = os.path.join(cwd,"00 Results",future_dir, "df_wind.xlsx")
df_wind.to_excel(df_wind_file)
df_solar_file = os.path.join(cwd,"00 Results",future_dir, "df_solar.xlsx")
df_solar.to_excel(df_solar_file)
df_NG_file = os.path.join(cwd,"00 Results",future_dir, "df_NG.xlsx")
df_NG.to_excel(df_NG_file)
#%%
# text = ["total", "wind", "solar", "NG"]
# for i in text:
#     fig_tot_file = os.path.join(cwd,"00 Results",future_dir, "Futute_" + i +"_capacity.tif")
#     if i == "total":
#         df = df_tot
#     elif i == "wind":
#         df = df_wind
#     elif i == "solar":
#         df = df_solar
#     elif i == "NG":
#         df = df_NG
#     else:
#         "Text input is wrong. Please double check."
#     plt.plot(years, df, linewidth=2.0)
#     plt.xlabel("Year")
#     plt.ylabel("Capacity (MW)")
#     plt.savefig(fig_tot_file)
#     plt.close()








# %%
