#%% Main script
import os
cwd = 'c:/School/SloanProject/ABM_Model/SloanABM'  # current folder: 'c:\\2021\\Sloan Project\\ABM_Model\\SloanABM'
os.chdir(cwd)
#%%
from random import sample
# from sqlite3 import DatabaseError
import numpy as np
import pandas as pd
# from scipy import interpolate
# import matplotlib.pyplot as plt
from Market_Class_0620 import Market, Merge
from Price_predict import hist_price
from ABM_function import ZoneInvest, Agent_Investment, aggregation_fuel,create_dir 
from ABM_function import KGE_stat, remove_brackets
from ABM_function import generating_ABM_samples

# import scipy.stats
from datetime import date
import matplotlib.pyplot as plt
#%%
# Month abbreviation, day and year	
REC_p = 20     # renewable energy credit price ($/MW)

today = date.today()
date_now = today.strftime("%b-%d")  # date string
print("Today's date:", today)
# Month abbreviation, day and year	
date_now = today.strftime("%b-%d")  # date strin
create_dir(os.path.join(cwd,"00 Results","Calibration", "Calibration_" + date_now ))
Result_dir = os.path.join(cwd,"00 Results","Calibration", "Calibration_" + date_now )

Data_dir = "c:/School/SloanProject/ABM_Model/Data/"
Demand_dir = Data_dir + "Hist_Demand/"
Price_dir = Data_dir + "Hist_Price/"
Cost_dir = Data_dir + "Cost/"
CF_dir = Data_dir + "CF/"
Retire_dir = Data_dir + "Retirement/"

p_file =  "electricity_price.csv"
d_file = "Total_Demand.csv"
cf_file = "historical_capacity_factor_ABM.csv"
ncp_file = "new_capacity_edited.csv"
ca_re_file = "Retirement_Total_capacity.csv"
G_hist_cost_file = "historical_cost_data.csv"
# Zone_invest_file = "annual_zonal_invest.csv"
#%% read data files
    #%% read market data
    # price, demand,capacity factor, 
    # EIA-860: 3_1_Generator_Y2020.xlsx, which includes three tabs - Operable, Proposed, Retired and Canceled  

w = Market(ncp_file)
CP = w.read_data()    # new capacity
y = Market(d_file)
Demand = y.read_data()     # demand [Year, Total Demand (MWh)]
Demand = Demand.set_index('Year(MWh)')  # set index = 'Year' column

z = Market(cf_file)
CF = z.read_data()    # capacity factor (%)
za = Market(ca_re_file)  
CA = za.read_data()   # total capacity data
CA = CA.set_index('Year(MW)') # set index = 'Year' column

G_cost = Market(G_hist_cost_file)  
G_hist_cost = G_cost.read_data() # total cost data
# Zone_invest = Market(Zone_invest_file)  
# zone_invest_perc = Zone_invest.read_data() # investment in percentage of the capacity deficit based on demand growth
# zone_invest_perc = zone_invest_perc.set_index('Year')


#%% generate independent random ramples - consider Gibb sampling to generate correlated samples
# Capacity factor - long-term climate variability
# NG and Renewable are negatively correlated 
# NG CP will saturate at certain point
# Soalr's and Wind's CPs may increase 
mu_CF = CF.iloc[3:,1:4].mean()        # the mean capacity factor in the last 10 years
var_CF = CF.iloc[3:,1:4].var(axis=0)  # variance of the capacity factor in the last 10 years
G_name =['NG','Wind','Solar'] # generation types

# Cost, life span, and capacity factor 

Cost_NG = 1.510*1E6*1.1 # $917 in 2012 value = 1,510 in 2020 value (5% interest rate)
Cost_s = G_hist_cost['Solar'][0] # 2000 solar price
Cost_w = G_hist_cost['Wind'][0]  # 2000 wind price
# NG generator cost ($/MW) - capital cost is about 5*1e5 $/MW, O and M cost is assume 10% of the capital cost

# Cost_s = 1.34*1E6 # wind generator ($/MW), Source: Land-Based Wind Market Report: 2021 Edition
# Cost_w = 1.46*1E6 # wind generator ($/MW), Source: Utility-Scale Solar, 2021 Edition

life_span = {'NG':30,'Wind':30, 'Solar':25}  # NG, solar, wind (Ziegler et. al, 2018)
cost_t = {'NG':Cost_NG, 'Wind':Cost_w, 'Solar':Cost_s}  # initial installation cost

# costs are assumed random walk with a decreasing trend for solar and wind
# NG cost is assumed to increase. The assumptions are made in the Cost_Forecast function
new_CF = {'NG':56.6, 'Wind': 35.4, 'Solar':24.9}             # 2019 capacity factor
data = {'NG':[life_span['NG'],Cost_NG, new_CF['NG']],'Wind':[life_span['Wind'],Cost_w, new_CF['Wind']],
'Solar':[life_span['Solar'],Cost_s, new_CF['Solar']]}
df_G = pd.DataFrame.from_dict(data) # data for generators
df_G.index = ['LS','Cost','CF']     # add indices

#%% Agent-based model 
# If IRR > 0.1 - > invest, otherwise don't invest
# Invest = Normal (mu,sigma) of existing capacity 
# Investment Threshold 10% return 
# Generally speaking, a typical solar system in the U.S. can produce 
# electricity at the cost of $0.06 to $0.08 per kilowatt-hour. 
# a payback period of 5.5 years in New Jersey, Massachusetts, and California,
#  your ROI is 18.2% per year. In some other states, the payoff period can be 
#  up to 12 years, which translates to an 8.5% annual return. 
# G_index: index of the generation tech. 0:NG, 1:Solar, 2: Wind
# calcualte NPV, Pay-back-period (PBP), and IRR

n_agt = 161 # number of agents; 74/161 are wind/solar, 8 solar only, which makes 82/161 ~= 50%
sample_agt = np.random.gamma(shape=1.3, scale= 3, size = n_agt)  
agt_size_dist = np.sort(sample_agt/sum(sample_agt))  # agent's ranked captial distribution in the market
LZ = ['LZ_AEN', 'LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST'] 

# hist, bin_edges = np.histogram(agt_size_dist, density=True) # hist: counts, bin_edges: bins
# print(hist) 
# print(bin_edges)
# print(np.sum(hist * np.diff(bin_edges)))
# costs are assumed random walk with a decreasing trend for solar and wind
# NG cost is assumed to increase. The assumptions are made in the Cost_Forecast function
IRR_threshold = 0.06
dict = {'NG': [Cost_NG],'Solar':[Cost_s],'Wind':[Cost_w]}
cost_t = [Cost_NG, Cost_w, Cost_s]
df_cost = G_hist_cost
hist_new_capacity = CP.loc[11:]   # historical capacity installed (NG, Solar, and Wind)
renewable = int(0.5*n_agt) # number of the renewalbe companies 

                                              
# Market: Load Zones
LZ = ['LZ_AEN', 'LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST'] 
d_percent = [0.04,0.06,0.27,0.33,0.13,0.12]    # the percentage to total demand based on 2020 data
D_zone_percent = pd.DataFrame(d_percent, index = LZ, columns = ['D_perc']) # demand percentage dataframe

cap_hist = CP.iloc[12:,:]  # historical new and total capacity 
cap_retired = CA.iloc[13:,1:]  # the capacity retired from 2012 - 2020 and the total capacity

# initialization
R = []  # Pearson Coefficient of Correllation
KGE= [] # Kling-Gupta efficiency scores (KGE)
P = []  # P-value 

for k in range(10):  # explore the incentive amount that can reproduce historical data
    # stochasticity of the system
                                       
    random_cost_dict, agt_size_risk_dict = generating_ABM_samples(renewable, n_agt, date_now, k)
    ng_cost_dist = random_cost_dict["NG"]
    wind_cost_dist = random_cost_dict["wind"]
    solar_cost_dist = random_cost_dict["solar"]
    agt_risk_f = agt_size_risk_dict["risk"]
    agt_size_dist = agt_size_risk_dict["size"]
    rec_dist = agt_size_risk_dict["Rec_dist"]

    ## historical investment simulation
    agt_rec = REC_p*rec_dist   # the financial incentive the agents received
    frame = []  # investment of the simuation period
    row_list = []  # investment records
    tot_ca = CA.loc[2011][2]  # 2011 total planned generation capacity (MW)
    tot_new_ca = []  # new total capacity installed each year

    for t in range(9): # only zonal demand data only available from 2011 to 2020

        y = 2012 + t # year
        new_D = Demand.loc[y].values # new demand
       # demand_increase = 0.05  # annual demand increase rate
        # new_D = D*(1+ demand_increase)
        new_cost = df_cost.iloc[:,1:][df_cost["Year($MW)"] == y]          # new cost estimates 

        # linear function Totoal capacity = total demand*0.0003 - 12410
        Capacity_prediction = new_D *0.00029 + 5638  # capacity prediction using demand forecast data (actually historical demand)
        capacity_deficit = Capacity_prediction - tot_ca + CA.loc[y][1]  # the agent's perception about capacity deficit

        if capacity_deficit < 0:  # if no demand for capacity, no investment
            capacity_deficit = 0

        # Agent's Supply Curve: linear model with an error term
        # from Price_predict import future_price
        # years = np.arange(y,y+1)  # historical data for generating the supply curves
        supply_curves = hist_price(y)   # the supply curves [x_coeff, interception] of the load zone markets

        capacity_deficit_agt = capacity_deficit*agt_size_dist # the quantity agents need to invest
        new_capacity = 0  # new capacity 
        for agt_i in range(n_agt):
            # invest_t = []
            IRR_t = []     # create a list
            G_tech_lz = [] # create a list of tech. to invest in the load zones  
            tech_row = {'Year':y, 'Agt_I':str(agt_i)}  # a row that records technology invested in LZ
            new_cost_dict = {"NG":new_cost["NG"].values[0]*ng_cost_dist[agt_i], "Wind":new_cost["Wind"].values[0]*wind_cost_dist[agt_i], 
            "Solar":new_cost["Solar"].values[0]*solar_cost_dist[agt_i]}
            df_G.loc['Cost'] = new_cost_dict  # update cost
    
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

            frame.append(lz_invest)
        
        tot_ca = tot_ca + new_capacity - cap_retired.loc[y][0] # add new capacity and sbtract retirement
        tot_new_ca.append(tot_ca)  # a list of new capacity installed for the simulation period
        # consider adding randomness to capacity deficit to represent human decision uncertainty
    
    # result_irr = pd.DataFrame(row_list)
    # result_irr_file = os.path.join(cwd,"00 Results","Calibration", "ABM_Investment_records_0410_"+str(k)+".xlsx")
    # result_irr.to_excel(result_irr_file)

    result = pd.DataFrame(frame)
    # result.index = range(2012,2021)
    # result_file = os.path.join(cwd,"00 Results","Calibration", "ABM_Investment_0410_"+str(k)+".xlsx")
    # result.to_excel(result_file)

    # organize investment by agent and fuel type
    fuel_agt, fuel_yr = aggregation_fuel(n_agt, result)  # fuel_agt: agents' investment in fuel types
                                                         # fuel_yr: total investment in fuel types from 2012-2020
    fuel_agt_file = os.path.join(Result_dir, "ABM_fuel_agt_"+str(k)+".xlsx")
    fuel_agt.to_excel(fuel_agt_file)
    fuel_yr_file = os.path.join(Result_dir, "ABM_fuel_agt_"+str(k)+".xlsx")
    fuel_yr.to_excel(fuel_yr_file)
    fuel_cumulative = fuel_yr.sum()   # cumulative investment in MW at the end of 2020

    kge, p = KGE_stat(fuel_yr,tot_new_ca, cap_hist) # calculate Pearson Correlation Coefficient
    kge['solar_cost_STD'] = 0.1 + 0.02*k  # added the STD_wind to the dictionary
    kge['Wind_c'] = fuel_cumulative['Wind']
    kge['Solar_c'] = fuel_cumulative['Solar']
    KGE.append(kge)  # append the result to a list

df_KGE = pd.DataFrame.from_dict(KGE)
df_KGE_file = os.path.join(Result_dir, "KGE_" + date_now + ".xlsx")
df_KGE.to_excel(df_KGE_file)



# %%
