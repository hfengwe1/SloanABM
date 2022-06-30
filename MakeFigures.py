# Make Figures
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

for i in range(3):
    filename =os.path.join(cwd, "00 Results", "cost_forecast_" + G_name[i]+".png") 
    plt.plot(df_cost[G_name[i]])
    plt.xlabel('T (Year)')
    plt.ylabel('Installation Cost ($)')
    plt.title(G_name[i])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

for i in range(3):
    filename =os.path.join(cwd, "00 Results", "cost_forecast_.png") 
    plt.plot(df_cost[G_name[i]])
plt.ylim([0, 1.8e6])
plt.xlabel('T (Year)')
plt.ylabel('Installation Cost ($)')
  #  plt.title(G_name[i])
plt.grid(True)
plt.tight_layout()
plt.legend(G_name)
plt.savefig(filename)
plt.close()

