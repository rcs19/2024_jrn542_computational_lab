"""
Create a stacked plot of chosen variables against PNBI (neutral beam injeciton power).  
"""
import pandas as pd
from matplotlib import pyplot as plt

# df = pd.read_csv('csv/2024-12-10/ncell_10-200.csv')  # Varying ccells Dataset generated by main.py
df = pd.read_csv('csv/2024-12-10/npart_1000-200000.csv')  # Varying npart Dataset generated by main.py

ydatas = ['runtime','noise_level','damp','omega',]          # list of y-variables to include in stacked plot.
# xdata = df['ncells']
xdata = df['npart']

# Create stacked plot with shared x-axis ('pnbi')
fig, ax = plt.subplots(len(ydatas), sharex=True)    

# Iterate through chosen y-variables and plot on a separate axis each
for i, var in enumerate(ydatas):
    ydata = df[var]
    ax[i].plot(xdata,ydata,label='Data')
    ax[i].set_ylabel(var)

    try:
        yerr = df[f'{var}_std']
        ax[i].fill_between(xdata,ydata-yerr,ydata+yerr, alpha=0.2, label='Error') # Plot error as shaded region 
    except:
        print("No error")
    if var=='damp':
        ax[i].axhline(-0.153, color = "black", ls="--", label="Analytical Result")
    elif var=='omega':
        ax[i].axhline(1.413, color = "black", ls="--", label="Analytical Result")

# ax[0].set_title('Varying Number of Cells')
# ax[-1].set_xlabel('Number of Cells')
ax[0].set_title('Parameters vs Number of Particles')
ax[-1].set_xlabel('Number of Particles')

plt.legend()
plt.subplots_adjust(hspace=0)
plt.show()