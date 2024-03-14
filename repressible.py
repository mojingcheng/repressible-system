# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:25:31 2024

@author: 12095
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
plt.rcParams['axes.unicode_minus'] = False  # Fix display issue with negative signs

plt.close('all')

plt.rcdefaults()
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'normal'  # 'bold'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.labelweight'] = 'normal'  # 'bold'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.formatter.limits'] = (-2, 3)
plt.rcParams['figure.subplot.bottom'] = 0.17
plt.rcParams['figure.subplot.left'] = 0.12
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.fontsize'] = 10
plt.rcParams["legend.handletextpad"] = 0.3
plt.rcParams["legend.columnspacing"] = 0.5
plt.rcParams["legend.borderaxespad"] = 0
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})
plt.rcParams['lines.markersize'] = 5

some_initial_pm_level = 135

def repressible_model_odes(y, t, params, BL):
    synM, Krep, synP, degP, Kmat = params[:]

    mRNA, pr, pm = y[:]

    degM = 0.1386

    dmRNA = synM*(1-Krep*BL) - degM*mRNA
    dpr = synP*mRNA - Kmat*pr
    dpm = Kmat*pr - degP*pm
    dydt = [dmRNA, dpr, dpm]

    return dydt


def computeSSE(x):
    pm_m = np.zeros((len(tspan), len(BL)))

    for i, _ in enumerate(BL):
        sol = odeint(repressible_model_odes,y0, tspan, args=(x, BL[i])) 
        pm_m[:, i] = sol[:, 1]
    sse=0
    sse = np.sum((Fluo_timedata[:, 1:] - pm_m) ** 2)

    print(sse)

    return sse


if __name__ == "__main__":
    # Read experimental data
    Data = pd.read_csv("BLUE LIGHT Repressible System.csv")
    Header = Data.columns[1:3].tolist()
    Fluo_timedata = Data.iloc[:, 0:3].to_numpy()

    # Plot experimental data
    plt.figure(figsize=(6, 4))
    plt.plot(Fluo_timedata[:, 0], Fluo_timedata[:, 1:], '.')
    plt.legend(Header)
    

    y0 = [0, some_initial_pm_level]  
    tspan = np.linspace(0, 600, 6)
    BL = [1,0]

    
    bounds = [(1, 100), (0, 1), (0.01, 1), (0.010, 0.05),(0.01,1)]
    result_global = differential_evolution(computeSSE,bounds)
    
    param = result_global.x

    print(param)

    
    mRNA_ = np.zeros((len(tspan), len(BL)))
    pm_ = np.zeros((len(tspan), len(BL)))

    for i, _ in enumerate(BL):
        sol = odeint(repressible_model_odes, y0, tspan, args=(param, BL[i]))
        mRNA_[:,i] = sol[:,0]
        pm_[:, i] = sol[:, 1]

   
    plt.figure(figsize=(12, 4))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    axes2 = [plt.subplot(1, 2, i) for i in range(1, 3)]
    for i, a in enumerate(axes2):
        a.set_xlabel('Time (min)')
        
        if i == 0:
            a.plot(tspan, mRNA_)
            a.set_ylabel('mRNA (au)')
            
            
        elif i == 1:
            a.plot(Fluo_timedata[:,0], Fluo_timedata[:,1:], '.')
            a.set_prop_cycle(None) # reset the color order
            a.plot(tspan, pm_)
            a.set_ylabel('pm (au)')
            
        a.legend(Header)
    
    plt.tight_layout()
    
    plt.show()