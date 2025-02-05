import streamlit as st
import streamlit_vertical_slider as svs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import microclimate_model as m
import constantes as cte

def interp1d(array: np.ndarray, new_len: int) -> np.ndarray:
    la = len(array)
    return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)



st.title("Inputs")

### Enter specifications
 
st.write("### Tree specifications")
col1,col2,col3=st.columns(3)
h_max_forest = col1.number_input("Max canopy height (m)",value=25)
h_min_forest = col2.number_input("Min canopy height (m)",value=5)
max_lai = col3.number_input("LAI (m^2/m^2)",value=5)

col1,col2 = st.columns(2)
col1.write("LAI shape")
col1.write("")
col1.write("")
col1.write("")
col1.write("")
col2.write("Resulting tree shape")
lai_share_90 = col1.slider("LAI share at 90% of tree height",0,100)
lai_share_70 = col1.slider("LAI share at 70% of tree height",0,100)
lai_share_50 = col1.slider("LAI share at 50% of tree height",0,100)
lai_share_30 = col1.slider("LAI share at 30% of tree height",0,100)
lai_share_10 = col1.slider("LAI share at 10% of tree height",0,100)

heights = np.linspace(h_min_forest,h_max_forest,20)
lai_shares=np.array([0,lai_share_10,lai_share_30,lai_share_50,lai_share_70,lai_share_90,0])
lai_shares_show = interp1d(lai_shares,20)

fig =plt.figure(figsize=(20,47))
ax=fig.add_subplot(111)
ax.stackplot(lai_shares_show, heights,color="darkgreen", alpha=0.2, labels=['LAI'])
font=40
plt.ylabel('Height',fontsize = font)
plt.xlabel('Canopy section',fontsize = font)
ax.tick_params(axis='both', which='major', labelsize=font)
#fig.colorbar(cbar,shrink=0.7)
col2.pyplot(plt.gcf())






st.write("### Forest outlook")
col1,col2=st.columns(2)
h_max_atm = 3*h_max_forest#col1.number_input("Max atm height (m - min 2 times forest height)",value=50)
L_forest = col1.number_input("Forest width (m)",value=100)
grid_size = col2.number_input("Grid size (m)",value=1)


### Calculate grids

## Global grid
L_max = L_forest + 2*L_forest//10
n_grid = int(h_max_atm//grid_size)
m_grid = int(L_max//grid_size)
grid=np.zeros((n_grid,m_grid))

## LAI repartition
lai_map=grid.copy()
n_start_lai=int(h_min_forest//grid_size)
n_stop_lai=int(h_max_forest//grid_size)
m_start_forest = int(L_forest//10//grid_size)+1
m_stop_forest = int(L_max//grid_size-L_forest//10//grid_size)


col1,col2 = st.columns(2)
max_lai_start = col1.number_input("Percentage of forest length at which the LAI starts to be maximum (%)",value=5)
max_lai_stop = col2.number_input("Percentage of forest length at which the LAI stops to be maximum (%)",value=95)

horiz_lai_shares = np.concatenate((np.linspace(0.1,1,max_lai_start+1),np.ones((max_lai_stop-max_lai_start-2)),np.linspace(1,0.1,100-max_lai_stop-1)))
horiz_lai_shares = interp1d(horiz_lai_shares,m_stop_forest-m_start_forest)
vert_lai_shares = interp1d(lai_shares,n_stop_lai-n_start_lai)

for i in range(n_start_lai,n_stop_lai):
    for j in range(m_start_forest,m_stop_forest):
        lai_map[i,j]=max_lai*horiz_lai_shares[j-m_start_forest]*vert_lai_shares[i-n_start_lai]/np.sum(vert_lai_shares[:])
cumul_lai=np.concatenate((np.cumsum(lai_map,axis=0),[lai_map[0,:]]),axis=0)
fig =plt.figure()
plt.matshow(lai_map,cmap='Greens')
plt.gca().invert_yaxis()
plt.ylabel('Height')
plt.xlabel('Canopy section')
st.pyplot(plt.gcf())



st.write("### Climate specifications")
col1,col2,col3=st.columns(3)
T_atm = col1.number_input("Atmospheric temperature (°C)",value=20)
U_atm = col2.number_input("Wind speed at the top of the column (m/s)",value=10)
q_atm = col3.number_input("Atmospheric humidity (kg/m^3)",value=0.1)
col4,col5,col6=st.columns(3)
co2_atm = col4.number_input("Atmospheric CO2 concentration (ppm)",value=400)
SW_top = col5.number_input("Short-Wave radiations from the Sun (W/m²)",value=400)
LW_top = col6.number_input("Long-Wave radiations from the Atmosphere (W/m²)",value=200)


canopy_mask = np.zeros((n_grid,m_grid))
canopy_mask[0:n_stop_lai,m_start_forest:m_stop_forest]=1
delta_z = grid_size
heights_grid = np.array([i*grid_size for i in range(n_grid)])


## Display prints

col1,col2 = st.columns([0.35,0.65])
col1.write("### Display prints ?")
prints = col2.toggle("")


