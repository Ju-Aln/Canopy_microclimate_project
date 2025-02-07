import streamlit as st
import streamlit_vertical_slider as svs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import microclimate_model as m
import constantes as cte

for k, v in st.session_state.items():
    st.session_state[k] = v


def interp1d(array: np.ndarray, new_len: int) -> np.ndarray:
    la = len(array)
    return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)

# This code is used in the main page of the application. 
# It displays the homepage where the user can fill in the tree and forest specifications along with the climate data.
# The microclimate model is then calculated at the end of the code to make the main variables availables for the 
# other pages of code which aims at displaying the main variables of the microclimate.


st.title("Inputs")

### Enter tree related specifications

# Set the title of the page 
st.write("### Tree specifications")

# Divide the page in 3 columns
col1,col2,col3=st.columns(3)
# Create a default value for the variable h_max_forest in the cache
if "h_max_forest" not in st.session_state: st.session_state.h_max_forest = 25.0
# The variable is then updated by the input the user enters
col1.number_input("Max canopy height (m)",value=25,key='h_max_forest')
if "h_min_forest" not in st.session_state: st.session_state.h_min_forest = 5.0
col2.number_input("Min canopy height (m)",value=5,key='h_min_forest')
if "max_lai" not in st.session_state: st.session_state.max_lai = 5.0
col3.number_input("LAI (m^2/m^2)",value=5,key='max_lai')

# The screen is here divided in 2 columns, the left one gathers the sliders which will control the sharing of the LAI in the tree,
# the right column displays the share of LAI according to height. 
col1,col2 = st.columns(2)
# Small spaces to place the sliders roughly in front of their corresponding heights in the graph.
col1.write("LAI shape")
col1.write("")
col1.write("")
col1.write("")
col1.write("")
col2.write("Resulting tree shape")
if "lai_share_90" not in st.session_state: st.session_state.lai_share_90 = 15.0
col1.slider("LAI share at 90% of tree height",0,100,15,key='lai_share_90')
if "lai_share_70" not in st.session_state: st.session_state.lai_share_70 = 45.0
col1.slider("LAI share at 70% of tree height",0,100,45,key='lai_share_70')
if "lai_share_50" not in st.session_state: st.session_state.lai_share_50 = 30.0
col1.slider("LAI share at 50% of tree height",0,100,30,key='lai_share_50')
if "lai_share_30" not in st.session_state: st.session_state.lai_share_30 = 15.0
col1.slider("LAI share at 30% of tree height",0,100,15,key='lai_share_30')
if "lai_share_10" not in st.session_state: st.session_state.lai_share_10 = 5.0
col1.slider("LAI share at 10% of tree height",0,100,5,key='lai_share_10')

# Calculates the heights of the 20 LAI layers (for the graph only)
st.session_state['heights'] = np.linspace(st.session_state['h_min_forest'],st.session_state['h_max_forest'],20)
# Calculates the shares of LAI in each LAI layer
lai_shares=np.array([0,st.session_state['lai_share_10'],st.session_state['lai_share_30'],st.session_state['lai_share_50'],st.session_state['lai_share_70'],st.session_state['lai_share_90'],0])
st.session_state['lai_shares_show'] = interp1d(lai_shares,20)

# Display the LAI sharing according to the sliders
fig =plt.figure(figsize=(20,47))
ax=fig.add_subplot(111)
ax.stackplot(st.session_state['lai_shares_show'], st.session_state['heights'],color="darkgreen", alpha=0.2, labels=['LAI'])
font=40
plt.ylabel('Height',fontsize = font)
plt.xlabel('Canopy section',fontsize = font)
ax.tick_params(axis='both', which='major', labelsize=font)
#fig.colorbar(cbar,shrink=0.7)
col2.pyplot(plt.gcf())





# This section aims at spreading the tree shape in the forest. The user shapes the horizontal LAI sharing roughly.
st.write("### Forest outlook")
col1,col2=st.columns(2)
# We set the atmospheric height at 3 times the one of the forest
h_max_atm = 3*st.session_state['h_max_forest']#col1.number_input("Max atm height (m - min 2 times forest height)",value=50)
# Horizontal length of the forest
if "L_forest" not in st.session_state: st.session_state.L_forest = 100
col1.number_input("Forest width (m)",value=100,key='L_forest')
# The user sets the size of the resolution grid
if "grid_size" not in st.session_state: st.session_state.grid_size = 1
col2.number_input("Grid size (m)",value=1,key='grid_size')


### Calculate grids
# This parts aims at calculating the properties of the resolution grid that will be used by the model and the displays

## Global grid
# We set the maximum width of the grid as the width of the forest plus 2 times a tenth of the width
L_max = st.session_state['L_forest'] + 2*st.session_state['L_forest']//10
# Calculation on the indices 
st.session_state['n_grid'] = int(h_max_atm//st.session_state['grid_size'])
st.session_state['m_grid'] = int(L_max//st.session_state['grid_size'])
# Set up of the grid
st.session_state['grid']=np.zeros((st.session_state['n_grid'],st.session_state['m_grid']))

## LAI repartition
# Set up of the LAI map and the corresponding start and stop indices of the LAI crown
st.session_state['lai_map']=st.session_state['grid'].copy()
st.session_state['n_start_lai']=int(st.session_state['h_min_forest']//st.session_state['grid_size'])
st.session_state['n_stop_lai']=int(st.session_state['h_max_forest']//st.session_state['grid_size'])
st.session_state['m_start_forest'] = int(st.session_state['L_forest']//10//st.session_state['grid_size'])+1
st.session_state['m_stop_forest'] = int(L_max//st.session_state['grid_size']-st.session_state['L_forest']//10//st.session_state['grid_size'])


# The user here sets up where the LAI starts and stops to be at maximum.
col1,col2 = st.columns(2)
col1.number_input("Percentage of forest length at which the LAI starts to be maximum (%)",value=5,key='max_lai_start')
col2.number_input("Percentage of forest length at which the LAI stops to be maximum (%)",value=95,key='max_lai_stop')

# Here we calculate the shares of LAI horizontally (according to the 2 inputs above) and vertically (according to the tree shape determined via the sliders)
horiz_lai_shares = np.concatenate((np.linspace(0.1,1,st.session_state['max_lai_start']+1),np.ones((st.session_state['max_lai_stop']-st.session_state['max_lai_start']-2)),np.linspace(1,0.1,100-st.session_state['max_lai_stop']-1)))
horiz_lai_shares = interp1d(horiz_lai_shares,st.session_state['m_stop_forest']-st.session_state['m_start_forest'])
vert_lai_shares = interp1d(lai_shares,st.session_state['n_stop_lai']-st.session_state['n_start_lai'])

# Calculation of the resulting 2D LAI map
for i in range(st.session_state['n_start_lai'],st.session_state['n_stop_lai']):
    for j in range(st.session_state['m_start_forest'],st.session_state['m_stop_forest']):
        st.session_state['lai_map'][i,j]=st.session_state['max_lai']*horiz_lai_shares[j-st.session_state['m_start_forest']]*vert_lai_shares[i-st.session_state['n_start_lai']]/np.sum(vert_lai_shares[:])
# Vertical cumulative LAI (for model purposes)
st.session_state['cumul_lai']=np.concatenate((np.cumsum(st.session_state['lai_map'],axis=0),[st.session_state['lai_map'][0,:]]),axis=0)

# Display of the LAI map in the grid
fig =plt.figure()
ax=fig.add_subplot(111)
cbar = ax.matshow(st.session_state['lai_map'],cmap='Greens')
plt.gca().invert_yaxis()
plt.ylabel('Height (m)')
plt.xlabel('Canopy section (m)')
fig.colorbar(cbar,shrink=0.5, label='LAI (m²/m²)',location='top')
ax.tick_params(top=True, labeltop=False, bottom=True, labelbottom=True)
fig.canvas.draw()
labels = [0]+[int(item.get_text()) for item in ax.get_xticklabels()[1:]]
labels = [i*st.session_state['grid_size'] for i in labels]
ax.set_xticklabels(labels)
st.pyplot(plt.gcf())


# Here the user selects what vertical section of the forest he wants to study in the "Profile" page
col1,col2 = st.columns([0.15, 0.85])
if "profile_m" not in st.session_state: st.session_state.profile_m = L_max//2
col2.slider("Place the slider where you want to see the profiles in the profile page",
          0,L_max,L_max//2,key='profile_m')



# Set up of the climatic conditions by the user
st.write("### Climate specifications")
col1,col2,col3=st.columns(3)
if "T_atm" not in st.session_state: st.session_state.T_atm = 20
col1.number_input("Atmospheric temperature (°C)",value=20,key='T_atm')
if "U_atm" not in st.session_state: st.session_state.U_atm = 10
col2.number_input("Wind speed at the top of the column (m/s)",value=10,key='U_atm')
if "q_atm" not in st.session_state: st.session_state.q_atm = 0.1
col3.number_input("Atmospheric humidity (kg/m^3)",value=0.1,key='q_atm')
col4,col5,col6=st.columns(3)
if "co2_atm" not in st.session_state: st.session_state.co2_atm = 400
col4.number_input("Atmospheric CO2 concentration (ppm)",value=400,key='co2_atm')
if "SW_top" not in st.session_state: st.session_state.SW_top = 400
col5.number_input("Short-Wave radiations from the Sun (W/m²)",value=400,key='SW_top')
if "LW_top" not in st.session_state: st.session_state.LW_top = 200
col6.number_input("Long-Wave radiations from the Atmosphere (W/m²)",value=200,key='LW_top')

# Some additional variables useful for the model
st.session_state['canopy_mask'] = np.zeros((st.session_state['n_grid'],st.session_state['m_grid']))
st.session_state['canopy_mask'][0:st.session_state['n_stop_lai'],st.session_state['m_start_forest']:st.session_state['m_stop_forest']]=1
st.session_state['delta_z'] = st.session_state['grid_size']
st.session_state['heights_grid'] = np.array([i*st.session_state['grid_size'] for i in range(st.session_state['n_grid'])])


## Display prints
# For debug purposes only, enables to dispay the tables in the app
col1,col2 = st.columns([0.35,0.65])
col1.write("### Display prints ?")
if "prints" not in st.session_state: st.session_state.prints = False
col2.toggle("",key='prints')
col2.write("For debug purpose only")


### Model run
# Initialisation of the different output maps.
st.session_state['T_map']=st.session_state['grid'].copy()
st.session_state['T_map'][:,:]=st.session_state['T_atm']
st.session_state['wind_map']=st.session_state['grid'].copy()
st.session_state['wind_map'][:,:]=st.session_state['U_atm']
st.session_state['q_map']=st.session_state['grid'].copy()
st.session_state['q_map'][:,:]=st.session_state['q_atm']
st.session_state['co2_map']=st.session_state['grid'].copy()
st.session_state['co2_map'][:,:]=st.session_state['co2_atm']
st.session_state['co2leaf_map']=st.session_state['grid'].copy()
st.session_state['co2leaf_map'][:,:]=st.session_state['co2_atm']
st.session_state['Tleaf_map']=np.zeros((st.session_state['n_grid'],st.session_state['m_grid']))
st.session_state['Tleaf_map'][:,:]=20
st.session_state['k_eddy_ver'] = np.ones((st.session_state['n_grid'],st.session_state['m_grid']))/2
st.session_state['k_eddy_hor'] = np.zeros((st.session_state['n_grid'],st.session_state['m_grid']))
# Set up of an initial vertical gradient to avoid problems
for i in range(st.session_state['n_grid']-2,-1,-1):
    st.session_state['T_map'][i,:] = st.session_state['T_map'][i+1,:]-0.1

# Select time duration
delta_t = 1
# Solve the model
for i in range(1):
    # The calculation of the vertical sensible heat flux is mandatory to solve the Monin-Obukhov length calculation and the wind module resolution
    H = cte.cp_air * cte.rho * (np.concatenate((np.diff(st.session_state['T_map'],axis=0),np.array([[20-st.session_state['n_grid']*0.1 for j in range(st.session_state['m_grid'])]]))))*st.session_state['k_eddy_ver']*st.session_state['delta_z']
    # Solves the wind and turbulent transport
    st.session_state['wind_map'],st.session_state['k_eddy_ver'] = m.solve_wind(st.session_state['wind_map'],st.session_state['T_map'],H)
    if st.session_state['prints']:
        st.write('k_eddy_ver')
        st.write(st.session_state['k_eddy_ver'])
        st.write('wind_map')
        st.write(st.session_state['wind_map'])
    
    # Solves the short-wave radiations resolution
    SW_map = m.solve_short_waves(st.session_state['SW_top'])[0]
    LW_map = np.zeros((st.session_state['n_grid'],st.session_state['m_grid']))
    for j in range(st.session_state['m_grid']):
        # Solves the long-wave radiations resolution
        LW_veg = m.solve_long_wave(st.session_state['LW_top'],j,st.session_state['Tleaf_map'])
        for i in range(st.session_state['n_start_lai'],st.session_state['n_stop_lai']):
            LW_map[i,j] = LW_veg[i]
    if st.session_state['prints']:
        st.write('LW_map')
        st.write(LW_map)
    # Solves the leaf temperature calculations via the leaf energy budget
    st.session_state['Tleaf_map'] = delta_t/(cte.cp_veg*cte.rho_veg) * m.source_sink(st.session_state['Tleaf_map'],SW_map,LW_map,st.session_state['T_map'],st.session_state['q_map'],st.session_state['wind_map'])[0] + st.session_state['Tleaf_map']
    if st.session_state['prints']:
        st.write('Tleaf_map')
        st.write(st.session_state['Tleaf_map'])
    # Solves the leaf saturated humidity calculations 
    st.session_state['qleaf_map'] = 0.622*10**(-5) * 0.611*10**3 * np.exp(17.27*(st.session_state['Tleaf_map'])/(st.session_state['Tleaf_map']-36 + 273))
    # Calculates the source of temperature for the atmosphere
    S_T = (st.session_state['Tleaf_map']-st.session_state['T_map'])/m.source_sink(st.session_state['Tleaf_map'],SW_map,LW_map,st.session_state['T_map'],st.session_state['q_map'],st.session_state['wind_map'])[1]*st.session_state['canopy_mask'] 
    if st.session_state['prints']:
        st.write('S_T')
        st.write(S_T)
        st.write('R_bl_heat')
        st.write(m.source_sink(st.session_state['Tleaf_map'],SW_map,LW_map,st.session_state['T_map'],st.session_state['q_map'],st.session_state['wind_map'])[1])
    # Calculates the source of humidity for the atmosphere
    S_q = (st.session_state['qleaf_map']-st.session_state['q_map'])/m.source_sink(st.session_state['Tleaf_map'],SW_map,LW_map,st.session_state['T_map'],st.session_state['q_map'],st.session_state['wind_map'])[2]*st.session_state['canopy_mask']
    if st.session_state['prints']:
        st.write('S_q')
        st.write(S_q)
        st.write('R_tot_wat')
        st.write(m.source_sink(st.session_state['Tleaf_map'],SW_map,LW_map,st.session_state['T_map'],st.session_state['q_map'],st.session_state['wind_map'])[2])
    # Solves the transport of energy
    st.session_state['T_map'] = delta_t * m.solve_transport(st.session_state['T_map'],st.session_state['wind_map'],st.session_state['k_eddy_ver'],st.session_state['k_eddy_hor'],S_T) + st.session_state['T_map']
    if st.session_state['prints']:
        st.write('Tatm')
        st.write(st.session_state['T_map'])
    # Solves the transport of humidity
    st.session_state['q_map'] = delta_t * m.solve_transport(st.session_state['q_map'],st.session_state['wind_map'],st.session_state['k_eddy_ver'],st.session_state['k_eddy_hor'],S_q) + st.session_state['q_map']
    if st.session_state['prints']:
        st.write('q_map')
        st.write(st.session_state['q_map'])
