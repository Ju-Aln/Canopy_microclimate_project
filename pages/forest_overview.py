import streamlit as st
import streamlit_vertical_slider as svs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import constantes as cte
import homepage as h
import microclimate_model as m

st.title("Forest overview")

wind_map=h.grid.copy()
wind_map[:,:]=h.U_atm
T_map=h.grid.copy()
T_map[:,:]=h.T_atm
q_map=h.grid.copy()
q_map[:,:]=h.q_atm
co2_map=h.grid.copy()
co2_map[:,:]=h.co2_atm
Tleaf_map=np.zeros((h.n_grid,h.m_grid))
Tleaf_map[:,:]=20
k_eddy_ver = np.ones((h.n_grid,h.m_grid))/2
k_eddy_hor = np.zeros((h.n_grid,h.m_grid))
for i in range(h.n_grid-2,-1,-1):
    T_map[i,:] = T_map[i+1,:]-0.1

delta_t = 1
for i in range(1):
    H = cte.cp_air * cte.rho * (np.concatenate((np.diff(T_map,axis=0),np.array([[20-h.n_grid*0.1 for j in range(h.m_grid)]]))))*k_eddy_ver*h.delta_z
    wind_map,k_eddy_ver = m.solve_wind(wind_map,T_map,H)
    if h.prints:
        st.write('k_eddy_ver')
        st.write(k_eddy_ver)
        st.write('wind_map')
        st.write(wind_map)
    
    SW_map = m.solve_short_waves(h.SW_top)[0]
    LW_map = np.zeros((h.n_grid,h.m_grid))
    for j in range(h.m_grid):
        LW_veg = m.solve_long_wave(h.LW_top,j,Tleaf_map)
        for i in range(h.n_start_lai,h.n_stop_lai):
            LW_map[i,j] = LW_veg[i]
    if h.prints:
        st.write('LW_map')
        st.write(LW_map)
    Tleaf_map = delta_t/(cte.cp_veg*cte.rho_veg) * m.source_sink(Tleaf_map,SW_map,LW_map,T_map,q_map,wind_map)[0] + Tleaf_map
    if h.prints:
        st.write('Tleaf_map')
        st.write(Tleaf_map)
    qleaf_map = 0.622*10**(-5) * 0.611*10**3 * np.exp(17.27*(Tleaf_map)/(Tleaf_map-36 + 273))
    S_T = (Tleaf_map-T_map)/m.source_sink(Tleaf_map,SW_map,LW_map,T_map,q_map,wind_map)[1]*h.canopy_mask
    if h.prints:
        st.write('S_T')
        st.write(S_T)
        st.write('R_bl_heat')
        st.write(m.source_sink(Tleaf_map,SW_map,LW_map,T_map,q_map,wind_map)[1])
    S_q = (qleaf_map-q_map)/m.source_sink(Tleaf_map,SW_map,LW_map,T_map,q_map,wind_map)[2]*h.canopy_mask
    if h.prints:
        st.write('S_q')
        st.write(S_q)
        st.write('R_tot_wat')
        st.write(m.source_sink(Tleaf_map,SW_map,LW_map,T_map,q_map,wind_map)[2])
    T_map = delta_t * m.solve_transport(T_map,wind_map,k_eddy_ver,k_eddy_hor,S_T) + T_map
    if h.prints:
        st.write('Tatm')
        st.write(T_map)
    q_map = delta_t * m.solve_transport(q_map,wind_map,k_eddy_ver,k_eddy_hor,S_q) + q_map
    if h.prints:
        st.write('q_map')
        st.write(q_map)

st.write("# Results")  
st.write("### Wind outlook")
fig =plt.figure()
ax=fig.add_subplot(111)
cbar = ax.matshow(wind_map,cmap='bwr',vmin=0,vmax=h.U_atm)
plt.gca().invert_yaxis()
plt.ylabel('Height')
plt.xlabel('Canopy section')
fig.colorbar(cbar,shrink=0.7)
st.pyplot(plt.gcf())

st.write("### Temperature outlook")
fig =plt.figure()
ax=fig.add_subplot(111)
cbar = ax.matshow(T_map,cmap='bwr')
plt.gca().invert_yaxis()
plt.ylabel('Height')
plt.xlabel('Canopy section')
fig.colorbar(cbar,shrink=0.7)
st.pyplot(plt.gcf())

st.write("### Humidity outlook")
fig =plt.figure()
ax=fig.add_subplot(111)
cbar = ax.matshow(q_map,cmap='bwr')
plt.gca().invert_yaxis()
plt.ylabel('Height')
plt.xlabel('Canopy section')
fig.colorbar(cbar,shrink=0.7)
st.pyplot(plt.gcf())

st.write("### CO2 outlook")
fig =plt.figure()
ax=fig.add_subplot(111)
cbar = ax.matshow(co2_map,cmap='bwr')
plt.gca().invert_yaxis()
plt.ylabel('Height')
plt.xlabel('Canopy section')
fig.colorbar(cbar,shrink=0.7)
st.pyplot(plt.gcf())