import streamlit as st
import streamlit_vertical_slider as svs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

for k, v in st.session_state.items():
    st.session_state[k] = v

st.title("In-canopy profiles")

# This code is the second displaying page of the application.
# On this page, the user will see vertical gradients of temperature/humidity/wind for the tree and the atmosphere.

m_index = st.session_state['profile_m']//st.session_state['grid_size']

st.write("### Atmospheric variables")

fig,ax =plt.subplots(1,4)
ax[0].plot(st.session_state['wind_map'][:,m_index],st.session_state['heights_grid'])
ax[1].plot(st.session_state['T_map'][:,m_index],st.session_state['heights_grid'])
ax[2].plot(st.session_state['q_map'][:,m_index],st.session_state['heights_grid'])
ax[3].plot(st.session_state['co2_map'][:,m_index],st.session_state['heights_grid'])
plt.gca().invert_yaxis()
ax[0].set_ylabel('Height')
ax[0].set_xlabel('Wind speed \n (m/s)')
ax[1].set_xlabel('Temperature \n (K)')
ax[2].set_xlabel('Humidity \n (kg/m²)')
ax[3].set_xlabel('CO2 \n (ppm)')
ax[1].tick_params(labelleft=False)
ax[2].tick_params(labelleft=False)
ax[3].tick_params(labelleft=False)
st.pyplot(plt.gcf())


st.write("### Tree and leaf variables")
fig,ax =plt.subplots(1,4)
ax[0].plot(st.session_state['lai_map'][:,m_index],st.session_state['heights_grid'])
ax[1].plot(st.session_state['Tleaf_map'][:,m_index],st.session_state['heights_grid'])
ax[2].plot(st.session_state['qleaf_map'][:,m_index],st.session_state['heights_grid'])
ax[3].plot(st.session_state['co2leaf_map'][:,m_index],st.session_state['heights_grid'])
plt.gca().invert_yaxis()
ax[0].set_ylabel('Height')
ax[0].set_xlabel('LAI \n (m²/m²)')
ax[1].set_xlabel('Temperature \n (K)')
ax[2].set_xlabel('Humidity \n (kg/m²)')
ax[3].set_xlabel('CO2 \n (ppm)')
ax[1].tick_params(labelleft=False)
ax[2].tick_params(labelleft=False)
ax[3].tick_params(labelleft=False)
st.pyplot(plt.gcf())


st.write("### Exchanges")