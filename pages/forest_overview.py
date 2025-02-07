import streamlit as st
import streamlit_vertical_slider as svs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import constantes as cte
import homepage as h
import microclimate_model as m

for k, v in st.session_state.items():
    st.session_state[k] = v

# This code is the main displaying page of the application.
# On this page, the user will see 2D representations of the temperature/humidity/wind.

st.title("Forest overview")
st.write("# Results")  
st.write("### Wind outlook")
fig =plt.figure()
ax=fig.add_subplot(111)
cbar = ax.matshow(st.session_state['wind_map'],cmap='bwr',vmin=0,vmax=st.session_state['U_atm'])
plt.gca().invert_yaxis()
plt.ylabel('Height')
plt.xlabel('Canopy section')
fig.colorbar(cbar,shrink=0.7)
st.pyplot(plt.gcf())

st.write("### Temperature outlook")
fig =plt.figure()
ax=fig.add_subplot(111)
cbar = ax.matshow(st.session_state['T_map'],cmap='bwr')
plt.gca().invert_yaxis()
plt.ylabel('Height')
plt.xlabel('Canopy section')
fig.colorbar(cbar,shrink=0.7)
st.pyplot(plt.gcf())

st.write("### Humidity outlook")
fig =plt.figure()
ax=fig.add_subplot(111)
cbar = ax.matshow(st.session_state['q_map'],cmap='bwr')
plt.gca().invert_yaxis()
plt.ylabel('Height')
plt.xlabel('Canopy section')
fig.colorbar(cbar,shrink=0.7)
st.pyplot(plt.gcf())

"""
st.write("### CO2 outlook")
fig =plt.figure()
ax=fig.add_subplot(111)
cbar = ax.matshow(st.session_state['co2_map'],cmap='bwr')
plt.gca().invert_yaxis()
plt.ylabel('Height')
plt.xlabel('Canopy section')
fig.colorbar(cbar,shrink=0.7)
st.pyplot(plt.gcf())
"""