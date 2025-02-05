import streamlit as st
import streamlit_vertical_slider as svs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

## Constants 
rho = 1.204 # kg/m3
cp_air = 1004.675 # J.kg-1.K-1
rho_veg = 997 # kg/m3 (water)
cp_veg = 4200 # J.kg-1.K-1 (water)
lambda_ev = 2.5 * 10**6 # J/kg 
kappa = 0.41
grav= 9.81 # m/s2

## Revoir LAI
## Debug diffusivités et nu_air
## Debug LW
## Debug grid_size (ne marche pas si = 0.5)
## Vérifier les T et T-273
## Re-test
