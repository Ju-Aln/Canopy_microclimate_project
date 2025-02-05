import streamlit as st
import streamlit_vertical_slider as svs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import constantes as cte
import homepage as h

def solve_wind(u,T,H):

    disp_h = np.zeros((1,h.m_grid))
    disp_h = 0.66 * h.h_max_forest*h.canopy_mask[0,:]
    # MO length
    MO_length = u.copy()


    ## Massman & Weil (1999)
    pm=1
    massman_params = [6.14,0.1,0.1,-0.1,0.06]
    
    reverse_cumul_lai=np.zeros((h.n_grid,h.m_grid))
    for i in range(h.n_grid-2,-1,-1):
        reverse_cumul_lai[i,:] = reverse_cumul_lai[i+1,:]+h.lai_map[i,:]

    cdrag = massman_params[0]**(-reverse_cumul_lai/massman_params[1]) + massman_params[2]**(-reverse_cumul_lai/massman_params[3])+massman_params[4]
    eta_z=u.copy()
    eta_z[0,:]=cdrag[0,:]*h.lai_map[0,:]
    for i in range(1,h.n_grid):
        eta_z[i,:] = cdrag[i,:]*h.lai_map[i,:]+eta_z[i-1,:]
    u_star = (u[h.n_stop_lai+1,:] * (0.32 - 0.264 * np.exp(-15.1*eta_z[h.n_stop_lai+1,:]))*h.canopy_mask[1,:]
              + u[1,:] * (0.32 - 0.264 * np.exp(-15.1*eta_z[1,:]))*(1-h.canopy_mask[1,:]))
    

    small_n = eta_z[h.n_stop_lai+1,:]*u[h.n_stop_lai+1,:]**2/(2*u_star[:]**2)     

    nu1 = (2.4**2+1.9**2+1.25**2)**(-0.5)
    nu3 = (2.4**2+1.9**2+1.25**2)**(3/2)
    nu2 = nu3/6 - 1.25**2/(2*nu1)

    lambda_k_eddy = np.sqrt(7/(3*0.05**2 * nu1 * nu3) + (1/3 - 1.25**2*nu1**2)/(3*0.05**2*nu1*nu2))

    B1 = -(9*u_star/u[h.n_stop_lai+1,:])/(2*0.05*nu1*(9/4 - lambda_k_eddy**2*u_star**4/u[h.n_stop_lai+1,:]**4))

    sigma1 = eta_z.copy()
    sigma2 = eta_z.copy()
    sigma3 = eta_z.copy()
    sigma_e_over_ustar = eta_z.copy()
    T_l = eta_z.copy()
    sigma_w = eta_z.copy()
    for i in range(h.n_stop_lai):
        sigma1[i,:] = -lambda_k_eddy * eta_z[h.n_stop_lai+1,:]*(1 - eta_z[i,:]/eta_z[h.n_stop_lai+1,:])
        sigma2[i,:] = -3 * small_n[:] * (1 - eta_z[i,:]/eta_z[h.n_stop_lai+1,:])
        sigma3[i,:] = sigma1[i,:]
        sigma_e_over_ustar[i,:] = (nu3 * np.exp(sigma1[i,:]) + B1 *(np.exp(sigma2[i,:])-np.exp(sigma3[i,:])))**(1/3)
        sigma_w[i,:] = 1.25 * nu1 * sigma_e_over_ustar[i,:] /u_star[:]

        T_l[i,:] = h.h_max_forest/u_star[:] * np.maximum(0.3,(0.4*(h.heights_grid[i]- disp_h[:])/(1.25*h.heights_grid[i])))

    #k_eddy=np.ones((n_grid,m_grid))  

    #H = -cp_air * rho * (np.concatenate((np.diff(T,axis=0),np.zeros((1,m_grid)))))*k_eddy*delta_z
    
    
    
    ## Harman & Finigan (2007)
    
    pot_temp = h.T_atm+273.15   
    for i in range(h.n_grid):
        MO_length[i,:]=-(cte.rho * cte.cp_air * u_star[:]**3 * pot_temp)/(cte.kappa * cte.grav * H[i,:])
 
    stab_param=u.copy()
    for i in range(h.n_grid):
        stab_param[i,:] = (h.heights_grid[i] - disp_h)/MO_length[i,:]
    ita_phi = np.where(stab_param<0, (1-16*stab_param)**(-1/4),(1+5*stab_param))
    z_star = 2*h.h_max_forest - disp_h
    c1=np.zeros((h.m_grid))
    small_phi=np.ones((h.n_grid,h.m_grid))
    c2=0.5
    for j in range(h.m_start_forest,h.m_stop_forest):
        c1[j]= (1-(small_n[j]/eta_z[h.n_stop_lai+1,j]*(cdrag[h.n_stop_lai+1,j]*reverse_cumul_lai[h.n_stop_lai+1,j]/pm)*u[h.n_stop_lai+1,j]*(cte.kappa*(h.h_max_forest-disp_h[j])/u_star[j]))/ita_phi[h.n_stop_lai+1,j])/np.exp(-c2*(h.h_max_forest-disp_h[j])/z_star[j])
    for i in range(h.n_grid):
        small_phi[i,:] = (1 - c1[:]*np.exp(-c2*(h.heights_grid[i]-disp_h[:])/z_star[:]))
 
    u[h.n_grid-1,:] = h.U_atm
    for i in range(h.n_grid-2,-1,-1):
        u[i,:]=u[i+1,:]-u_star[:]*h.delta_z/(cte.kappa*(h.heights_grid[i]-disp_h[:]))*small_phi[i,:]*ita_phi[i,:]

    for i in range(h.n_stop_lai,-1,-1):
        for j in range(h.m_start_forest,h.m_stop_forest):
            u[i,j] = u[i,j]*(1-h.canopy_mask[i,j]) + u[h.n_stop_lai+1,j]*np.exp(-small_n[j]*(1-eta_z[i,j]/eta_z[h.n_stop_lai+1,j]))*h.canopy_mask[i,j]

    k_eddy = np.zeros((h.n_grid,h.m_grid))

    k_eddy_top = sigma_w[h.n_stop_lai+1,:]**2*T_l[h.n_stop_lai+1,:]
    c2_k = 0.7
    c1_k = np.zeros((h.m_grid))

    for j in range(h.m_start_forest,h.m_stop_forest):
        c1_k[j] = 1-u_star[j]*0.41*(h.heights_grid[h.n_stop_lai+1]-disp_h[j])/(k_eddy_top[j]*ita_phi[h.n_stop_lai+1,j]**(-1/2))/np.exp(-c2_k*(h.heights_grid[h.n_stop_lai+1]-disp_h[j])/z_star[j])

    small_phi_c = np.ones((h.n_grid,h.m_grid))
    for i in range(h.n_grid):
        small_phi_c[i,:] = (1- c1_k[:] * np.exp(-c2_k *(h.heights_grid[i]-disp_h[:])/z_star[:]))

    for i in range(h.n_grid-1,-1 ,-1):
        k_eddy[i,:] = u_star[:] * cte.kappa * (h.heights_grid[i]-disp_h[:])/(ita_phi[i,:]**(-1/2)*small_phi_c[i,:])# *(1-canopy_mask[i,:])
    for i in range(h.n_stop_lai):
        for j in range(h.m_start_forest,h.m_stop_forest):
            k_eddy[i,j] = sigma_w[i,j]**2*T_l[i,j] 

    return(u,k_eddy)

def solve_transport(X,u,k_ver,k_hor,S):
    deriv_X=X.copy()
    for i in range(1,h.n_grid-1):
        for j in range(1,h.m_grid-1):
            deriv_X[i,j] = (k_ver[i,j]*(X[i+1,j]-X[i,j])/h.delta_z**2 - k_ver[i-1,j]*(X[i,j]-X[i-1,j])/h.delta_z**2    # Horizontal diffusion
                            + k_hor[i,j]*(X[i,j+1]-X[i,j])/h.delta_z**2 - k_hor[i,j-1]*(X[i,j]-X[i,j-1])/h.delta_z**2  # Vertical diffusion
                            + S[i,j]                                                                               # Source/Sink term
                            + cte.rho*(X[i+1,j]*u[i+1,j] - X[i,j]*u[i,j])/h.delta_z)                                     # Horizontal convection
                              
    return(deriv_X)


def source_sink(Tleaf, SW, LW, T, q, u):
    deriv_Tleaf=Tleaf.copy()
    qleaf = Tleaf.copy()
    R_bl_heat = np.zeros((h.n_grid,h.m_grid))
    R_bl_water = np.zeros((h.n_grid,h.m_grid))
    R_stom = np.zeros((h.n_grid,h.m_grid))
    q_sat_Ta = np.zeros((h.n_grid,h.m_grid))
    wat_def_atm = np.zeros((h.n_grid,h.m_grid))


    leaf_length = 0.1
    qleaf[:,:] = 0.622*10**(-5) * 0.611*10**3 * np.exp(17.27*(Tleaf[:,:])/(Tleaf[:,:]-36+273))

    heat_diffu = np.maximum(10**(-6),7*10**(-3) * T * 18.9*10**(-6))
    mu_air = np.maximum(15*10**(-6),-1.36*10**(-14) * T**3 + 1.01*10**(-10) * T**2 + 3.45*10**(-8) * T - 3.40*10**(-6))
    reynolds = leaf_length*u/mu_air
    prandtl = 0.7
    nusselt = 0.66 * (reynolds)**(0.5) + (prandtl)**(1.0/3.0)
    R_bl_heat = leaf_length / (nusselt * heat_diffu)
    water_diffu = np.maximum(10**(-6),7*10**(-3) * T * 21.2*10**(-6))
    schmidt = 0.63#mu_air / water_diffu
    #if (reynolds <= 8000.0):
    sherwood = np.where(reynolds<=8000, 0.66 * (reynolds)**(0.5) + (schmidt)**(0.33), 0.03 * (reynolds)**(0.8) + (schmidt)**(0.33))
    #else: 
    #    sherwood = 0.03 * (reynolds)**(0.8) + (schmidt)**(0.33)
    
    R_bl_water = leaf_length / (sherwood * water_diffu)

    r_stom_min = 23.0 # s/m
    k_0 = 0.25
    R_0 = 125 # W/m2
    lambda_stom = 1.5*10**(-3) # m2.s/kg
    q_sat_Ta[:,:] = 0.622*10**(-5) * 0.611*10**3 * np.exp(17.27*(T[:,:])/(T[:,:]-36+273))
    wat_def_atm[:,:] = cte.rho*np.maximum(0, (q_sat_Ta[:,:]-q[:,:]))
    R_stom[:,:] = r_stom_min/k_0 * (1/h.lai_map[:,:]*(SW[:h.n_grid,:]+R_0)/SW[:h.n_grid,:] * (1+lambda_stom * wat_def_atm[:,:]/r_stom_min)) # M. Guimberteau thesis


    deriv_Tleaf[:,:] = (SW[:h.n_grid,:] + LW[:,:] - cte.cp_air*cte.rho*(Tleaf[:,:]-T[:,:])/R_bl_heat[:,:] - cte.lambda_ev*cte.rho*(qleaf[:,:] - q[:,:])/(R_stom[:,:] + R_bl_water[:,:]))/(h.delta_z*cte.cp_veg*cte.rho_veg)
    return(deriv_Tleaf, R_bl_heat, R_bl_water+R_stom)


def solve_short_waves(SW_top):
    #cumul_lai=np.concatenate((np.cumsum(lai_map,axis=0),[lai_map[0,:]]),axis=0)
    alpha_sw = 0.9
    SW_veg = SW_top * (1-np.exp(-alpha_sw*h.cumul_lai))
    SW_soil = SW_top * np.exp(-alpha_sw * np.sum(h.cumul_lai[:,:],axis=0))

    return(SW_veg,SW_soil)

def gu(x):
    return(np.exp(-0.9*x))

def solve_long_wave(LW_top,k_col,Tleaf):
    alpha = np.zeros((h.n_stop_lai,h.n_stop_lai))
    for i in range(h.n_stop_lai):
        for j in range(h.n_stop_lai):
            if i==0 & j==0:
                alpha[i,j]=-1
            elif i==0 & (j>0 & j<h.n_stop_lai-1):
                alpha[i,j]=gu(h.cumul_lai[0,k_col] - h.cumul_lai[j-1,k_col]) - gu(h.cumul_lai[0,k_col] - h.cumul_lai[j,k_col])
            elif i==0 & j==h.n_stop_lai-1:
                alpha[i,j]=gu(h.cumul_lai[0,k_col])
            elif ((i>0  & i<h.n_stop_lai-1) & (j>1 & j<=i-1)):
                alpha[i,j]=(gu(h.cumul_lai[j,k_col] - h.cumul_lai[i-1,k_col]) - gu(h.cumul_lai[j-1,k_col] - h.cumul_lai[i-1,k_col])
                            - gu(h.cumul_lai[j,k_col] - h.cumul_lai[i,k_col]) + gu(h.cumul_lai[j-1,k_col] - h.cumul_lai[i,k_col]))
            elif ((i>0  & i<h.n_stop_lai-1) & (j==i)):
                alpha[i,j]=2*gu(h.cumul_lai[i-1,k_col] - h.cumul_lai[i,k_col]) - 2
            elif ((i>0  & i<h.n_stop_lai-1) & (j>=i+1 & j<h.n_stop_lai-1)):
                alpha[i,j]=(gu(h.cumul_lai[i,k_col] - h.cumul_lai[j-1,k_col]) - gu(h.cumul_lai[i,k_col] - h.cumul_lai[j,k_col])
                            - gu(h.cumul_lai[i-1,k_col] - h.cumul_lai[j-1,k_col]) + gu(h.cumul_lai[i-1,k_col] - h.cumul_lai[j,k_col]))
            elif i==h.n_stop_lai-1 & j==0:
                alpha[i,j]=gu(h.cumul_lai[0,k_col])
            elif i==0 & (j>0 & j<=h.n_stop_lai-1):
                alpha[i,j]=gu(h.cumul_lai[j,k_col]) - gu(h.cumul_lai[j-1,k_col,])
            elif i==h.n_stop_lai-1 & j==h.n_stop_lai-1:
                alpha[i,j]=-1
            elif (i>0 & i<h.n_stop_lai-1) & j==h.n_stop_lai-1:
                alpha[i,j]=gu(h.cumul_lai[0,k_col]-h.cumul_lai[i-1,k_col]) - gu(h.cumul_lai[0,k_col,]-h.cumul_lai[i,k_col])
            elif (i>0 & i<h.n_stop_lai-1) & j==0:
                alpha[i,j]=gu(h.cumul_lai[i,k_col]) - gu(h.cumul_lai[i-1,k_col,])
    
    LW_veg = np.zeros((h.n_stop_lai,1))
    sigma = 5.6 * 10 ** (-8)
    LW_row=np.concatenate(np.array([sigma*Tleaf[j,k_col]**4 for j in range(h.n_stop_lai)]),np.array(LW_top))
    
    for i in range(h.n_stop_lai):
        for j in range(h.n_stop_lai):
            LW_veg[i,0] = LW_veg[i,0] + alpha[i,j] * LW_row[j]

    return(LW_veg[:,0])
