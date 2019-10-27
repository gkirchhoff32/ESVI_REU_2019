# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:06:12 2019

@author: Grant
"""
   
def saline(vrms_list):
    salinity = []
    # newly calibrated using 7.0 and 35.0 salinity values
    for i in vrms_list:
        m = 0.02118707
        b = 0.09562055
        S = (i - b)/m
        salinity.append(S)
        
    return salinity
