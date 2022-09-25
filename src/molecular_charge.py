#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:59:52 2022

@author: jace
"""

def dictionary(molecular):
    dict_charge = {
        'H4 C8 O4':-2,
        'H12 C6 N2': 0
    }

    if molecular in dict_charge.keys():
        return dict_charge[molecular]
    else:
        print("Warning, you are using dict to predict charge! but the molecule you used is not defined, 0 return!!")
        return 0

def calculator():

    return 0
