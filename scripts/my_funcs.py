import numpy as np
import pandas as pd

cross_sec = {13:{'VBF': {50: 0.231009, 30: 0.233562, 10: 0.234305}},
             8: {'VBF': {50: 1.12885, 30: 1.14555, 10: 1.13663},'GF': {50: 7.4004, 30: 7.4165, 10: 7.409}}} # pb
l = {8: 20.3,13: 300} #fb-1
br_nn = 0.21
br_np = {8: {50: 0.719,30: 0.935,10: 0.960}, 13:{50: 0.799, 30: 0.938, 10: 0.960}}

def my_arctan(y,x):

    arctan = np.arctan2(y,x)
    if not isinstance(x, (pd.Series, np.ndarray)):
        arctan = np.asarray(arctan)
    arctan[arctan < 0] += 2*np.pi
    return arctan

def get_scale(tev,type,mass):
    return ((cross_sec[tev][type][mass]*1000)*l[tev]*br_nn*(br_np[tev][mass])**2)/10000