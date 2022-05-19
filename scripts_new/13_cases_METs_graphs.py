import sys
import os
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from pathlib import Path
import re
from my_funcs import get_scale

types = ['VBF', 'GF']
met_labels = ['BKG', 'CR', 'SR']
cards = [13, 14, 15]
masses = {13: 50, 14: 30, 15: 10}
tevs = [13]
edges = [-0.1, 30, 50, np.inf]
labels = list(range(len(edges) - 1))
limits_z = [0, 50, 100, 200, 300, 2000]

# print(burrito)
for tev in tevs[:]:
    for type in types[:1]:
        for card in cards[:1]:

            print(f'RUNNING: {type} {card} {tev} ')
            origin = f"./data/bins/{tev}/{type}/{card}/"
            folder_ims = f"./cases/{tev}/{type}/{card}/ATLAS/after_Delphes_and_VBF/"
            paper = f'./paper/{tev}/{type}/{card}/'

            mass = masses[card]

            Path(folder_ims).mkdir(exist_ok=True, parents=True)
            Path(paper).mkdir(exist_ok=True, parents=True)

            total = 0
            for jet_file in glob.glob(origin + f"*_jets.pickle"):

                ph_file = jet_file.replace('_jets', '')
                df = pd.read_pickle(ph_file)

                if 'All' in ph_file:
                    print(df)
                    print(df.index.get_level_values(0).nunique())
                else:
                    total += df.index.get_level_values(0).nunique()

            print(total)
            '''
            for file in glob.glob(origin + f'*_afterJetFilters.pickle'):

                value = pd.read_pickle(file)

                value = value[np.abs(value.eta) < 1.52]
                value = value[(0.3 * value.E) > 10]
                value = value.loc[value.groupby('N')['pt'].idxmax()]

                print(value)
                print(value.shape[0])
            '''
