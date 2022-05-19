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

            total_All = 0
            total_parts = 0
            for file in glob.glob(origin + f"*.hepmc"):

                #print(file)
                hepmc = open(file, "r")

                i = 1
                limit = 3

                while i <= limit:
                    sentence = hepmc.readline()
                    i += 1

                if 'All' in file:
                    for sentence in hepmc:
                        line = sentence.split()
                        if line[0] == 'E':
                            total_All+=1
                            print(total_All)
                else:
                    for sentence in hepmc:
                        line = sentence.split()
                        if line[0] == 'E':
                            total_parts+=1
                            print(total_parts)

                print(f'TOTALS: {total_All} {total_parts}')

