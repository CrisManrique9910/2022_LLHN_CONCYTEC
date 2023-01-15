import json
import numpy as np
from pathlib import Path
import pandas as pd
import sys

channels = ['1','2+']
types = ['VBF', 'GF']
cards = [13, 14, 15]
tevs = [13]

for type in types[:1]:
    for card in cards[:]:
        for tev in tevs[:]:

            # Programming Parameters

            file_in = f"./data/raw/{type}_{card}_{tev}.hepmc"
            destiny = f"./data/bins/{tev}/{type}/{card}/"
            dfs = {key: {'pos':'', 'neg':''} for key in channels}
            pretext = []

            for key in dfs.keys():
                for tsign in dfs[key].keys():
                    df_ = pd.read_pickle(f'./data/clean/df_photon_{key}_smeared-{type}_{card}_{tev}_{tsign}.pickle')
                    dfs[key][tsign] = df_.reset_index(level=[1])[['t_binned','z_binned']]
                #print(dfs[key])
            #sys.exit()
            it = 0
            i = 1
            limit = 3

            # Action
            hepmc = open(file_in, "r")
            Path(destiny).mkdir(exist_ok=True, parents=True)

            #for sentence in df:
            while i <= limit:
                sentence = hepmc.readline()
                pretext.append(sentence)
                i+=1

            pretext = ''.join(pretext)

            file_all = open(destiny + f'{type}_{card}_{tev}-dfAll_pos.hepmc', 'w')
            file_all.write(pretext)

            for key, predf in dfs.items():
                for tsign, df_ in predf.items():
                    for zb in range(min(df_.z_binned),max(df_.z_binned)+1):
                        for tb in range(min(df_.t_binned), max(df_.t_binned) + 1):
                            output = destiny + f'{type}_{card}_{tev}-df{key}_z{zb}_t{tb}_{tsign}.hepmc'
                            #print(output)
                            file = open(output,'w')
                            file.write(pretext)
                            file.close()
            #sys.exit()

            #while i <= 10000:
            #   i += 1
            #   sentence = hepmc.readline()
            ix=0
            for sentence in hepmc:
                line = sentence.split()
                if line[0] == 'E':
                    file.close()
                    zbn = tbn = -1
                    tlabel = ''
                    event = it
                    line[1] = str(it)
                    sentence = ' '.join(line) + '\n'
                    print(f'RUNNING: {type} {card} {tev} ' + f'Event {event}')
                    for key, predf in dfs.items():
                        for tsign, df_ in predf.items():
                            if event in df_.index:
                                ix+=1
                                zbn = df_.at[event, 'z_binned']
                                tbn = df_.at[event, 't_binned']
                                tlabel = tsign
                                file = open(destiny + f'{type}_{card}_{tev}-df{key}_z{zbn}_t{tbn}_{tlabel}.hepmc','a')
                            #print(f'{key} z{zbn} t{tbn}')
                    it += 1
                if zbn > 0 and tbn > 0 :
                    file.write(sentence)
                    if tlabel == 'pos':
                        file_all.write(sentence)
                    #elif tlabel == 'neg':
                    #    print(it)
                    #    sys.exit()

            file.close()
            file_all.close()
            hepmc.close()
            #print(ix)