import json
import numpy as np
from pathlib import Path

types = ['VBF', 'GF']
cards = [13, 14, 15]
tevs = [13]

# Particle Parameters
neutralinos = [9900016, 9900014, 9900012]
neutrinos = [12, 14, 16, 18]
cleptons = [11, 13, 15, 17]

for type in types[:1]:
    for card in cards[:]:
        for tev in tevs[:]:

            # Programming Parameters

            file_in = f"./data/raw/{type}_{card}_{tev}.hepmc"
            destiny = "./data/clean/"
            file_out = f'recollection_leptons-{type}_{card}_{tev}.json'

            it = 0
            i = 0
            limit = 2

            # Action
            df = open(file_in, "r")
            Path(destiny).mkdir(exist_ok=True, parents=True)

            while it < 2:
                df.readline()
                it += 1

            # Initializing values
            data = dict()
            codes = []
            num = 0
            p_scaler = None
            d_scaler = None

            for sentence in df:
                # while i<(limit+20):
                # sentence = df.readline()
                # print(sentence)
                line = sentence.split()
                if line[0] == 'E':
                    # num = int(line[1])
                    if num > 0:  # Selection of relevant particles/vertices in the last event
                        # print(mpx,mpy)
                        selection = set()
                        data[num - 1] = {'params': params, 'v': dict(), 'l': []}
                        for photon in holder['l']:
                            # SIgnal which vertex give charged leptons
                            outg_a = photon[-1]
                            data[num - 1]['l'].append(photon)
                            selection.add(outg_a)
                        for vertex in selection:
                            # select only the vertices that give charged leptons
                            data[num - 1]['v'][vertex] = holder['v'][vertex]
                    # print(data)
                    holder = {'v': dict(), 'l': []}
                    i += 1
                    print(f'RUNNING: {type} {card} {tev} ' + f'Event {num}')
                    num += 1
                elif line[0] == 'U':
                    params = line[1:]
                    if params[0] == 'GEV':
                        p_scaler = 1
                    else:
                        p_scaler = 1 / 1000
                    if params[1] == 'MM':
                        d_scaler = 1
                    else:
                        d_scaler = 10
                    # print(p_scaler)
                elif line[0] == 'V':
                    outg = int(line[1])
                    info = *[float(x) for x in line[3:6]], int(line[8])  # x,y,z,number of outgoing
                    holder['v'][outg] = list(info)
                    # print(outg)
                elif line[0] == 'P':
                    pdg = line[2]
                    if (abs(int(pdg)) in cleptons) and (line[8] == '1'):
                        info = pdg, *[float(x) for x in line[3:8]], outg  # px,py,pz,E,m,vertex from where it comes
                        holder['l'].append(list(info))
                    codes.append(pdg)
            df.close()

            # Event selection for the last event
            selection = set()
            data[num - 1] = {'params': params, 'v': dict(), 'l': []}
            for photon in holder['l']:
                # SIgnal which vertex give charged leptons
                outg_a = photon[-1]
                data[num - 1]['l'].append(photon)
                selection.add(outg_a)
            for vertex in selection:
                # select only the vertices that give charged leptons
                data[num - 1]['v'][vertex] = holder['v'][vertex]

            # print(data[num])
            # print(data.keys())

            with open(destiny + file_out, 'w') as file:
                json.dump(data, file)

            print(f'RUNNING: {type} {card} {tev} ' + f'info saved in {file_out}')