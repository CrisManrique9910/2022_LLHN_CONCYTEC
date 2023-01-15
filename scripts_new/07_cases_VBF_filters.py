import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from my_funcs import my_arctan

types = ['VBF','GF']
names = {'VBF':'Vector Boson Fusion','GF':'Gluon Fusion'}
cards = [13,14,15]
mass = {13:50,14:30,15:10}
tevs = [13]

for tev in tevs[:]:
    for type in types[:1]:
        for card in cards[:]:

            print(f'RUNNING: {type} {card} {tev} ')

            ip = 1
            case = f"./cases/{tev}/{type}/{card}/"
            destiny_info = f'./data/clean/'
            #destiny_paper = f"./cases/"
            #destiny_paper = f"./paper/"
            photon_in = f'./data/clean/photon_df-{type}_{card}_{tev}.pickle'
            leptons_in = f'./data/clean/lepton_df-{type}_{card}_{tev}.pickle'
            jets_in = f'./data/clean/jets-{type}_{card}_{tev}.pickle'

            #Path(destiny_paper).mkdir(parents=True, exist_ok=True)

            df = pd.read_pickle(photon_in)
            df_jets = pd.read_pickle(jets_in)
            df_leps = pd.read_pickle(leptons_in)
            #print(df.z.max(), df.z.min(), df.r.max())
            #print(df.index.get_level_values(0).nunique())
            #sys.exit()

            ################ Jet Filters ####################
            ################ 1
            #############hmore than one jet

            events = df.shape[0]
            #print(f'Eventos totales:'.ljust(50),events)
            jet_nums = df_jets.groupby(['Event']).size()
            jet_nums = jet_nums.reindex(df.index.get_level_values(0).unique(),fill_value=0)
            #print(df.index.get_level_values(0).unique())
            #sys.exit()
            plus2 = jet_nums[jet_nums>1].index

            #######################################
            ####### 2
            ############# PT de ambos jets >30 GeV
            df = df.loc[plus2]
            df_jets = df_jets.loc[plus2]
            events = plus2.size
            df_jets2 = df_jets.query('id in [0,1]')

            df_jets2 = df_jets2.pivot_table(index="Event", columns="id")

            ##############################################
            ################
            ############ eta1.eta2 <0 y |eta| <5
            #print(df_jets2.index)
            df_jets2 = df_jets2[df_jets2.pt[1] > 30]
            df_jets = df_jets.loc[df_jets2.index]
            df = df.loc[df_jets2.index]
            events = df_jets2.shape[0]
            #print(f'Eventos con lead jets pt > 30 GeV:'.ljust(50), events)

            ##############################################
            ################
            ############ |D eta| > 4.2
            df_jets2 = df_jets2[(abs(df_jets2.eta[0]) < 5) & (abs(df_jets2.eta[1]) < 5) &
                                ((df_jets2.eta[0] * df_jets2.eta[1]) < 0)]
            events = df_jets2.shape[0]
            #print(f'Eventos con restricciones de eta:'.ljust(50), events)
            deta_jet = df_jets2[abs(df_jets2.eta[0] - df_jets2.eta[1]) > 4.2].index
            # print(deta_jet)

            #######################################
            #######
            ############# HT > 200 GeV
            df_jets2 = df_jets2.loc[deta_jet]
            df_jets = df_jets.loc[deta_jet]
            events = deta_jet.size
            #print(f'Eventos con delta eta entre lead jets > 4.2:'.ljust(50), events)
            pts = df_jets.query('pt > 20').groupby(['Event']).sum().pt
            ht = pts[pts > 200].index
            # print(ht.size)

            #######################################
            #######
            ############# MI > 750 GeV
            df_jets = df_jets.loc[ht]
            df_jets2 = df_jets2.loc[ht]
            events = ht.size
            #print(f'Eventos con HT mayor de 200 GeV:'.ljust(50),events)
            inv_m = np.sqrt((df_jets2.E[0] + df_jets2.E[1]) ** 2 -
                            ((df_jets2.px[0] + df_jets2.px[1]) ** 2 + (df_jets2.py[0] + df_jets2.py[1]) ** 2 +
                             (df_jets2.pz[0] + df_jets2.pz[1]) ** 2))
            inv_m = inv_m[inv_m > 750].index

            ########### CONTAINED ################
            df = df.loc[inv_m]
            events = inv_m.size

            df = df[(df.r < 1) & (np.abs(df.z) < 1)]

            ######## Min(|DELTA R|)  > 0.2

            # First, between photons
            df["dr_ph"] = 100.0
            for ix in df.index.get_level_values(0).unique()[:]:
                event = df.loc[ix]
                #rint(event.loc[1])
                for index1, row1 in event.iterrows():
                    drs = []
                    #print(index1)
                    for index2, row2 in event.iterrows():
                        if index1 != index2:
                            drs.append(np.sqrt((row2.phi - row1.phi)**2 + (row2.eta - row1.eta)**2))
                    #print(drs)
                    if len(drs) != 0:
                        df.at[(ix,index1),'dr_ph'] = min(drs)

            df = df[df['dr_ph'] > 0.2]

            # Then, jets
            df_jets = df_jets.loc[df.index.get_level_values(0).unique()]

            df["dr_jets"] = 100.0
            for ix in df.index.get_level_values(0).unique()[:]:
                event_ph = df.loc[ix]
                # print(ix)
                event_jets = df_jets.loc[ix]
                for index_ph, row_ph in event_ph.iterrows():
                    drs = []
                    # print(index1)
                    for index_j, row_j in event_jets.iterrows():
                        drs.append(np.sqrt((row_j.phi - row_ph.phi) ** 2 + (row_j.eta - row_ph.eta) ** 2))
                    #print(drs)
                    df.at[(ix, index_ph), 'dr_jets'] = min(drs)

            #print(df_jets.loc[3])
            df = df[df['dr_jets'] > 0.2]
            #print(df['dr_jets'])

            # Finally, leptons
            df_leps = \
                    df_leps.loc[
                        df.index.get_level_values(0).unique().intersection(df_leps.index.get_level_values(0).unique())
                    ]
            #print(df_leps)

            df["dr_lep"] = 100.0
            for ix in df.index.get_level_values(0).unique()[:]:
                event_ph = df.loc[ix]
                #print(ix)
                try:
                    event_lep = df_leps.loc[ix]
                    for index_ph, row_ph in event_ph.iterrows():
                        drs = []
                        # print(index1)
                        for index_l, row_l in event_lep.iterrows():
                            drs.append(np.sqrt((row_l.phi - row_ph.phi) ** 2 + (row_l.eta - row_ph.eta) ** 2))
                        # print(drs)
                        df.at[(ix, index_ph), 'dr_lep'] = min(drs)
                except KeyError:
                    continue

            df = df[df['dr_lep'] > 0.2]
            #print(df['dr_lep'])

            # Photons not in the barrel
            ph_num = df.groupby(['Event']).size()
            dfs = {'1': df.loc[ph_num[ph_num == 1].index], '2+': df.loc[ph_num[ph_num > 1].index]}

            for key, df_ in dfs.items():
                df_ = df_[1.52 > np.abs(df_.eta)]
                dfs[key] = df_

            #print(dfs['2+'][['pt','pz']])
            dfs['2+'] = dfs['2+'].loc[dfs['2+'].groupby('Event')['pt'].idxmax()]

            for key, df_ in dfs.items():
                df_.to_pickle(destiny_info + f'df_photon_{key}-{type}_{card}_{tev}.pickle')
            print('dfs saved!')



