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
masses = {13:50,14:30,15:10}
tevs = [13]
edges = [-0.1, 30, 50, np.inf]
labels = list(range(len(edges) - 1))
limits_z = [0,50,100,200,300,2000]

burrito = {card:{'1':np.zeros((5,7,3)), '2+':np.zeros((5,7,3))} for card in cards}

#print(burrito)
for tev in tevs[:]:
    for type in types[:1]:

        ymin = ymax = []
        for card in cards[:1]:
            
            print(f'RUNNING: {type} {card} {tev} ')
            origin = f"./data/bins/{tev}/{type}/{card}/"
            folder_ims = f"./cases/{tev}/{type}/{card}/ATLAS/after_Delphes_and_VBF/"
            paper = f'./paper/{tev}/{type}/{card}/'
            ones1 = ones2 = 0
            totals1 = totals2 = 0

            mass = masses[card]
            scale = get_scale(tev,type,mass)
            #print(scale)
            #sys.exit()

            Path(folder_ims).mkdir(exist_ok=True, parents=True)
            Path(paper).mkdir(exist_ok=True, parents=True)

            for jet_file in glob.glob(origin + f"*_jets.pickle"):

                ph_file = jet_file.replace('_jets', '')
                df = pd.read_pickle(ph_file)
                df_jets = pd.read_pickle(jet_file)

                nums = df.groupby(['N']).size()
                if '2+' in jet_file:
                    ones2 += nums[nums == 1].shape[0]
                    totals2 += nums.shape[0]
                else:
                    ones1 += nums[nums > 1].shape[0]
                    totals1 += nums.shape[0]

                df_jets['px'] = df_jets.pt * np.cos(df_jets.phi)
                df_jets['py'] = df_jets.pt * np.sin(df_jets.phi)
                df_jets['pz'] = df_jets.pt / np.tan(2 * np.arctan(np.exp(df_jets.eta)))
                #print(df_jets[['pt','eta','pz']])
                #print(df_jets[['px', 'py', 'pt']])

                df_jets['E'] = np.sqrt(df_jets.M**2 + df_jets.pt**2 + df_jets.pz**2)
                #print(df_jets[['M', 'pt', 'pz', 'E']])

                ################ Jet Filters ####################
                ################ 1
                #############hmore than one jet

                events = df.shape[0]
                # print(f'Eventos totales:'.ljust(50),events)
                jet_nums = df_jets.groupby(['N']).size()
                jet_nums = jet_nums.reindex(df.index.get_level_values(0).unique(), fill_value=0)
                # print(df.index.get_level_values(0).unique())
                # sys.exit()
                plus2 = jet_nums[jet_nums > 1].index

                #######################################
                ####### 2
                ############# PT de ambos jets >30 GeV
                df = df.loc[plus2]
                df_jets = df_jets.loc[plus2]
                events = plus2.size
                df_jets2 = df_jets.query('id in [0,1]')

                df_jets2 = df_jets2.pivot_table(index="N", columns="id")

                ##############################################
                ################
                ############ eta1.eta2 <0 y |eta| <5
                # print(df_jets2.index)
                df_jets2 = df_jets2[df_jets2.pt[1] > 30]
                df_jets = df_jets.loc[df_jets2.index]
                df = df.loc[df_jets2.index]
                events = df_jets2.shape[0]
                # print(f'Eventos con lead jets pt > 30 GeV:'.ljust(50), events)

                ##############################################
                ################
                ############ |D eta| > 4.2
                df_jets2 = df_jets2[(abs(df_jets2.eta[0]) < 5) & (abs(df_jets2.eta[1]) < 5) &
                                    ((df_jets2.eta[0] * df_jets2.eta[1]) < 0)]
                events = df_jets2.shape[0]
                # print(f'Eventos con restricciones de eta:'.ljust(50), events)
                deta_jet = df_jets2[abs(df_jets2.eta[0] - df_jets2.eta[1]) > 4.2].index
                # print(deta_jet)

                #######################################
                #######
                ############# HT > 200 GeV
                df_jets2 = df_jets2.loc[deta_jet]
                df_jets = df_jets.loc[deta_jet]
                events = deta_jet.size
                # print(f'Eventos con delta eta entre lead jets > 4.2:'.ljust(50), events)
                pts = df_jets.query('pt > 20').groupby(['N']).sum().pt
                ht = pts[pts > 200].index
                # print(ht.size)

                #######################################
                #######
                ############# MI > 750 GeV
                df_jets = df_jets.loc[ht]
                df_jets2 = df_jets2.loc[ht]
                events = ht.size
                # print(f'Eventos con HT mayor de 200 GeV:'.ljust(50),events)
                inv_m = np.sqrt((df_jets2.E[0] + df_jets2.E[1]) ** 2 -
                                ((df_jets2.px[0] + df_jets2.px[1]) ** 2 + (df_jets2.py[0] + df_jets2.py[1]) ** 2 +
                                 (df_jets2.pz[0] + df_jets2.pz[1]) ** 2))
                inv_m = inv_m[inv_m > 750].index

                ###########################
                df = df.loc[inv_m]

                ############# PHOTON CUTS ###############3

                if 'All' in ph_file:
                    df.to_pickle(ph_file.replace('.pickle','_afterJetFilters.pickle'))
                    #sys.exit()
                    continue
                else:
                    z = int(re.search('z(.)_', jet_file).group(1)) - 1
                    t = int(re.search('t(.)_', jet_file).group(1)) - 1

                nums = df.groupby(['N']).size()
                if '2+' in jet_file:
                    df2 = df.loc[nums[nums > 1].index]
                    #dfs = {'2+': df2}
                    df1 = df.loc[nums[nums == 1].index]
                    dfs = {'1':df1,'2+':df2}
                else:
                    df1 = df.loc[nums[nums == 1].index]
                    #df2 = df.loc[nums[nums > 1].index]
                    dfs = {'1':df1}

                for key, value in list(dfs.items())[:]:
                    value = value[np.abs(value.eta) < 1.52]
                    value = value[(0.3*value.E) > 10]
                    value = value.loc[value.groupby('N')['pt'].idxmax()]
                    #print(value.E.min())
                    value['met_bin'] = pd.cut(value['MET'], bins=edges, labels=labels)
                    for bin in labels:
                        burrito[card][key][z,t,bin] += scale*value[value.met_bin == bin].shape[0]
                        #if t in [6] and z in [0,1] and bin == 2:
                        #    print(ph_file)
                        #    print(value[value.met_bin == bin]['MET'])
            #print(burrito)
            events = sum([item.sum() for item in burrito[card].values()])
            #print(events)
            sys.exit()
            fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(20, 30))
            plt.subplots_adjust(left=None, bottom=0.05, right=None, top=0.95, wspace=None, hspace=0.3)
            for key in burrito[card].keys():
                nbins = np.array(range(burrito[card][key].shape[1] + 1)) + 0.5
                ix = int(key[0])-1
                ir = 0
                for row in axs:
                    row[ix].hist(np.tile(nbins[:-1],(burrito[card][key].shape[-1],1)).transpose(),
                                 bins=nbins, weights=burrito[card][key][ir], histtype='step',
                                 stacked=False,color=['C3','C2','C0'],label=met_labels)
                    row[ix].set_yscale('log')
                    row[ix].set_xticks(np.array(range(burrito[card][key].shape[1])) + 1)
                    row[ix].set_title(f'Dataset {key} ph - bin z {ir + 1}')
                    row[ix].legend()
                    ymax.append(row[ix].get_ylim()[1])
                    ymin.append(row[ix].get_ylim()[0])
                    ir+=1
            plt.setp(axs, ylim=(min(ymin), max(ymax)))
            plt.suptitle(f'{type} - {mass} GeV - {round(events)} events')
            #plt.show()
            fig.savefig(folder_ims + f'{type}_{card}_zbins_tbins_AD_scaled.png')
            fig.savefig(paper + f'{type}_{card}_zbins_tbins_AD_scaled.png')
            plt.close()
            #print('De 1 a 2+:',ones1/totals1*100)
            #print('De 2+ a 1:',ones2/totals2*100)

        for card, data in list(burrito.items())[:]:

            folder_ims = f"./cases/{tev}/{type}/{card}/ATLAS/after_Delphes_and_VBF/"
            paper = f'./paper/{tev}/{type}/{card}/'

            message=''
            for key, value in list(data.items())[:]:
                nbins = np.array(range(value.shape[1] + 1)) + 0.5
                #print(nbins)
                final = '['
                i = 0
                for zbin in list(range(1,value.shape[0]+1))[:]:
                    #print(zbin)
                    events = np.sum(value[zbin - 1])
                    #sys.exit()
                    if i == (value.shape[0] - 1):
                        final = ']'
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(np.tile(nbins[:-1], (value.shape[-1], 1)).transpose(),
                                 bins=nbins, weights=value[zbin-1], histtype='step',
                                 stacked=False, color=['C3', 'C2', 'C0'], label=met_labels)
                    ax.set_yscale('log')
                    ax.set_xlabel('$\mathregular{t_{\gamma}}$ bins', fontsize=15)
                    ax.set_xticks(nbins[1:-1])
                    ax.set_xticks(nbins[1:] - 0.5, minor=True)
                    ax.tick_params(axis='x', which='major', direction='in',
                                    labelbottom=False, width=1.5, length=7)
                    ax.tick_params(axis='x', which='minor', length=0,labelsize=15)
                    ax.tick_params(axis='y', labelsize=15)
                    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
                    ax.set_xlim(0.5, 7.5)
                    ax.set_ylim(min(ymin), max(ymax))
                    ax.text(0.99, 0.66, f'{round(events)} Events',
                            horizontalalignment="right", fontsize=14, transform=ax.transAxes)
                    ax.text(0.99, 0.55, f'|z| interval (mm):\n[{limits_z[i]},{limits_z[i + 1]}{final}',
                             horizontalalignment="right",fontsize=14, transform=ax.transAxes)
                    ax.legend(fontsize=15)
                    plt.tight_layout()
                    fig.savefig(paper + f'{type}_{card}_{key}ph_zbin{zbin}_scaled.png')
                    plt.close()
                    #plt.show()

                    i+=1

                    for idx, metl in enumerate(met_labels):
                        m = masses[card]
                        dist = np.sum(value[zbin-1,:,idx])
                        if events != 0:
                            message+=f'{m}\t{key:2}\t{zbin}\t{metl:3}\t{round(dist):4}\t{100*dist/events:.2f} %\n'
                        else:
                            message += f'{m}\t{key:2}\t{zbin}\t{metl:3}\t{round(dist):4}\t - %\n'
                    message += '\n'
                message += '\n'

            with open(paper + 'region_dist.txt','w') as file:
                file.write(message)
            #print(message)

