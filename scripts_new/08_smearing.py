from matplotlib.ticker import ScalarFormatter, FuncFormatter
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from my_funcs import get_scale
import sys

p0_h = 1.962
p1_h = 0.262

p0_m = 3.650
p1_m = 0.223

points = pd.read_csv(f'./data/raw/z0_res_points.csv',delimiter=',',header=None).values
linear = interp1d(points[:,0],points[:,1])


def t_res(ecell):

    if ecell >= 25:
        resol= np.sqrt((p0_m/ecell)**2 + p1_m**2)
    else:
        resol= min(np.sqrt((p0_h / ecell) ** 2 + p1_h ** 2), 0.57)

    return resol


def z_res(z0):

    z = abs(z0)
    if z < points[0,0]:
        resol = points[0,1]
    elif z < points[-1,0]:
        resol = linear(z)
    else:
        resol = points[-1,1]

    return resol


types = ['VBF','GF']
names = {'VBF':'Vector Boson Fusion','GF':'Gluon Fusion'}
cards = [13,14,15]
mass = {13:50,14:30,15:10}
tevs = [13]

z_bins = [0,50,100,200,300,2000.1]
t_bins = {'1': [0,0.2,0.4,0.6,0.8,1.0,1.5,12.1], '2+': [0,0.2,0.4,0.6,0.8,1.0,1.5,12.1]}

bin_matrix = dict()
lost_ev = dict()

for key, tbin in t_bins.items():
    bin_matrix[key] = np.zeros((len(z_bins) - 1, len(tbin) -1))
#print(bin_matrix)

np.random.seed(0)

for tev in tevs[:]:
    for type in types[:1]:
        for card in cards[:]:

            lost_ev[card] = dict()
            print(f'RUNNING: {type} {card} {tev} ')

            folder_ims = f"./cases/{tev}/{type}/{card}/ATLAS/after_filters/"
            paper_ims = f"./paper/{tev}/{type}/{card}/"
            folder_txt = f"./cases/{tev}/{type}/"
            paper_txt = f"./paper/{tev}/{type}/"
            destiny_info = f'./data/clean/'
            dfs = {'1': '','2+': ''}

            Path(folder_ims).mkdir(parents=True, exist_ok=True)
            Path(folder_txt).mkdir(parents=True, exist_ok=True)
            Path(paper_txt).mkdir(parents=True, exist_ok=True)

            scale = get_scale(tev, type, mass[card])

            ix = 0
            for key in dfs.keys():

                lost_ev[card][key] = []

                dfs[key] = pd.read_pickle(f'./data/clean/df_photon_{key}-{type}_{card}_{tev}.pickle')
                dfs[key]['E'] = np.sqrt(dfs[key]['ET']**2 + dfs[key]['pz']**2)
                dfs[key]['rt_smeared'] = \
                    dfs[key].apply(lambda row: row['rel_tof'] + t_res(0.3*row['E'])*np.random.normal(0,1), axis=1)

                plt.close()
                plt.hist(dfs[key]['rel_tof'],bins=50, color=f'C{ix}')
                plt.xlabel('t_gamma [ns]')
                plt.savefig(folder_ims + f'rel_tof_{key}.jpg')
                #plt.show()

                plt.close()
                plt.hist(dfs[key]['rt_smeared'], bins=50, color=f'C{ix}')
                plt.xlabel('t_gamma Smeared [ns]')
                plt.savefig(folder_ims + f'rel_tof_{key}_smeared.jpg')
                #plt.show()

                dfs[key]['zo_smeared'] = \
                    dfs[key].apply(lambda row: row['z_origin'] + z_res(row['z_origin']) * np.random.normal(0, 1), axis=1)

                #print(dfs[key]['zo_smeared'])
                #sys.exit()
                #print(np.bincount(dfs[key]['z_origin'].values.astype(np.float)))

                plt.close()
                plt.hist(dfs[key]['z_origin'], bins=60, color=f'C{ix}')
                plt.xlabel('z_origin [mm]')
                plt.savefig(folder_ims + f'z_origin_{key}.jpg')
                #plt.show()

                plt.close()
                plt.hist(dfs[key]['zo_smeared'], bins=50, color=f'C{ix}')
                plt.xlabel('z_origin Smeared [mm]')
                plt.savefig(folder_ims + f'z_origin_{key}_smeared.jpg')
                #plt.show()

                lost_ev[card][key].append(dfs[key].shape[0])

                dfs[key] = dfs[key][(0 <= dfs[key]['rt_smeared']) & (dfs[key]['rt_smeared'] <= 12)]
                dfs[key] = dfs[key][np.abs(dfs[key]['zo_smeared']) <= 2000]
                #print(np.min(dfs[key]['zo_smeared']), np.max(dfs[key]['zo_smeared']))

                lost_ev[card][key].append(dfs[key].shape[0])

                plt.close()
                plt.hist(dfs[key]['rt_smeared'], bins=50, color=f'C{ix}')
                plt.xlabel('t_gamma Smeared [ns]')
                plt.savefig(folder_ims + f'rel_tof_{key}_smeared_cutted.jpg')

                plt.close()
                plt.hist(dfs[key]['zo_smeared'], bins=50, color=f'C{ix}')
                plt.xlabel('z_origin Smeared [mm]')
                plt.savefig(folder_ims + f'z_origin_{key}_smeared_cutted.jpg')
                # plt.show()

                dfs[key]['t_binned'] = np.digitize(dfs[key]['rt_smeared'], t_bins[key])
                dfs[key]['z_binned'] = np.digitize(np.abs(dfs[key]['zo_smeared']),z_bins)
                #print(np.min(dfs[key]['t_binned']), np.max(dfs[key]['t_binned']))

                for ind, row in dfs[key].iterrows():
                    #print(ind)
                    bin_matrix[key][row['z_binned'] - 1, row['t_binned'] - 1] +=1
                    #print(row['z_binned'], row['t_binned'])
                ix += 1

            #print(bin_matrix)

            plt.close()
            fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(20, 30))
            plt.subplots_adjust(left=None, bottom=0.05, right=None, top=0.95, wspace=None, hspace=0.3)
            for key in bin_matrix.keys():
                nbins = np.array(range(bin_matrix[key].shape[1] + 1)) + 0.5
                ix = int(key[0])-1
                ir = 0
                for row in axs:
                    row[ix].hist(nbins[:-1], bins=nbins, weights=bin_matrix[key][ir]*scale, histtype='step')
                    row[ix].set_yscale('log')
                    row[ix].set_xticks(np.array(range(bin_matrix[key].shape[1])) + 1)
                    row[ix].set_title(f'Dataset {key} ph - bin z {ir + 1}')
                    ir+=1

            fig.savefig(folder_ims + f'{type}_{card}_zbins_tbins.jpg')
            fig.savefig(paper_ims + f'{type}_{card}_zbins_tbins_BeforeDelphes.jpg')
            #plt.show()

            for key in dfs.keys():
                print(dfs[key].shape)
                dfs[key].to_pickle(f'./data/clean/df_photon_{key}_smeared-{type}_{card}_{tev}.pickle')
            print('dfs saved!')

        message = ''
        for card, value1 in lost_ev.items():
            for ch, value2 in value1.items():
                losses = value2[0] - value2[1]
                message += f'{mass[card]}\t{ch:2}\t{round(losses*scale):5}\t{100*losses/value2[0]:.2f} %\n'
            message += '\n'

        with open(folder_txt + 'negative_events.txt', 'w') as file:
            file.write(message)
        with open(paper_txt + 'negative_events.txt', 'w') as file:
            file.write(message)

        #print(message)





