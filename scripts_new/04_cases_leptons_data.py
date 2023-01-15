import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from my_funcs import my_arctan
import sys
import glob

#CMSdet_radius = 1.29 # meters
#CMSdet_semilength = 2.935
ATLASdet_radius= 1.4
ATLASdet_semilength = 2.9

mass_conversion = 1.78266192*10**(-27)	#GeV to kg
p_conversion = 5.344286*10**(-19)	#GeV to kg.m/s
c_speed = 299792458	#m/s


def plot_config(figsize,xtitle,ytitle,xsize, ysize, xtsize, ytsize):
    plt.figure(figsize=figsize)
    plt.ylabel(ytitle, fontsize=ysize)
    plt.xlabel(xtitle, fontsize=xsize)
    plt.xticks(fontsize=xtsize)
    plt.yticks(fontsize=ytsize)
    return

def pipeline(detec_radius, detec_semilength, detec_name):

    pts = []
    pzs = []
    nus = []

    counter = 0
    dicts = []

    for file_in in sorted(glob.glob(f'./data/clean/recollection_leptons-{type}_{card}_{tev}-*.json')):

        print(file_in)
        try:
            del data
        except UnboundLocalError:
            file_in

        with open(file_in, 'r') as file:
            data = json.load(file)
        #print(len(data.keys()))
        for event in list(data.keys())[:]:
            print(f'RUNNING: {type} {card} {tev} - {detec_name} - Event {event}')
            holder = data[event]
            params = holder['params']
            # Defining scaler according to parameters units
            if params[0] == 'GEV':
                p_scaler = 1  # GeV to GeV
            elif params[0] == 'MEV':
                p_scaler = 1 / 1000  # MeV to GeV
            else:
                print(params[0])
                continue

            if params[1] == 'MM':
                d_scaler = 1  # mm to mm
            elif params[1] == 'CM':
                d_scaler = 10  # cm to mm
            else:
                print(params[1])
                continue

            # Adjusting detector boundaries
            r_detec = detec_radius * 1000  # m to mm
            z_detec = detec_semilength * 1000

            # Define our holder for pairs:
            pt_dum = 0
            ix = 0
            for lepton in holder['l']:
                info = dict()
                info['Event'] = int(event)

                vertex = str(lepton[-1])
                pdg = lepton[0]
                px, py, pz = [p_scaler*ix for ix in lepton[1:4]]
                x, y, z = [d_scaler*ix for ix in holder['v'][vertex][0:3]]
                mass = lepton[-2] * p_scaler
                r = np.sqrt(x ** 2 + y ** 2)
                # Calculating transverse momentum
                pt = np.sqrt(px ** 2 + py ** 2)
                Et = np.sqrt(mass ** 2 + pt ** 2)

                if r >= (r_detec) or abs(z) >= (z_detec):
                     continue
                elif pt < 10:
                    continue

                # print(mass_ph)
                info['id'] = ix
                info['pdg'] = pdg
                info['r'] = r / r_detec
                info['z'] = z / z_detec
                info['px'] = px
                info['py'] = py
                info['pt'] = pt
                info['pz'] = pz
                info['ET'] = Et

                ix += 1
                counter += 1

                phi = my_arctan(py, px)

                theta = np.arctan2(pt, pz)
                nu = -np.log(np.tan(theta / 2))

                pts.append(pt)
                pzs.append(pz)
                nus.append(nu)

                info['eta']=nu
                info['phi']=phi

                dicts.append(info)

    print(f'Detected leptons in {detec_name}: {counter}')

    # Create the directory for the images
    destiny_ims = destiny_ims0 + detec_name
    Path(destiny_ims).mkdir(parents=True,exist_ok=True)


    dicts = pd.DataFrame(dicts)
    dicts = dicts.sort_values(by=['Event','pt'],ascending=[True,False])
    g = dicts.groupby('Event', as_index=False).cumcount()
    dicts['id'] = g
    dicts = dicts.set_index(['Event','id'])
    #print(dicts.head(20))
    #print(dicts.tail(20))
    #print(dicts.z.max(),dicts.z.min(),dicts.r.max())

    dicts.to_pickle(destiny_info+f'lepton_df-{type}_{card}_{tev}.pickle')
    print('df saved!')
    # Grafiquitos
    lim = 7500
    nbins = 100

    plot_config((12, 8), 'PT lepton [GeV]', 'Counts', 16, 16, 14, 14)
    plt.hist(pts, bins=nbins, color='C1')
    plt.savefig(f'{destiny_ims}/{detec_name}_leptons_pt.jpg')
    plt.close()

    plot_config((12, 8), 'PZ lepton [GeV]', 'Counts', 16, 16, 14, 14)
    plt.hist(pzs, bins=nbins, color='C2')
    plt.savefig(f'{destiny_ims}/{detec_name}_leptons_pz.jpg')
    plt.close()

    plot_config((12, 8), '(Pseudo)rapidity lepton', 'Counts', 16, 16, 14, 14)
    plt.hist(nus, bins=nbins, color='C4')
    plt.savefig(f'{destiny_ims}/{detec_name}_leptons_pseudorapidity.jpg')
    plt.close()

    return


types = ['VBF','GF']
cards = [14, 15]
tevs = [13]

for type in types[:1]:
    for card in cards[:]:
        for tev in tevs[:]:
            case = f"./cases/{tev}/{type}/{card}/"

            destiny_info = './data/clean/'
            destiny_ims0 = case

            pipeline(ATLASdet_radius,ATLASdet_semilength,'ATLAS')