import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from scipy.stats import hmean
import pandas as pd
import math
import glob

n = 6
masses = [50,30,10]
param_mass = {50:4, 30:5, 10:6}
nlines_mass = {50:0, 30:2, 10:4}
nlines_type = {'VBF':0,'GF':1}
maxi = 1.0
mini = 0.1

ATLASdet_radius= 1.775
ATLASdet_semilength = 4.050

types = ['VBF','GF']

if not (os.path.isfile(f".optimal_widths.txt")):
    file = open(f".optimal_widths.txt", 'w+')
    file.write('\n'*(len(masses)*len(types)))
    file.close()

for mass in masses[:]:
    #if mass == 50:
     #   continue
    for type in types[:]:
        i = 1
        nlb = 5 * 10 ** (-17)
        nub = 5 * 10 ** (-13)
        hm_max = 1
        hm_min = 0
        Path(f"./{mass}/{type}/graphs/").mkdir(parents=True, exist_ok=True)

        if not(os.path.isfile(f"./{mass}/{type}/optimal_widths-{mass}_{type}.txt")):
            file = open(f"./{mass}/{type}/optimal_widths-{mass}_{type}.txt",'w+')
            file.close()

        #while i < 2:
         #   print(i)
        while (abs(hm_max - hm_min) > (0.02*(maxi - mini))) and \
                (((nub - nlb)/(10**(math.floor(math.log(nlb, 10))))) > 0.1):
            #print(hn_max - hn_min)
            print(i)
            lb = nlb
            ub = nub
            if ((ub - lb)/(10**(math.floor(math.log(nlb, 10)))) ) > 10:
                print(f'log:\t{((ub - lb)/(10**(math.floor(math.log(nlb, 10)))) )}')
                init = np.logspace(np.log10(lb), np.log10(ub), n)
                xsc = 1
            else:
                print(f'lineal:\t{((ub - lb)/(10**(math.floor(math.log(nlb, 10)))) )}')
                init = np.linspace(lb, ub, n)
                xsc = 0
            print(init)
            Path(f"./{mass}/{type}/{i}/raw/").mkdir(parents=True, exist_ok=True)
            d_cards = f'./{mass}/{type}/{i}/cards'
            Path(d_cards).mkdir(parents=True, exist_ok=True)
            origin_card = f'/home/cristian/Desktop/HEP_Jones/param_c/param_card_{param_mass[mass]}.dat'

            if not(os.path.isfile(f'./{mass}/{type}/{i}/clean/photon_df-{type}_iter{i:03}_card{n}_8.xlsx')):

                with open(origin_card, 'r') as file_r:
                    list_of_lines = file_r.readlines()
                # print(list_of_lines[192]

                for iw, width in enumerate(init[:]):
                    list_of_lines[192] = f'DECAY  9900014   {width:.6e}\n'

                    file_w = open(f"{d_cards}/param_card_{iw + 1}.dat", "w")
                    file_w.writelines(list_of_lines)
                    file_w.close()

                os.system('sed -i "4s/.*/'f'mass={mass}''/" "master_scan.sh"')
                os.system('sed -i "5s/.*/'f'n={n}''/" "master_scan.sh"')
                os.system('sed -i "6s/.*/'f'iter={i}''/" "master_scan.sh"')
                os.system('sed -i "7s/.*/'f'type=\x27{type}\x27''/" "master_scan.sh"')

                input(f"If you already have the .hepmc's for this iteration, press Enter")

                os.system('sed -i "5s/.*/'f'mass={mass}''/" "cases_hepmc_reader.py"')
                os.system('sed -i "6s/.*/'f'n={n}''/" "cases_hepmc_reader.py"')
                os.system('sed -i "7s/.*/'f'iter={i}''/" "cases_hepmc_reader.py"')
                os.system('sed -i "9s/.*/'f'types=[\x27{type}\x27]''/" "cases_hepmc_reader.py"')

                os.system('python cases_hepmc_reader.py')

                os.system('sed -i "7s/.*/'f'mass={mass}''/" "cases_photons_data.py"')
                os.system('sed -i "8s/.*/'f'n={n}''/" "cases_photons_data.py"')
                os.system('sed'
                          ' -i "9s/.*/'f'iter={i}''/" "cases_photons_data.py"')
                os.system('sed -i "10s/.*/'f'types=[\x27{type}\x27]''/" "cases_photons_data.py"')

                os.system('python cases_photons_data.py')

            if (os.path.isfile(f'./{mass}/{type}/{i}/clean/photon_df-{type}_iter{i:03}_card{n}_8.xlsx'))\
                    and len(glob.glob(f"./{mass}/{type}/{i}/raw/*.hepmc")) != 0:
                #print(0)
                os.system(f'rm -rf ./{mass}/{type}/{i}/raw')

                #input(f"If you already have the df's for this iteratio, press Enter")

            predf = {'widths':init,f'r_{type}':[],f't_{type}':[],f's_{type}':[]}

            for ind in list(range(1,n+1)):

                file_in = f'./{mass}/{type}/{i}/clean/photon_df-{type}_iter{i:03}_card{ind}_8.xlsx'
                df = pd.read_excel(file_in, header=[0,1],index_col=0)

                df['r'] = (df['r'] / 1000) / ATLASdet_radius
                df['z'] = (np.abs(df['z']) / 1000) / ATLASdet_semilength
                df['inside'] = ~((df['z_origin',1].isna()) | (df['z_origin',2].isna()))
                df_util = df[df['inside']]

                t = np.median(df_util['rel_tof'].to_numpy().flatten())
                s = np.std(df_util[(np.abs(df_util['z_origin',1]) <= 2000) & (np.abs(df_util['z_origin',2]) <= 2000)]
                           ['z_origin'].to_numpy().flatten(),ddof=1)
                #s = np.std(df_util['z_origin'].to_numpy().flatten(), ddof=1)
                r = df['inside'].sum()/df['inside'].count()
                predf[f'r_{type}'].append(r)
                predf[f't_{type}'].append(t)
                predf[f's_{type}'].append(s)
                #print(np.amax(df_util[(np.abs(df_util['z_origin',1]) <= 2000) & (np.abs(df_util['z_origin',2]) <= 2000)]
                #              ['z_origin'].to_numpy().flatten()))

            new_df = pd.DataFrame(predf)
            #print(new_df)
            atris = new_df.columns[1:]
            if i == 1:
                maxis = new_df[atris].max()
                minis = new_df[atris].min()
            #else:
            #    maxis = pd.concat([maxis, new_df[atris].max()], axis=1).max(axis=1)
            #    minis = pd.concat([minis, new_df[atris].min()], axis=1).min(axis=1)
            new_df[atris] = mini + (new_df[atris] - minis)*(maxi - mini)/(maxis - minis)
            new_df['hmean'] = hmean(new_df[atris],axis=1)
            #print(new_df)
            #if i>=5:
            #    print(new_df['hmean'])
            hm_max = new_df['hmean'].max()
            maxw = new_df[new_df['hmean']==new_df['hmean'].max()].widths.values[0]
            pos = np.where(init==maxw)[0][0]
            if pos == 0:
                nub = init[1]
                nlb = init[0]
            elif pos == (n-1):
                nub = init[pos]
                nlb = init[pos-1]
            else:
                nub = init[pos+1]
                nlb = init[pos-1]
            hm_min = np.min(new_df[new_df['widths'].isin([nlb,nub])].hmean)
            #print(hm_max - hm_min)
            print(nlb,nub)
            bases = ['r','s','t']
            markers = {'VBF':',','GF':'^'}
            colors = ['c','m','y']

            # plt.subplots(figsize=(6,8))

            for ixa, atr in enumerate(bases[:]):
                plt.scatter(x=new_df['widths'], y=new_df[f'{atr}_{type}'],
                            marker=markers[type], color= colors[ixa], label=f'{atr}_{type}',zorder=2)
            plt.scatter(x=new_df['widths'], y=new_df[f'hmean'], marker='o', color='r', label='h.mean',zorder=2)
            plt.axvline(x=nlb,color='b',linestyle='--',zorder=1)
            plt.axvline(x=nub, color='b', linestyle='--',zorder=1)
            plt.axvspan(nlb,nub,facecolor='pink',alpha=0.4,zorder=0)
            plt.title(f'Mass: {mass} - Type: {type} - Iter: {i}')
            if xsc == 1:
                plt.xscale('log')
            plt.legend()
            plt.savefig(f'./{mass}/{type}/graphs/{i:03}.png')
            #plt.show()
            plt.close()

            #with open(f"./{mass}/{type}/optimal_widths-{mass}_{type}.txt", 'r') as file_r:
            #    nlines = len(file_r.readlines())
            #print(nlines)
            if i != 1:
                file = open(f"./{mass}/{type}/optimal_widths-{mass}_{type}.txt", "a")
            else:
                file = open(f"./{mass}/{type}/optimal_widths-{mass}_{type}.txt", "w")
            file.write(f"{mass}\t{type}\t{init[pos]:.6e}\t{i}\n")
            file.close()
            #os.system(f'sed -i "{i}s/.*/'f'{mass}\t{type}\t{init[pos]:.6e}\t{i}''/" '
            #          f'"./{mass}/{type}/optimal_widths-{mass}_{type}.txt"')
            i+=1

        '''
        with open("optimal_widths.txt", 'r') as file_r:
            nlines = len(file_r.readlines())
        #print(nlines)
        if nlines == (nlines_mass[mass]+nlines_type[type]) :
            file = open("optimal_widths.txt", "a")
        else:
            file = open("optimal_widths.txt", "w")

        #file = open("optimal_widths.txt", "a")
        file.write(f"{mass}\t{type}\t{init[pos]:.6e}\t{i-1}\n")
        file.close()
        '''
        os.system(f'sed -i "{nlines_mass[mass]+nlines_type[type]+1}s/.*/'f'{mass}\t{type}\t{init[pos]:.6e}\t{i-1}''/" '
                  f'"optimal_widths.txt"')
        print(f'\nOPTIMAL WIDTH: {mass}\t{type}\t{init[pos]:.6e}')
