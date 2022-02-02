import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def amax2(m):
    try:
        d = np.amax(m)
    except ValueError:
        d = 0

    return d


def amin2(m):
    try:
        f = np.amin(m)
    except ValueError:
        f = 0

    return f


ATLASdet_radius= 1.775
ATLASdet_semilength = 4.050
nbins = 50

folder_list = ['MET','eta','pt','pz','z_origin','ET','rel_tof']
folder_color = ['Brown','Purple','Orange','Green','Blue','Red','Black']
features =  list(zip(folder_list,folder_color))
met_marks = [0,20,50,75,np.inf]
met_labels = ['BKG','CR1','CR2','SR']
metcolors= ['C3','C2','C1','C0']

types = ['VBF','GF']
names = {'VBF':'Vector Boson Fusion','GF':'Gluon Fusion'}
cards = [13,14,15]
mass = {13:50,14:30,15:10}
tevs = [8]

for tev in tevs[:]:

    maxet = 0
    juices = {i: {j:'' for j in types} for i in cards}
    #print(juices)
    for type in types[:1]:
        for card in cards[:1]:

            print(f'RUNNING: {type} {card} {tev} ')

            ip = 1
            case = f"./cases/{type}/{card}/{tev}/"
            #destiny_paper = f"./cases/"
            destiny_paper = f"./paper/"
            destiny_filters = case + 'images/ATLAS/filters/'
            destiny_cutflow = case + 'images/ATLAS/cutflow/'
            df_in = f'data/clean/photon_df-{type}_{card}_{tev}.xlsx'
            jets_in = f'data/clean/jets-{type}_{card}_{tev}_pj.pickle'
            Path(destiny_filters).mkdir(parents=True, exist_ok=True)
            Path(destiny_paper).mkdir(parents=True, exist_ok=True)

            df = pd.read_excel(df_in, header=[0,1],index_col=0)

            df['r'] = (df['r']/1000)/ATLASdet_radius
            df['z'] = np.abs(df['z']/1000)/ATLASdet_semilength
            '''
            events = df.shape[0]
            max_ed = int(np.ceil(amax2(df['r'].to_numpy())))

            if max_ed > 9:
                limit = max_ed
            else:
                limit = 9
            
            edges=[-0.1,0.25,0.5,0.75,1,5,limit]
            labels = list(range(len(edges)-1))
            df['dr',1]=pd.cut(df['r',1],bins=edges,labels=labels)
            df['dr',2]=pd.cut(df['r',2],bins=edges,labels=labels)

            prejuice = pd.crosstab(index=df['dr',2],columns=df['dr',1])
            juice=prejuice.values

            for r in labels:
                try:
                    prejuice.loc[r]
                except KeyError:
                    juice = np.insert(juice,r,0,axis=0)

            for c in labels:
                try:
                    prejuice.loc[:,c]
                except KeyError:
                    juice = np.insert(juice,c,0,axis=1)

            #print(juice)
            vmax = np.max(juice)
            vmin = 0
            juice1 = juice.copy().astype('float64')
            juice1[juice1==0] = vmin
            premask = np.full_like(juice,True,bool)
            mask1=premask.copy()
            mask2=premask.copy()
            mask3=premask.copy()
            mask1[:4,:4] = False
            mask2[:4,4:] = False
            mask2[4:,:4] = False
            mask3[4:,4:] = False
            #print(juice)

            fig, ax = plt.subplots(figsize=(8,8))
            for mask,colors in [(mask1,'Blues'),(mask2,'Greens'),(mask3,'Reds')]:
                block = np.ma.array(juice1,mask=mask)
                #im = ax.imshow(block, cmap=colors,norm=mat.colors.LogNorm(vmin=vmin,vmax=vmax),
                 #              origin='lower')
                im = ax.imshow(block, cmap=colors,vmin=vmin, vmax=vmax,origin='lower')

            ticks = np.array(range(len(edges))) - 0.5
            ticklabels = [f'{item} r' if item > 0 else '0' for item in edges]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)
            ax.grid(which='major', color='w', linestyle='-', linewidth=3)
            for i in range(juice.shape[1]):
                for j in range(juice.shape[0]):
                    number = juice[j,i]
                    if number <= (vmax + vmin)/2:
                    #if np.log(number) <= (np.log(vmax) + np.log(vmin)) / 2:
                        c = 'k'
                    else:
                        c = 'w'
                    ax.text(i,j,f'{number}',horizontalalignment="center",
                            verticalalignment="center",color=c)
            plt.xlabel('Photon 1')
            plt.ylabel('Photon 2')
            plt.title(f'Filtro {ip}a: Dentro del radio\n{events} eventos')
            plt.savefig(destiny_filters+f'filter{ip}a_2D.png')
            #plt.show()
            plt.close()
            '''

            ################ Filter 1b ####################

            df = df.loc[(df.r[1]<1) & (df.r[2]<1)]
            events = df.shape[0]
            max_ed = int(np.ceil(amax2(df['z'].to_numpy())))
            '''
            if max_ed > 20:
                limit = max_ed
            else:
                limit = 20

            edges=[-0.1,0.25,0.5,0.75,1,5,10,limit]
            labels = list(range(len(edges)-1))
            df['dz',1]=pd.cut(df['z',1],bins=edges,labels=labels)
            df['dz',2]=pd.cut(df['z',2],bins=edges,labels=labels)

            prejuice = pd.crosstab(index=df['dz',2],columns=df['dz',1])
            juice=prejuice.values
            
            for r in labels:
                try:
                    prejuice.loc[r]
                except KeyError:
                    juice = np.insert(juice,r,0,axis=0)

            for c in labels:
                try:
                    prejuice.loc[:,c]
                except KeyError:
                    juice = np.insert(juice,c,0,axis=1)

            #print(juice)
            vmax = np.max(juice)
            vmin = 0
            juice1 = juice.copy().astype('float64')
            juice1[juice1==0] = vmin
            premask = np.full_like(juice,True,bool)
            mask1=premask.copy()
            mask2=premask.copy()
            mask3=premask.copy()
            mask1[:4,:4] = False
            mask2[:4,4:] = False
            mask2[4:,:4] = False
            mask3[4:,4:] = False
            #print(juice)

            fig, ax = plt.subplots(figsize=(8,8))
            for mask,colors in [(mask1,'Blues'),(mask2,'Greens'),(mask3,'Reds')]:
                block = np.ma.array(juice1,mask=mask)
                #im = ax.imshow(block, cmap=colors,norm=mat.colors.LogNorm(vmin=vmin,vmax=vmax),
                 #              origin='lower')
                im = ax.imshow(block, cmap=colors,vmin=vmin, vmax=vmax,origin='lower')

            ticks = np.array(range(len(edges))) - 0.5
            ticklabels = [f'{item} z' if item > 0 else '0' for item in edges]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)
            ax.grid(which='major', color='w', linestyle='-', linewidth=3)
            for i in range(juice.shape[1]):
                for j in range(juice.shape[0]):
                    number = juice[j,i]
                    if number <= (vmax + vmin)/2:
                    #if np.log(number) <= (np.log(vmax) + np.log(vmin)) / 2:
                        c = 'k'
                    else:
                        c = 'w'
                    ax.text(i,j,f'{number}',horizontalalignment="center",
                            verticalalignment="center",color=c)
            plt.xlabel('Photon 1')
            plt.ylabel('Photon 2')
            plt.title(f'Filtro {ip}b: Dentro de la longitud z\n{events} eventos')
            plt.savefig(destiny_filters+f'filter{ip}b_2D.png')
            #plt.show()
            plt.close()
            '''
            ip += 1
            ################ Jet Filters ####################
            ################ 1
            #############hmore than one jet
            df_jets = pd.read_pickle(jets_in)
            df = df.loc[(df.z[1]<1) & (df.z[2]<1)]
            events = df.shape[0]
            inicial=events
            print(f'Eventos dentro del detector:'.ljust(50),events)
            df_jets=df_jets.loc[df.index]
            jet_nums = df_jets.groupby(['Event']).size()
            plus2 = jet_nums[jet_nums>1].index
            #lead2 = df_jets.query('id in [0,1]')
            #lead2 = lead2.reset_index().pivot(index='Event', columns='id')
            #print(plus2)
            ip+=1

            #######################################
            ####### 2
            ############# PT de ambos jets >30 GeV
            df_jets = df_jets.loc[plus2]
            events = plus2.size
            print(f'Eventos con mas de un jet:'.ljust(50), events)
            max_ed = int(np.ceil(amax2(df_jets['pt'].to_numpy())))
            # print(events)

            df_jets2 = df_jets.query('id in [0,1]')
            df_jets2 = df_jets2.pivot_table(index="Event", columns="id")
            # print(df_jets2)

            if max_ed > 150:
                limit = max_ed
            else:
                limit = 150

            edges = [-0.1, 30, 60, 90, 120, limit]
            labels = list(range(len(edges) - 1))
            df_jets2['dpt', 0] = pd.cut(df_jets2['pt', 0], bins=edges, labels=labels)
            df_jets2['dpt', 1] = pd.cut(df_jets2['pt', 1], bins=edges, labels=labels)

            prejuice = pd.crosstab(index=df_jets2['dpt', 1], columns=df_jets2['dpt', 0])
            ##print(prejuice)

            juice = prejuice.values

            for r in labels:
                try:
                    prejuice.loc[r]
                except KeyError:
                    juice = np.insert(juice, r, 0, axis=0)

            for c in labels:
                try:
                    prejuice.loc[:, c]
                except KeyError:
                    juice = np.insert(juice, c, 0, axis=1)

            # print(juice)
            vmax = np.max(juice)
            vmin = 0
            juice1 = juice.copy().astype('float64')
            juice1[juice1 == 0] = vmin
            premask = np.full_like(juice, True, bool)
            mask1 = premask.copy()
            mask2 = premask.copy()
            mask3 = premask.copy()
            mask1[1:, 1:] = False
            mask2[:1, 1:] = False
            mask2[1:, :1] = False
            mask3[:1, :1] = False
            # print(juice)

            fig, ax = plt.subplots(figsize=(8, 8))
            for mask, colors in [(mask1, 'Blues'), (mask2, 'Greens'), (mask3, 'Reds')]:
                block = np.ma.array(juice1, mask=mask)
                # im = ax.imshow(block, cmap=colors,norm=mat.colors.LogNorm(vmin=vmin,vmax=vmax),
                #              origin='lower')
                im = ax.imshow(block, cmap=colors, vmin=vmin, vmax=vmax, origin='lower')

            ticks = np.array(range(len(edges))) - 0.5
            ticklabels = [f'{item}' if item > 0 else '0' for item in edges]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)
            ax.grid(which='major', color='w', linestyle='-', linewidth=3)
            for i in range(juice.shape[1]):
                for j in range(juice.shape[0]):
                    number = juice[j, i]
                    if number <= (vmax + vmin) / 2:
                        # if np.log(number) <= (np.log(vmax) + np.log(vmin)) / 2:
                        c = 'k'
                    else:
                        c = 'w'
                    ax.text(i, j, f'{number}', horizontalalignment="center",
                            verticalalignment="center", color=c)
            plt.xlabel('Jet 1')
            plt.ylabel('Jet 2')
            plt.title(f'Filtro {ip}: Jets con PT > 30 GeV\n{events} eventos')
            plt.savefig(destiny_filters + f'filter{ip}_2D.png')
            # plt.show()
            plt.close()

            ip += 1

            ##############################################
            ################
            ############ eta1.eta2 <0 y |eta| <5
            df_jets2 = df_jets2[df_jets2.pt[1] > 30]
            df_jets = df_jets.loc[df_jets2.index]
            events = df_jets2.shape[0]
            print(f'Eventos con lead jets pt > 30 GeV:'.ljust(50), events)
            max_ed = int(np.ceil(amax2(df_jets2['eta'].to_numpy())))
            # print(max_ed)
            # sys.exit()

            if max_ed > 9:
                limit = max_ed
            else:
                limit = 9

            edges = [-limit, -5, -3, -1, 0, 1, 3, 5, limit]
            labels = list(range(len(edges) - 1))
            df_jets2['deta', 0] = pd.cut(df_jets2['eta', 0], bins=edges, labels=labels)
            df_jets2['deta', 1] = pd.cut(df_jets2['eta', 1], bins=edges, labels=labels)

            prejuice = pd.crosstab(index=df_jets2['deta', 1], columns=df_jets2['deta', 0])
            ##print(prejuice)

            juice = prejuice.values

            for r in labels:
                try:
                    prejuice.loc[r]
                except KeyError:
                    juice = np.insert(juice, r, 0, axis=0)

            for c in labels:
                try:
                    prejuice.loc[:, c]
                except KeyError:
                    juice = np.insert(juice, c, 0, axis=1)

            # print(juice)
            vmax = np.max(juice)
            vmin = 0
            juice1 = juice.copy().astype('float64')
            juice1[juice1 == 0] = vmin
            premask = np.full_like(juice, True, bool)
            mask1 = premask.copy()
            mask2 = premask.copy()
            mask3 = premask.copy()
            mask1[1:7, 1:7] = False
            mask2 = premask ^ mask1
            for r in [0, len(edges) - 2]:
                for c in [0, len(edges) - 2]:
                    mask3[r, c] = False
            mask3[:4, :4] = False
            mask3[4:, 4:] = False

            # print(juice)

            fig, ax = plt.subplots(figsize=(8, 8))
            for mask, colors in [(mask1, 'Blues'), (mask2, 'Greens'), (mask3, 'Reds')]:
                block = np.ma.array(juice1, mask=mask)
                # im = ax.imshow(block, cmap=colors,norm=mat.colors.LogNorm(vmin=vmin,vmax=vmax),
                #              origin='lower')
                im = ax.imshow(block, cmap=colors, vmin=vmin, vmax=vmax, origin='lower')

            ticks = np.array(range(len(edges))) - 0.5
            ticklabels = [f'{item}' for item in edges]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)
            ax.grid(which='major', color='w', linestyle='-', linewidth=3)
            for i in range(juice.shape[1]):
                for j in range(juice.shape[0]):
                    number = juice[j, i]
                    if number <= (vmax + vmin) / 2:
                        # if np.log(number) <= (np.log(vmax) + np.log(vmin)) / 2:
                        c = 'k'
                    else:
                        c = 'w'
                    ax.text(i, j, f'{number}', horizontalalignment="center",
                            verticalalignment="center", color=c)
            plt.xlabel('Jet 1')
            plt.ylabel('Jet 2')
            plt.title(f'Filtro {ip}: Jets con restricciones en eta\n{events} eventos')
            plt.savefig(destiny_filters + f'filter{ip}_2D.png')
            # plt.show()
            plt.close()

            ip += 1

            ##############################################
            ################
            ############ |D eta| > 4.2
            df_jets2 = df_jets2[(abs(df_jets2.eta[0]) < 5) & (abs(df_jets2.eta[1]) < 5) &
                                ((df_jets2.eta[0] * df_jets2.eta[1]) < 0)]
            events = df_jets2.shape[0]
            print(f'Eventos con restricciones de eta:'.ljust(50), events)
            deta_jet = df_jets2[abs(df_jets2.eta[0] - df_jets2.eta[1]) > 4.2].index
            # print(deta_jet)
            ip += 1

            #######################################
            #######
            ############# HT > 200 GeV
            df_jets2 = df_jets2.loc[deta_jet]
            df_jets = df_jets.loc[deta_jet]
            events = deta_jet.size
            print(f'Eventos con delta eta entre lead jets > 4.2:'.ljust(50), events)
            pts = df_jets.query('pt > 30').groupby(['Event']).sum()
            ht = pts[pts > 200].index
            # print(ht.size)
            ip += 1

            #######################################
            #######
            ############# MI > 750 GeV
            df_jets = df_jets.loc[ht]
            df_jets2 = df_jets2.loc[ht]
            events = ht.size
            print(f'Eventos con HT mayor de 200 GeV:'.ljust(50),events)
            inv_m = np.sqrt((df_jets2.E[0] + df_jets2.E[1]) ** 2 -
                            ((df_jets2.px[0] + df_jets2.px[1]) ** 2 + (df_jets2.py[0] + df_jets2.py[1]) ** 2 +
                             (df_jets2.pz[0] + df_jets2.pz[1]) ** 2))
            inv_m = inv_m[inv_m > 750].index
            ip += 1

            #######################################
            #######
            ############# |D phi| >2.3
            df_jets2 = df_jets2.loc[inv_m]
            df_jets = df_jets.loc[inv_m]
            events = inv_m.size
            print(f'Eventos con masa invariante de jets > 750 GeV'.ljust(50), events)
            dphi_jet = df_jets2[abs(df_jets2.phi[0] - df_jets2.phi[1]) > 2.3].index
            ip+=1

            #######################################
            #######
            ############# |min(D (MTP,4J))| > 0.5
            df_jets = df_jets.loc[dphi_jet].query('id < 4')
            df = df.loc[dphi_jet]
            events = dphi_jet.size
            print(f'Eventos con delta phi entre lead jets > 2.3'.ljust(50), events)
            delta_phi = abs(df_jets.phi - np.arctan2(df['MPy'][1],df['MPx'][1]))
            delta_phi = delta_phi.groupby('Event').min()
            dphi=delta_phi[delta_phi > 0.5].index
            #print(dphi)
            ip+=1

            ###############Filter N ################
            df = df.loc[dphi]
            events=dphi.size
            final=events
            print(f'Eventos con deltaphi(MPT, 4Jets) > 0.5'.ljust(50),events)
            max_ed = np.ceil(amax2(df['ET'].to_numpy()))

            print(f'\nEventos sobrevivientes a jet cuts'.ljust(50), f' {100 * final / inicial:.2f} %')
            sys.exit()
            if max_ed > 450.0:
                limit = max_ed
            else:
                limit = 450.0

            edges=[-0.1,25.0,50.0,150.0,250.0,350.0,limit]
            labels = list(range(len(edges)-1))
            df['det',1]=pd.cut(df['ET',1],bins=edges,labels=labels)
            df['det',2]=pd.cut(df['ET',2],bins=edges,labels=labels)

            prejuice = pd.crosstab(index=df['det',2],columns=df['det',1])
            juice=prejuice.values

            for r in labels:
                try:
                    prejuice.loc[r]
                except KeyError:
                    juice = np.insert(juice,r,0,axis=0)

            for c in labels:
                try:
                    prejuice.loc[:,c]
                except KeyError:
                    juice = np.insert(juice,c,0,axis=1)


            juices[card][type] = [edges,labels,juice]

            juice = 100 * juice / events
            edges = edges[:-2]
            labels = labels[:-2]
            juice = juice[:-2,:-2]

            #print(juice)

            vmax = np.max(juice)
            vmin = 0
            juice1 = juice.copy().astype('float64')
            juice1[juice1==0] = vmin

            premask = np.full_like(juice,True,bool)
            mask1=premask.copy()
            mask2=premask.copy()
            mask3=premask.copy()
            mask1[:2,:2] = False
            mask2[:2,2:] = False
            mask2[2:,:2] = False
            mask3[2:,2:] = False

            big_mask = np.triu(np.ones_like(juice,dtype=bool))
            mask1 = np.logical_or(mask1,np.logical_not(big_mask))
            mask2 = np.logical_or(mask2, np.logical_not(big_mask))
            mask3 = np.logical_or(mask3, np.logical_not(big_mask))
            #print(big_mask)

            maxet = np.maximum(maxet,vmax)
            '''
            fig, ax = plt.subplots(figsize=(8,8))
            for mask,colors in [(mask1,'Reds'),(mask2,'Greens'),(mask3,'Blues')]:
                block = np.ma.array(juice1,mask=mask)
                #im = ax.imshow(block, cmap=colors,norm=mat.colors.LogNorm(vmin=vmin,vmax=vmax),
                 #              origin='lower')
                im = ax.imshow(block, cmap=colors,vmin=vmin, vmax=vmax,origin='lower')

            ticks = np.array(range(len(edges))) - 0.5
            ticklabels = [f'{item}' if item > 0 else '0' for item in edges]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels,fontsize=24)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels,fontsize=24)
            ax.grid(which='major', color='w', linestyle='-', linewidth=3)
            for j in range(juice.shape[0]):
                for i in range(juice.shape[1]):
                    if big_mask[j,i]:
                        number = juice[j,i]
                        if number <= (vmax + vmin)/2:
                        #if np.log(number) <= (np.log(vmax) + np.log(vmin)) / 2:
                            c = 'k'
                        else:
                            c = 'w'
                        ax.text(i,j,f'{number:.2f}%',horizontalalignment="center",
                                verticalalignment="center",color=c, fontsize=20)
            plt.xlabel('Photon 1\'s ET [GeV]', fontsize=26)
            plt.ylabel('Photon 2\'s ET [GeV]', fontsize=26)
            plt.title(f'{type} - {mass[card]} GeV - {tev} TeV\nFilter 2: ET > 50 GeV\n({events} Events)',
                      fontsize=26)
            plt.savefig(destiny_filters+'filter2_2D.png')
            #plt.show()
            plt.close()
            '''
            '''            
            ################### Filtro 3 #################
            df = df.loc[(df.ET[1]>50) & (df.ET[2] >50)]
            events = df.shape[0]
            max_ed = np.ceil(amax2(df['rel_tof'].to_numpy()))

            if max_ed > 6.0:
                limit = max_ed
            else:
                limit = 6.0

            edges=[-0.1,0.5,1.0,1.5,2.0,4.0,limit]
            labels = list(range(len(edges)-1))
            df['drt',1]=pd.cut(df['rel_tof',1],bins=edges,labels=labels)
            df['drt',2]=pd.cut(df['rel_tof',2],bins=edges,labels=labels)

            prejuice = pd.crosstab(index=df['drt',2],columns=df['drt',1])
            juice=prejuice.values

            for r in labels:
                try:
                    prejuice.loc[r]
                except KeyError:
                    juice = np.insert(juice,r,0,axis=0)

            for c in labels:
                try:
                    prejuice.loc[:,c]
                except KeyError:
                    juice = np.insert(juice,c,0,axis=1)

            #print(juice)
            vmax = np.max(juice)
            vmin = 0
            juice1 = juice.copy().astype('float64')
            juice1[juice1==0] = vmin
            premask = np.full_like(juice,True,bool)
            mask1=premask.copy()
            mask2=premask.copy()
            mask3=premask.copy()
            mask1[:5,:5] = False
            mask2[:5,5:] = False
            mask2[5:,:5] = False
            mask3[5:,5:] = False
            #print(juice)

            fig, ax = plt.subplots(figsize=(8,8))
            for mask,colors in [(mask1,'Blues'),(mask2,'Greens'),(mask3,'Reds')]:
                block = np.ma.array(juice1,mask=mask)
                #im = ax.imshow(block, cmap=colors,norm=mat.colors.LogNorm(vmin=vmin,vmax=vmax),
                 #              origin='lower')
                im = ax.imshow(block, cmap=colors,vmin=vmin, vmax=vmax,origin='lower')

            ticks = np.array(range(len(edges))) - 0.5
            ticklabels = [f'{item}' if item > 0 else '0' for item in edges]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)
            ax.grid(which='major', color='w', linestyle='-', linewidth=3)
            for i in range(juice.shape[1]):
                for j in range(juice.shape[0]):
                    number = juice[j,i]
                    if number <= (vmax + vmin)/2:
                    #if np.log(number) <= (np.log(vmax) + np.log(vmin)) / 2:
                        c = 'k'
                    else:
                        c = 'w'
                    ax.text(i,j,f'{number}',horizontalalignment="center",
                            verticalalignment="center",color=c)
            plt.xlabel('Photon 1')
            plt.ylabel('Photon 2')
            plt.title(f'Filtro 3: TOF relativo menor a 4 ns\n{events} eventos')
            plt.savefig(destiny_filters+'filter3_2D.png')
            #plt.show()
            plt.close()

            ################### Filtro 4 ##################
            df = df.loc[(df.rel_tof[1]<4) & (df.rel_tof[2]<4)]
            events = df.shape[0]

            max_ed = np.ceil(amax2(np.abs(df['eta'].to_numpy())))

            if max_ed > 3.5:
                limit = max_ed
            else:
                limit = 3.5

            edges=[-1*limit,-2.37,-1.7,-1.0,-0.3,0.3,1.0,1.7,2.37,limit]
            labels = list(range(len(edges)-1))
            df['deta',1]=pd.cut(df['eta',1],bins=edges,labels=labels)
            df['deta',2]=pd.cut(df['eta',2],bins=edges,labels=labels)

            prejuice = pd.crosstab(index=df['deta',2],columns=df['deta',1])
            juice=prejuice.values

            for r in labels:
                try:
                    prejuice.loc[r]
                except KeyError:
                    juice = np.insert(juice,r,0,axis=0)

            for c in labels:
                try:
                    prejuice.loc[:,c]
                except KeyError:
                    juice = np.insert(juice,c,0,axis=1)

            #print(juice)
            vmax = np.max(juice)
            vmin = 0
            juice1 = juice.copy().astype('float64')
            juice1[juice1==0] = vmin
            premask = np.full_like(juice,True,bool)
            mask1=premask.copy()
            mask2=premask.copy()
            mask3=premask.copy()
            mask1[1:8,1:8] = False
            mask2 = premask ^ mask1
            for r in [0,len(edges)-2]:
                for c in [0, len(edges) - 2]:
                    mask3[r,c] = False

            fig, ax = plt.subplots(figsize=(8,8))
            for mask,colors in [(mask1,'Blues'),(mask2,'Greens'),(mask3,'Reds')]:
                block = np.ma.array(juice1,mask=mask)
                #im = ax.imshow(block, cmap=colors,norm=mat.colors.LogNorm(vmin=vmin,vmax=vmax),
                 #              origin='lower')
                im = ax.imshow(block, cmap=colors,vmin=vmin, vmax=vmax,origin='lower')

            ticks = np.array(range(len(edges))) - 0.5
            ticklabels = [f'{item}' for item in edges]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)
            ax.grid(which='major', color='w', linestyle='-', linewidth=3)
            for i in range(juice.shape[1]):
                for j in range(juice.shape[0]):
                    number = juice[j,i]
                    if number <= (vmax + vmin)/2:
                    #if np.log(number) <= (np.log(vmax) + np.log(vmin)) / 2:
                        c = 'k'
                    else:
                        c = 'w'
                    ax.text(i,j,f'{number}',horizontalalignment="center",
                            verticalalignment="center",color=c)
            plt.xlabel('Photon 1')
            plt.ylabel('Photon 2')
            plt.title(f'Filtro 4: Eta menor a 2.37 \n{events} eventos')
            plt.savefig(destiny_filters+'filter4_2D.png')
            #plt.show()
            plt.close()

            ###################### Filtro 5 ######################
            df = df.loc[(np.abs(df.eta[1])<2.37) & (np.abs(df.eta[2])<2.37)]
            events = df.shape[0]

            edges=[-2.37,-1.52,-1.37,0,1.37,1.52,2.37]
            labels = list(range(len(edges)-1))
            df['deta2',1]=pd.cut(df['eta',1],bins=edges,labels=labels,right=False)
            df['deta2',2]=pd.cut(df['eta',2],bins=edges,labels=labels,right=False)

            prejuice = pd.crosstab(index=df['deta2',2],columns=df['deta2',1])
            juice=prejuice.values

            for r in labels:
                try:
                    prejuice.loc[r]
                except KeyError:
                    juice = np.insert(juice,r,0,axis=0)

            for c in labels:
                try:
                    prejuice.loc[:,c]
                except KeyError:
                    juice = np.insert(juice,c,0,axis=1)

            #print(juice)
            vmax = np.max(juice)
            vmin = 0
            juice1 = juice.copy().astype('float64')
            juice1[juice1==0] = vmin
            premask = np.full_like(juice,True,bool)
            mask1=premask.copy()
            mask2=premask.copy()
            mask3=premask.copy()
            mask1[2:4,2:4] = False
            mask2[0,2:4] = False
            mask2[2:4,0] = False
            mask2[-1,2:4] = False
            mask2[2:4,-1] = False
            mask3 = premask ^(mask1 & mask2)

            fig, ax = plt.subplots(figsize=(8,8))
            for mask,colors in [(mask1,'Blues'),(mask2,'Blues'),(mask3,'Reds')]:
                block = np.ma.array(juice1,mask=mask)
                #im = ax.imshow(block, cmap=colors,norm=mat.colors.LogNorm(vmin=vmin,vmax=vmax),
                 #              origin='lower')
                im = ax.imshow(block, cmap=colors,vmin=vmin, vmax=vmax,origin='lower')

            ticks = np.array(range(len(edges))) - 0.5
            ticklabels = [f'{item}' for item in edges]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)
            ax.grid(which='major', color='w', linestyle='-', linewidth=3)
            for i in range(juice.shape[1]):
                for j in range(juice.shape[0]):
                    number = juice[j,i]
                    if number <= (vmax + vmin)/2:
                    #if np.log(number) <= (np.log(vmax) + np.log(vmin)) / 2:
                        c = 'k'
                    else:
                        c = 'w'
                    ax.text(i,j,f'{number}',horizontalalignment="center",
                            verticalalignment="center",color=c)
            plt.xlabel('Photon 1')
            plt.ylabel('Photon 2')
            plt.title(f'Filtro 5 - 6: Ninguno en el intersec y >=1 en el barrel\n{events} eventos')
            plt.savefig(destiny_filters+'filter5_2D.png')
            #plt.show()
            plt.close()

            '''

    '''
    vmax = maxet
    #print(vmax)

    for card in list(juices.keys())[:]:
        for type in list(juices[card].keys())[:]:

            edges, labels, juice = juices[card][type]
            events = np.sum(juice)
            juice = 100 * juice / events
            edges = edges[:-2]
            labels = labels[:-2]
            juice = juice[:-2, :-2]

            vmin = 0
            juice1 = juice.copy().astype('float64')
            juice1[juice1 == 0] = vmin

            premask = np.full_like(juice, True, bool)
            mask1 = premask.copy()
            mask2 = premask.copy()
            mask3 = premask.copy()
            mask1[:2, :2] = False
            mask2[:2, 2:] = False
            mask2[2:, :2] = False
            mask3[2:, 2:] = False

            big_mask = np.triu(np.ones_like(juice, dtype=bool))
            mask1 = np.logical_or(mask1, np.logical_not(big_mask))
            mask2 = np.logical_or(mask2, np.logical_not(big_mask))
            mask3 = np.logical_or(mask3, np.logical_not(big_mask))
            # print(big_mask)

            fig, ax = plt.subplots(figsize=(8, 8))
            for mask, colors in [(mask1, 'Reds'), (mask2, 'Greens'), (mask3, 'Blues')]:
                block = np.ma.array(juice1, mask=mask)
                # im = ax.imshow(block, cmap=colors,norm=mat.colors.LogNorm(vmin=vmin,vmax=vmax),
                #              origin='lower')
                im = ax.imshow(block, cmap=colors, vmin=vmin, vmax=vmax, origin='lower')

            ticks = np.array(range(len(edges))) - 0.5
            ticklabels = [f'{int(item)}' if item > 0 else '0' for item in edges]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels, fontsize=20)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels, fontsize=20)
            ax.grid(which='major', color='w', linestyle='-', linewidth=3)
            for j in range(juice.shape[0]):
               for i in range(juice.shape[1]):
                   if big_mask[j, i]:
                       number = juice[j, i]
                       if number <= (vmax + vmin) / 2:
                           # if np.log(number) <= (np.log(vmax) + np.log(vmin)) / 2:
                           c = 'k'
                       else:
                           c = 'w'
                       ax.text(i, j, f'{number:.2f}%', horizontalalignment="center",
                               verticalalignment="center", color=c, fontsize=22)
            ax.text(0.05,0.76,f"{names[type]}\n$\mathregular{{M_{{h}}}}$ = {mass[card]} GeV\n{events} Events",
                    transform=ax.transAxes,linespacing=2,fontsize=22)
            plt.xlabel('$\mathregular{E_{T}^{\gamma1}}$ [GeV]', fontsize=24)
            plt.ylabel('$\mathregular{E_{T}^{\gamma2}}$ [GeV]', fontsize=24)
#            plt.title(f'{type} - {mass[card]} GeV - {tev} TeV\nFilter 2: ET > 50 GeV\n({events} Events)',
#                      fontsize=13)
            plt.savefig(destiny_paper + f'{card}_{type}_filter2_2D.png',bbox_inches='tight')
            # plt.show()
            plt.close()
    
    '''
