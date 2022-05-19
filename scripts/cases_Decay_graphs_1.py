import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from my_funcs import my_arctan

def format_exponent(ax, axis='y', size=13):
    # Change the ticklabel format to scientific format
    ax.ticklabel_format(axis=axis, style='sci', scilimits=(-2, 2))

    # Get the appropriate axis
    if axis == 'y':
        ax_axis = ax.yaxis
        x_pos = 0.0
        y_pos = 1.0
        horizontalalignment = 'left'
        verticalalignment = 'bottom'
    else:
        ax_axis = ax.xaxis
        x_pos = 1.0
        y_pos = -0.05
        horizontalalignment = 'right'
        verticalalignment = 'top'

    # Run plt.tight_layout() because otherwise the offset text doesn't update
    plt.tight_layout()
    ##### THIS IS A BUG
    ##### Well, at least it's sub-optimal because you might not
    ##### want to use tight_layout(). If anyone has a better way of
    ##### ensuring the offset text is updated appropriately
    ##### please comment!

    # Get the offset value
    offset = ax_axis.get_offset_text().get_text()

    if len(offset) > 0:
        # Get that exponent value and change it into latex format
        minus_sign = u'\u2212'
        expo = np.float(offset.replace(minus_sign, '-').split('e')[-1])
        offset_text = r'x$\mathregular{10^{%d}}$' % expo
    else:
        offset_text = "   "
    # Turn off the offset text that's calculated automatically
    ax_axis.offsetText.set_visible(False)

    # Add in a text box at the top of the y axis
    ax.text(x_pos, y_pos, offset_text, fontsize=size, transform=ax.transAxes,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment)

    return ax

def plot_params_1(filter_num,filter_desc):
    met_max = amax2(df['MET',1])
    if np.isnan(met_max):
        met_max = 0
    secciones = [df[df['MET_bin',1]==i] for i in met_labels]
    for param, color in features[:]:
        if param in (['MET']):
            metbins = list(range(0,int(met_max)+5,5))
            fig, ax = plt.subplots(figsize=(10,6))
            ax.hist([dfp[param, 1] for dfp in secciones],color=metcolors,
                    bins=metbins,histtype='step',label = met_labels)
            ax.set_ylim(0.5)
            ax.set_yscale('log')
            plt.legend()
        elif param not in specials:
            pmin = amin2(df[param].to_numpy())
            pmax = amax2(df[param].to_numpy())
            if param in centers:
                pmax = np.max(np.abs([pmin, pmax]))
                pmin = -1 * pmax
            if pmin == pmax:
                pmax += 10/nbins
                pmin -= 10/nbins
            param_bins = np.linspace(pmin,pmax + (pmax - pmin)/nbins,nbins)
            #print(pmax)
            ymax = []
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
            for ix, col in enumerate(axs):
                #print(np.max(df[param,ix+1]))
                col.hist(df[param, ix + 1], histtype='step', bins=param_bins, color=color)
                #print(dummy)
                if param == 'rel_tof' and pmax >= 4:
                    col.axvline(x=4,color='b',linestyle='--')
                col.set_title(f'Photon {ix + 1}')
                col.set_yscale('log')
                ymax.append(col.get_ylim()[1])
            plt.setp(axs, ylim=(0.8, max(ymax)))
        else:
            pmin = amin2(df[param].to_numpy())
            pmax = amax2(df[param].to_numpy())
            if param in centers:
                pmax = np.max(np.abs([pmin, pmax]))
                pmin = -1 * pmax
            if pmin == pmax:
                pmax += 10 / nbins
                pmin -= 10 / nbins
            param_bins = np.linspace(pmin, pmax + (pmax - pmin) / nbins, nbins)
            # print(pmax)
            ymax = []
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
            # print(np.max(df[param,ix+1]))
            axs.hist(df[param, 1], histtype='step', bins=param_bins, color=color)
            # print(dummy)
            axs.set_yscale('log')
            ymax.append(axs.get_ylim()[1])
            plt.setp(axs, ylim=(0.8, max(ymax)))
        plt.suptitle(f'After Filter {filter_num}: {filter_desc}\n{param}',fontsize=14)
        plt.savefig(destiny_cutflow+param+f'/{param}_after_filter{filter_num}.png')
        plt.close()

    return

def plot_params_2(filter_num,filter_desc):
    secciones = [df[df['MET_bin',1]==i] for i in met_labels]
    for param, color in features[:]:
        if param not in (['MET']+specials):
            pmin = amin2(df[param].to_numpy())
            pmax = amax2(df[param].to_numpy())
            if param in centers:
                pmax = np.max(np.abs([pmin,pmax]))
                pmin = -1*pmax
            if pmin == pmax:
                pmax += 10/nbins
                pmin -= 10/nbins
            param_bins = np.linspace(pmin, pmax + (pmax - pmin) / nbins, nbins)
            ymax = []
            ymin = []
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
            for ix, col in enumerate(axs):
                #print(np.max(df[param,ix+1]))
                col.hist([dfp[param, ix+1] for dfp in secciones],color=metcolors,
                    bins=param_bins,stacked=True,label = met_labels)
                #print(dummy)
                if param == 'rel_tof' and pmax >= 4:
                    col.axvline(x=4, color='b', linestyle='--')
                col.set_title(f'Photon {ix + 1}')
                ymax.append(col.get_ylim()[1])
                ymin.append(col.get_ylim()[0])
                col.legend()
            plt.setp(axs,ylim=(min(ymin),max(ymax)))
            plt.suptitle(f'After Filter {filter_num}: {filter_desc}\n{param}',fontsize=14)
            plt.savefig(destiny_cutflow+param+f'/{param}_after_filter{filter_num}_regions.png')
            plt.close()
        elif param in specials:
            pmin = amin2(df[param].to_numpy())
            pmax = amax2(df[param].to_numpy())
            if param in centers:
                pmax = np.max(np.abs([pmin, pmax]))
                pmin = -1 * pmax
            if pmin == pmax:
                pmax += 10 / nbins
                pmin -= 10 / nbins
            param_bins = np.linspace(pmin, pmax + (pmax - pmin) / nbins, nbins)
            ymax = []
            ymin = []
            fig, col = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
            col.hist([dfp[param, 1] for dfp in secciones], color=metcolors,
                         bins=param_bins, stacked=True, label=met_labels)
            ymax.append(col.get_ylim()[1])
            ymin.append(col.get_ylim()[0])
            col.legend()
            plt.setp(axs, ylim=(min(ymin), max(ymax)))
            plt.suptitle(f'After Filter {filter_num}: {filter_desc}\n{param}', fontsize=14)
            plt.savefig(destiny_cutflow + param + f'/{param}_after_filter{filter_num}_regions.png')
            plt.close()
    return

def amax2(m):

    try:
        d = np.amax(m)
    except ValueError:
        d = 1

    return d


def amin2(m):
    try:
        f = np.amin(m)
    except ValueError:
        f = 1

    return f


nbins = 50

specials = ['dphi','mt','dphi_mpt_4jets']
folder_list = ['MET','eta','z_origin','ET','rel_tof']
folder_color = ['Blue']*len(folder_list)

centers = ['eta','z_origin','pz']
features =  list(zip(folder_list,folder_color))
met_marks = [0,20,50,75,np.inf]
met_labels = ['BKG','CR1','CR2','SR']
metcolors= ['C3','C2','C1','C0']


types = ['VBF','GF']
names = {'VBF':'Vector\nBoson\nFusion','GF':'Gluon\nFusion'}
cards = [13,14,15]
mass = {13:50,14:30,15:10}
tevs = [8,13]

for tev in tevs[:1]:
    metdata = {i: {j: '' for j in cards} for i in types}
    metmaxs = {i: 0 for i in types}

    for type in types[:]:
        for card in cards[:]:
            print(f"RUNNING: {type} {card} {tev}")
            ip = 1
            case = f"./cases/{tev}/{type}/{card}/"
            destiny_filters = case+'ATLAS/filters/'
            destiny_cutflow = case+'ATLAS/cutflow/'
            df_in = f'./data/clean/photon_df-{type}_{card}_{tev}.xlsx'
            jets_in = f'data/clean/jets-{type}_{card}_{tev}.pickle'

            Path(destiny_filters).mkdir(parents=True, exist_ok=True)
            for folder in folder_list:
                Path(destiny_cutflow + folder).mkdir(parents=True, exist_ok=True)

            df = pd.read_excel(df_in, header=[0,1],index_col=0)
            #print(df.columns)
            df['MET_bin',1]=pd.cut(df['MET',1],bins=met_marks,labels=met_labels)
            df['MET_bin',2]=pd.cut(df['MET',2],bins=met_marks,labels=met_labels)
            df.to_excel(df_in)

            df['z'] = np.abs(df['z'])
            events = df.shape[0]

            ################### Filter 1a ##############################
            df['pair_r'] = 0
            df.loc[(df['r'][1] < 1.) & (df['r'][2] >= 1.), 'pair_r'] = 1
            df.loc[(df['r'][2] < 1.) & (df['r'][1] >= 1.), 'pair_r'] = 1
            df.loc[(df['r'][1] >= 1.) & (df['r'][2] >= 1.), 'pair_r'] = 2
            # print(df.pair_r.unique())

            rs=[]
            rs.append([df[df.pair_r==i]['r', 1].values for i in range(3)])
            rs.append([df[df.pair_r==i]['r', 2].values for i in range(3)])

            cuts = 5
            delta = 1/cuts
            factor = 1
            maximum = amax2(df['r'].values)
            #print(maximum)
            while factor < maximum:
                factor += 1

            rbins = np.arange(0,factor+delta,delta)
            rticks = np.arange(0,factor+delta*5,delta*5)

            rlabels = range(1,len(rticks))
            rlabels = [f'{int(x/(cuts/5))}r' if x%(cuts/5)==0 else f'{(x/(cuts/5)):.1f}r' for x in rlabels]
            rlabels.insert(0,'0')
            #print(rticks)
            #print(rlabels)

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,8))
            if events > 0:
                for ix,col in enumerate(axs):
                    col.hist(rs[ix],bins=rbins,histtype='step',
                     label=["Both in", "1 in 1 out", "Both out"],color=['b','g','r'])
                    col.set_title(f'Photon {ix+1}')
                    col.set_ylim(0.5,events)
                    col.set_xticks(rticks)
                    col.set_xticklabels(rlabels)
                    col.set_yscale('log')
                    col.legend()
            plt.suptitle(f'Filtro {ip}a: Dentro del radio\n{events} eventos')
            plt.savefig(destiny_filters+f'filter{ip}a_1D.png')
            #plt.show()
            plt.close()

            ################ Filter 1b ####################

            df = df.loc[df.pair_r == 0]

            events = df.shape[0]
            df['pair_z'] = 0
            df.loc[(df['z'][1] < 1.) & (df['z'][2] >= 1.), 'pair_z'] = 1
            df.loc[(df['z'][2] < 1.) & (df['z'][1] >= 1.), 'pair_z'] = 1
            df.loc[(df['z'][1] >= 1.) & (df['z'][2] >= 1.), 'pair_z'] = 2
            # print(df.loc[:,['z','pair_z']])

            rs=[]
            rs.append([df[df.pair_z==i]['z', 1].values for i in range(3)])
            rs.append([df[df.pair_z==i]['z', 2].values for i in range(3)])

            cuts =  30/(1+abs(amax2(df['z'].values) - amin2(df['z'].values)))
            delta = 1/cuts
            factor = 1
            maximum = amax2(df['z'].values)
            #print(maximum)

            while factor < maximum:
                factor += 1

            jump = 8
            rbins = np.arange(0,factor+delta,delta)
            rticks = np.arange(0,factor+delta*jump,delta*jump)
            #print(rticks)
            rlabels = range(1,len(rticks))
            rlabels = [f'{int(x/(cuts/jump))}z' if x%(cuts/jump)==0 else f'{(x/(cuts/jump)):.1f}z' for x in rlabels]
            rlabels.insert(0,'0')
            #print(rticks)
            #print(rlabels)

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,8))
            if events > 0:
                for ix,col in enumerate(axs):
                    col.hist(rs[ix],bins=rbins,histtype='step',
                     label=["Both in", "1 in 1 out", "Both out"],color=['b','g','r'])
                    col.set_title(f'Photon {ix+1}')
                    col.set_ylim(0.5,events)
                    col.set_xticks(rticks)
                    col.set_xticklabels(rlabels)
                    col.set_yscale('log')
                    col.legend()
            plt.suptitle(f'Filtro {ip}b: Dentro de la longitud z\n{events} eventos')
            plt.savefig(destiny_filters+f'filter{ip}b_1D.png')
            #plt.show()
            plt.close()

            # FILTRO 1 - CUTFLOW
            # df[df.z_origin > 20000].z_origin = 20000
            df.z_origin = np.clip(df.z_origin, -4000, 4000)
            df.rel_tof = np.clip(df.rel_tof, -10, 10)
            df.pz = np.clip(df.pz, -1000, 1000)

            df = df.loc[df.pair_z == 0]
            events = df.shape[0]


            #plot_params_1(ip,f'Dentro del detector - {events} eventos')
            #plot_params_2(ip,f'Dentro del detector - {events} eventos')

            ip+=1

            ################### Filtro 4 ##################
            df['abs_eta', 1] = np.abs(df['eta', 1])
            df['abs_eta', 2] = np.abs(df['eta', 2])
            df['pair_eta'] = 0
            df.loc[(df['abs_eta'][1] < 2.37) & (df['abs_eta'][2] >= 2.37), 'pair_eta'] = 1
            df.loc[(df['abs_eta'][2] < 2.37) & (df['abs_eta'][1] >= 2.37), 'pair_eta'] = 1
            df.loc[(df['abs_eta'][1] >= 2.37) & (df['abs_eta'][2] >= 2.37), 'pair_eta'] = 2
            # print(df.loc[:,['abs_eta','pair_eta']])

            rs = []
            rs.append([df[df.pair_eta == i]['abs_eta', 1].values for i in range(3)])
            rs.append([df[df.pair_eta == i]['abs_eta', 2].values for i in range(3)])

            cuts = 10
            delta = 1 / cuts
            factor = 1
            maximum = amax2(df['abs_eta'].values)
            # print(maximum)

            while factor < maximum:
                factor += 1

            jump = 4
            rbins = np.arange(0, factor + delta, delta)
            rticks = np.arange(0, factor + delta * jump, delta * jump)
            # print(rticks)
            # rlabels = range(-len(rticks)/2,len(rticks)/2)
            rlabels = range(len(rticks))
            rlabels = [f'{(x / (cuts / jump))}' if x > 0 else '0' for x in rlabels]
            # print(rticks)
            # print(rlabels)

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
            if events > 0:
                for ix, col in enumerate(axs):
                    col.hist(rs[ix], bins=rbins, histtype='step',
                             label=["Both less", "1 more 1 less", "Both more"], color=['b', 'g', 'r'])
                    col.set_title(f'Photon {ix + 1}')
                    col.set_ylim(0.5, events)
                    col.set_xticks(rticks)
                    col.set_xticklabels(rlabels)
                    col.set_yscale('log')
                    col.legend()

            plt.suptitle(f'Filtro {ip}: Eta menor a 2.37 \n{events} eventos')
            plt.savefig(destiny_filters + f'filter{ip}_1D.png')
            # plt.show()
            plt.close()

            # FIltro 4 - CUTFLOW
            df = df.loc[df.pair_eta == 0]
            events = df.shape[0]

            ip+=1

            ###################### Filtro 5 ######################
            df['pair_eta2'] = 2
            df.loc[(df['abs_eta'][1] < 1.37) & (df['abs_eta'][2] < 1.37), 'pair_eta2'] = 0
            df.loc[((((df['abs_eta'][1] < 1.37) | (df['abs_eta'][1] > 1.52)) &
                     ((df['abs_eta'][2] < 1.37) | (df['abs_eta'][2] > 1.52))) &
                    ((df['abs_eta'][1] < 1.37) != (df['abs_eta'][2] < 1.37))), 'pair_eta2'] = 1
            # print(df.loc[df.pair_eta2==1,['abs_eta','pair_eta2']])

            rs = []
            rs.append([df[df.pair_eta2 == i]['abs_eta', 1].values for i in range(3)])
            rs.append([df[df.pair_eta2 == i]['abs_eta', 2].values for i in range(3)])

            cuts = 10
            delta = 1 / cuts
            factor = 1
            maximum = amax2(df['abs_eta'].values)
            # print(maximum)

            while factor < maximum:
                factor += 1

            jump = 2
            rbins = np.arange(0, factor + delta, delta)
            rticks = np.arange(0, factor + delta * jump, delta * jump)
            # print(rticks)
            # rlabels = range(-len(rticks)/2,len(rticks)/2)
            rlabels = range(len(rticks))
            rlabels = [f'{(x / (cuts / jump))}' if x > 0 else '0' for x in rlabels]
            # print(rticks)
            # print(rlabels)

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
            if events > 0:
                for ix, col in enumerate(axs):
                    col.hist(rs[ix], bins=rbins, histtype='step',
                             label=["Both in barrel", "1 barrel 1 endcap", "2 endcap or at least 1\nin intersection"],
                             color=['b', 'g', 'r'])
                    col.set_title(f'Photon {ix + 1}')
                    col.set_ylim(0.5, events)
                    col.set_xticks(rticks)
                    col.set_xticklabels(rlabels)
                    col.set_yscale('log')
                    col.legend()

            plt.suptitle(f'Filtro {ip} - {ip+1}: Ninguno en el intersec y >=1 en el barrel\n{events} eventos')
            plt.savefig(destiny_filters + f'filter{ip}_1D.png')
            # plt.show()
            plt.close()

            df = df.loc[df.pair_eta2 < 2]
            events = df.shape[0]
            plot_params_1('5-6', f'Ninguno en el intersec y >=1 en el barrel - {events} eventos')
            plot_params_2('5-6', f'Ninguno en el intersec y >=1 en el barrel - {events} eventos')

            ip += 2

            ################## filtro E7

            df['pair_et'] = 0
            df.loc[(df['ET'][1] <= 50) & (df['ET'][2] > 50), 'pair_et'] = 1
            df.loc[(df['ET'][2] <= 50) & (df['ET'][1] > 50), 'pair_et'] = 1
            df.loc[(df['ET'][1] > 50) & (df['ET'][2] > 50), 'pair_et'] = 2
            # print(df.loc[:,['ET','pair_et']])

            rs = []
            rs.append([df[df.pair_et == i]['ET', 1].values for i in range(3)])
            rs.append([df[df.pair_et == i]['ET', 2].values for i in range(3)])

            cuts = 1 / 10
            delta = 1 / cuts
            factor = 1
            maximum = amax2(df['ET'].values)
            # print(maximum)

            while factor < maximum:
                factor += 1

            jump = 5
            rbins = np.arange(0, factor + delta, delta)
            rticks = np.arange(0, factor + delta * jump, delta * jump)
            # print(rticks)
            rlabels = range(len(rticks))
            rlabels = [f'{(x / (cuts / jump))}' if x > 0 else '0' for x in rlabels]
            # print(rticks)
            # print(rlabels)

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
            if events > 0:
                for ix, col in enumerate(axs):
                    col.hist(rs[ix], bins=rbins, histtype='step',
                             label=["Both less", "1 more 1 less", "Both more"], color=['r', 'g', 'b'])
                    col.set_title(f'Photon {ix + 1}')
                    col.set_ylim(0.5, events)
                    col.set_xticks(rticks)
                    col.set_xticklabels(rlabels)
                    col.set_yscale('log')
                    col.legend()

            plt.suptitle(f'Filtro {ip}: ET mayor a 50 GeV\n{events} eventos')
            plt.savefig(destiny_filters + f'filter{ip}_1D.png')
            # plt.show()
            plt.close()

            # FILTRO 2 - CUTFLOW
            df = df.loc[df.pair_et == 2]
            events = df.shape[0]

            plot_params_1(2, f'ET > a 50 GeV - {events} eventos')
            plot_params_2(2, f'ET > a 50 GeV - {events} eventos')

            ip += 1
            ################### Filtro 3 #################
            df['pair_rt'] = 0
            df.loc[(df['rel_tof'][1] < 4) & (df['rel_tof'][2] >= 4), 'pair_rt'] = 1
            df.loc[(df['rel_tof'][2] < 4) & (df['rel_tof'][1] >= 4), 'pair_rt'] = 1
            df.loc[(df['rel_tof'][1] >= 4) & (df['rel_tof'][2] >= 4), 'pair_rt'] = 2
            # print(df.loc[:,['rel_tof','pair_rt']])

            rs = []
            rs.append([df[df.pair_rt == i]['rel_tof', 1].values for i in range(3)])
            rs.append([df[df.pair_rt == i]['rel_tof', 2].values for i in range(3)])

            cuts = 10
            delta = 1 / cuts
            factor = 1
            maximum = amax2(df['rel_tof'].values)
            # print(maximum)

            while factor < maximum:
                factor += 1

            jump = 2
            rbins = np.arange(0, factor + delta, delta)
            rticks = np.arange(0, factor + delta * jump, delta * jump)
            # print(rticks)
            rlabels = range(len(rticks))
            rlabels = [f'{(x / (cuts / jump))}' if x > 0 else '0' for x in rlabels]
            # print(rticks)
            # print(rlabels)

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
            if events > 0:
                for ix, col in enumerate(axs):
                    col.hist(rs[ix], bins=rbins, histtype='step',
                             label=["Both less", "1 more 1 less", "Both more"], color=['b', 'g', 'r'])
                    col.set_title(f'Photon {ix + 1}')
                    col.set_ylim(0.5, events)
                    col.set_xticks(rticks)
                    col.set_xticklabels(rlabels)
                    col.set_yscale('log')
                    col.legend()

            plt.suptitle(f'Filtro {ip}: TOF relativo menor a 4 ns\n{events} eventos')
            plt.savefig(destiny_filters + f'filter{ip}_1D.png')
            # plt.show()
            plt.close()

            # FILTRO3 - CUTFLOW
            df = df.loc[df.pair_rt == 0]
            events = df.shape[0]
            plot_params_1(3, f'TOF relativo menor a 4 ns - {events} eventos')

            if tev == 8:
                df = df[df['MET',1]<=120]
                df['MET1'] = df['MET',1]
            elif tev == 13:
                df = df[df['MET',1]<=300]
                df['MET1'] = df['MET', 1]
            #print(df.columns)
            metdata[type][card] = df.loc[:,['MET1','MET_bin']]
            metmaxs[type] = np.maximum(metmaxs[type], amax2(df['MET1']))


            '''
            ################ Filter 2 ####################
            df = df.loc[df.pair_z == 0]
            
            
            
            
            ################### Filtro 3 #################
            df['pair_rt']=0
            df.loc[(df['rel_tof'][1]<4) & (df['rel_tof'][2]>=4),'pair_rt'] = 1
            df.loc[(df['rel_tof'][2]<4) & (df['rel_tof'][1]>=4),'pair_rt'] = 1
            df.loc[(df['rel_tof'][1]>=4) & (df['rel_tof'][2]>=4),'pair_rt'] = 2
            #print(df.loc[:,['rel_tof','pair_rt']])

            rs=[]
            rs.append([df[df.pair_rt==i]['rel_tof', 1].values for i in range(3)])
            rs.append([df[df.pair_rt==i]['rel_tof', 2].values for i in range(3)])

            cuts = 10
            delta = 1/cuts
            factor = 1
            maximum = amax2(df['rel_tof'].values)
            #print(maximum)

            while factor < maximum:
                factor += 1

            jump = 2
            rbins = np.arange(0,factor+delta,delta)
            rticks = np.arange(0,factor+delta*jump,delta*jump)
            #print(rticks)
            rlabels = range(len(rticks))
            rlabels = [f'{(x/(cuts/jump))}' if x > 0 else '0' for x in rlabels]
            #print(rticks)
            #print(rlabels)

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,8))
            if events > 0:
                for ix,col in enumerate(axs):
                    col.hist(rs[ix],bins=rbins,histtype='step',
                     label=["Both less", "1 more 1 less", "Both more"],color=['b','g','r'])
                    col.set_title(f'Photon {ix+1}')
                    col.set_ylim(0.5,events)
                    col.set_xticks(rticks)
                    col.set_xticklabels(rlabels)
                    col.set_yscale('log')
                    col.legend()

            plt.suptitle(f'Filtro 3: TOF relativo menor a 4 ns\n{events} eventos')
            plt.savefig(destiny_filters+'filter3_1D.png')
            #plt.show()
            plt.close()

            # FILTRO3 - CUTFLOW
            df = df.loc[df.pair_rt==0]
            events = df.shape[0]
            plot_params_1(3,f'TOF relativo menor a 4 ns - {events} eventos')

            ################### Filtro 4 ##################
            df['abs_eta',1] = np.abs(df['eta',1])
            df['abs_eta',2] = np.abs(df['eta',2])
            df['pair_eta']=0
            df.loc[(df['abs_eta'][1]<2.37) & (df['abs_eta'][2]>=2.37),'pair_eta'] = 1
            df.loc[(df['abs_eta'][2]<2.37) & (df['abs_eta'][1]>=2.37),'pair_eta'] = 1
            df.loc[(df['abs_eta'][1]>=2.37) & (df['abs_eta'][2]>=2.37),'pair_eta'] = 2
            #print(df.loc[:,['abs_eta','pair_eta']])

            rs=[]
            rs.append([df[df.pair_eta==i]['abs_eta', 1].values for i in range(3)])
            rs.append([df[df.pair_eta==i]['abs_eta', 2].values for i in range(3)])

            cuts = 10
            delta = 1/cuts
            factor = 1
            maximum = amax2(df['abs_eta'].values)
            #print(maximum)

            while factor < maximum:
                factor += 1

            jump = 4
            rbins = np.arange(0,factor+delta,delta)
            rticks = np.arange(0,factor+delta*jump,delta*jump)
            #print(rticks)
            #rlabels = range(-len(rticks)/2,len(rticks)/2)
            rlabels = range(len(rticks))
            rlabels = [f'{(x/(cuts/jump))}' if x > 0 else '0' for x in rlabels]
            #print(rticks)
            #print(rlabels)

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,8))
            if events > 0:
                for ix,col in enumerate(axs):
                    col.hist(rs[ix],bins=rbins,histtype='step',
                     label=["Both less", "1 more 1 less", "Both more"],color=['b','g','r'])
                    col.set_title(f'Photon {ix+1}')
                    col.set_ylim(0.5,events)
                    col.set_xticks(rticks)
                    col.set_xticklabels(rlabels)
                    col.set_yscale('log')
                    col.legend()

            plt.suptitle(f'Filtro 4: Eta menor a 2.37 \n{events} eventos')
            plt.savefig(destiny_filters+'filter4_1D.png')
            #plt.show()
            plt.close()

            #FIltro 4 - CUTFLOW
            df = df.loc[df.pair_eta==0]
            events = df.shape[0]
            plot_params_1(4,f'Eta menor a 2.37 - {events} eventos')

            ###################### Filtro 5 ######################
            df['pair_eta2']=2
            df.loc[(df['abs_eta'][1]<1.37) & (df['abs_eta'][2]<1.37),'pair_eta2'] = 0
            df.loc[((((df['abs_eta'][1]<1.37) | (df['abs_eta'][1]>1.52)) &
                   ((df['abs_eta'][2]<1.37) | (df['abs_eta'][2]>1.52))) &
                    ((df['abs_eta'][1]<1.37) != (df['abs_eta'][2]<1.37))),'pair_eta2']=1
            #print(df.loc[df.pair_eta2==1,['abs_eta','pair_eta2']])

            rs=[]
            rs.append([df[df.pair_eta2==i]['abs_eta', 1].values for i in range(3)])
            rs.append([df[df.pair_eta2==i]['abs_eta', 2].values for i in range(3)])

            cuts = 10
            delta = 1/cuts
            factor = 1
            maximum = amax2(df['abs_eta'].values)
            #print(maximum)

            while factor < maximum:
                factor += 1

            jump = 2
            rbins = np.arange(0,factor+delta,delta)
            rticks = np.arange(0,factor+delta*jump,delta*jump)
            #print(rticks)
            #rlabels = range(-len(rticks)/2,len(rticks)/2)
            rlabels = range(len(rticks))
            rlabels = [f'{(x/(cuts/jump))}' if x > 0 else '0' for x in rlabels]
            #print(rticks)
            #print(rlabels)

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,8))
            if events > 0:
                for ix,col in enumerate(axs):
                    col.hist(rs[ix],bins=rbins,histtype='step',
                     label=["Both in barrel", "1 barrel 1 endcap", "2 endcap or at least 1\nin intersection"],color=['b','g','r'])
                    col.set_title(f'Photon {ix+1}')
                    col.set_ylim(0.5,events)
                    col.set_xticks(rticks)
                    col.set_xticklabels(rlabels)
                    col.set_yscale('log')
                    col.legend()

            plt.suptitle(f'Filtro 5 - 6: Ninguno en el intersec y >=1 en el barrel\n{events} eventos')
            plt.savefig(destiny_filters+'filter5_1D.png')
            #plt.show()
            plt.close()

            #FIltro 5-6 - CUTFLOW
            df = df.loc[df.pair_eta2<2]
            events = df.shape[0]
            plot_params_1('5-6',f'Ninguno en el intersec y >=1 en el barrel - {events} eventos')
            plot_params_2('5-6',f'Ninguno en el intersec y >=1 en el barrel - {events} eventos')
            print(f'RUNNING: {type} {card} {tev} '+ f'# EVENTOS FINAL: {events}')
            '''

    dir=f'./paper/{tev}/'
    Path(dir).mkdir(parents=True, exist_ok=True)

    for type in types[:]:
        #print(metdata[type][card])
        secciones = [metdata[type][card].loc[:,'MET1'] for card in cards]
        met_max = metmaxs[type]
        bin_lims = np.array(range(0, int(met_max) + 5, 5))

        #hists = [np.histogram(data, bins=bin_lims)[0] for data in secciones]
        #hists = [data/np.max(data) for data in hists]
        #xs = [bin_lims[:-1] for data in hists]
        #print(hists_n)
        #print(bin_lims.shape)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(secciones,bins=bin_lims,
                density=True, histtype='step', label= [f'{mass[card]}' for card in cards])
        for ix, pos in enumerate(met_marks[:-1]):
            ax.axvline(x=pos, color='k', linestyle='--')
            ax.text(pos+1.5,0.97, met_labels[ix], horizontalalignment="left",
                    verticalalignment="center", color='k', transform=ax.get_xaxis_transform(),fontsize=16)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ypos = ax.get_ylim()[1]
        ax.set_ylim(0,ypos*1.05)
        #ax.set_yscale('log')
        #ax.text(0.87, 0.67, f'VBF',horizontalalignment="left", transform=ax.transAxes, fontsize=13)
        ax.text(0.85,0.53,names[type], transform=ax.transAxes,
                horizontalalignment="left",fontsize=18)

        ax = format_exponent(ax, axis='y', size=16)
        ax.tick_params(axis='both', labelsize=18, width=1.5, length=7)

        leg=plt.legend(title='$\mathregular{{M_{{h}}}}$ [GeV]', fontsize=18)
        plt.setp(leg.get_title(), fontsize=18)
        leg._legend_box.align = "left"
        #plt.title(f'{type} METs', fontsize=13)
        plt.xlabel('MET [GeV]', fontsize=22)
        plt.savefig(dir + f'{type}_METs.png', bbox_inches='tight')
        #plt.show()
    

