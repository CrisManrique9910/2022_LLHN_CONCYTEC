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

gbins = 50
xlims={i:dict() for i in range(1,4)}

edges = [-0.1, 30, 50, np.inf]
labels = list(range(len(edges) - 1))

metcolors= ['C3','C2','C0']
met_labels = ['BKG','CR','SR']
cuts = {'z_origin':4000,'rel_tof':7,'pt':np.inf}
xlabels = {'z_origin':'$\mathregular{{\Delta}z_{\gamma}}$ [mm]',
           'rel_tof':'$\mathregular{t_{\gamma}}$ [ns]', 'pt':'PT [GeV]'}
textpos = {'z_origin':[0.04,0.74],'rel_tof':[0.98,0.47],'pt':[0.98,0.47]}
halig = {'z_origin':'left','rel_tof':'right','pt':'right'}

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


def format_exponent(ax, axis='y', size=13):
    # Change the ticklabel format to scientific format
    ax.ticklabel_format(axis=axis, style='sci', scilimits=(0, 0))

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

def z0_tof(df):
    for col in ['z_origin','rel_tof'][:]:
        txmin, txmax = textpos[col]
        cut = cuts[col]
        data = df[[col]]
        #data = df[(df['select']==True)][[col]]
        #data = data[col,1].combine_first(data[col,2])
        #print(maxi)
        data = df[['MET_bin1']].join(data)
        data.columns = ['MET_bin1',col]
        events=round(get_scale(tev,type,mass[card])*data.shape[0])
        data = data[(data[col] <= cut) & (data[col] >= -cut)]
        maxi = max(np.abs(data[col]))

        if mass[card] == 50 and type=='VBF':
            maxi = max(np.abs(data[col]))
            xlims[g][col] = maxi
        else:
            maxi=xlims[g][col]

        fig, ax = plt.subplots(figsize=(8, 6))

        if col == 'z_origin':
            gbins = np.linspace(-maxi,maxi,51)
        else:
            gbins = np.linspace(0, maxi, 51)

        #plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
        ax.hist([data[data['MET_bin1']==label][col] for label in met_labels],color=metcolors,
                    bins=gbins,density=True,stacked=True,label=met_labels)
        ax.text(txmin,txmax, f"{channels[key]} channel\n$\mathregular{{M_{{h}}}}$ = {mass[card]} GeV\n",
                transform=ax.transAxes,horizontalalignment=halig[col], linespacing=2, fontsize=18)

        #if col == 'rel_tof' and maxi > 4:
        #    ax.axvline(x=4, color='b', linestyle='--')

        ax = format_exponent(ax, axis='y', size=16)
        ax.tick_params(axis='both', labelsize=18,width=1.5,length=7)

        #ax.yaxis.offsetText.set_fontsize(14)
        plt.xlabel(xlabels[col], fontsize=22)
        plt.legend(fontsize=18)
        #plt.show()
        fig.savefig(paper_ims + f'{type}_{mass[card]}_{key}ph_{col.replace("_","")}_it{g}.png',
                    bbox_inches='tight')
        print(f'{type}_{mass[card]}_{key}ph_{col.replace("_","")}_it{g}.png')
        plt.close()
    return g+1

types = ['VBF','GF']
names = {'VBF':'Vector Boson Fusion','GF':'Gluon Fusion'}
channels = {'1': '$\mathregular{\gamma}$', '2+':  '$\mathregular{\gamma}$'*2}
cards = [13,14,15]
mass = {13:50,14:30,15:10}
tevs = [13]

z_bins = [0,50,100,200,300,2000.1]
t_bins = {'1': [0,0.2,0.4,0.6,0.8,1.0,1.5,12.1], '2+': [0,0.2,0.4,0.6,0.8,1.0,1.5,12.1]}

bin_matrix = dict()
bin_matrix_neg = dict()
lost_ev = dict()

#print(bin_matrix)

np.random.seed(0)

for tev in tevs[:]:
    for type in types[:1]:
        for card in cards[:]:

            for key, tbin in t_bins.items():
                bin_matrix[key] = np.zeros((len(z_bins) - 1, len(tbin) - 1))
                bin_matrix_neg[key] = np.zeros((len(z_bins) - 1, len(tbin) - 1))

            lost_ev[card] = dict()
            print(f'RUNNING: {type} {card} {tev} ')

            folder_ims = f"./cases/{tev}/{type}/{card}/ATLAS/after_filters/"
            paper_ims = f"./paper/{tev}/{type}/{card}/"
            folder_txt = f"./cases/{tev}/{type}/"
            paper_txt = f"./paper/{tev}/{type}/"
            destiny_info = f'./data/clean/'
            dfs = {'1': '','2+': ''}
            dfs_neg = {'1': '', '2+': ''}

            Path(folder_ims).mkdir(parents=True, exist_ok=True)
            Path(folder_txt).mkdir(parents=True, exist_ok=True)
            Path(paper_txt).mkdir(parents=True, exist_ok=True)
            Path(paper_ims).mkdir(parents=True, exist_ok=True)

            scale = get_scale(tev, type, mass[card])

            ix = 0
            for key in dfs.keys():
                g = 1
                lost_ev[card][key] = []

                dfs[key] = pd.read_pickle(f'./data/clean/df_photon_{key}-{type}_{card}_{tev}.pickle')
                dfs[key]['E'] = np.sqrt(dfs[key]['ET']**2 + dfs[key]['pz']**2)
                dfs[key]['rt_smeared'] = \
                    dfs[key].apply(lambda row: row['rel_tof'] + t_res(0.3*row['E'])*np.random.normal(0,1), axis=1)


                plt.hist(dfs[key]['rel_tof'],bins=50, color=f'C{ix}')
                plt.xlabel('t_gamma [ns]')
                plt.savefig(folder_ims + f'rel_tof_{key}.png')
                # plt.show()
                plt.close()

                plt.hist(dfs[key]['rt_smeared'], bins=50, color=f'C{ix}')
                plt.xlabel('t_gamma Smeared [ns]')
                plt.savefig(folder_ims + f'rel_tof_{key}_smeared.png')
                # plt.show()
                plt.close()

                dfs[key]['zo_smeared'] = \
                    dfs[key].apply(lambda row: row['z_origin'] + z_res(row['z_origin']) * np.random.normal(0, 1), axis=1)

                #print(dfs[key]['zo_smeared'])
                #sys.exit()
                #print(np.bincount(dfs[key]['z_origin'].values.astype(np.float)))

                plt.close()
                plt.hist(dfs[key]['z_origin'], bins=60, color=f'C{ix}')
                plt.xlabel('z_origin [mm]')
                plt.savefig(folder_ims + f'z_origin_{key}.png')
                #plt.show()

                plt.close()
                plt.hist(dfs[key]['zo_smeared'], bins=50, color=f'C{ix}')
                plt.xlabel('z_origin Smeared [mm]')
                plt.savefig(folder_ims + f'z_origin_{key}_smeared.png')
                #plt.show()

                dfs[key] = dfs[key][np.abs(dfs[key]['zo_smeared']) <= 2000]
                lost_ev[card][key].append(dfs[key].shape[0])

                dfs_neg[key] = dfs[key][(0 > dfs[key]['rt_smeared']) & (dfs[key]['rt_smeared'] >= -12)]
                dfs[key] = dfs[key][(0 <= dfs[key]['rt_smeared']) & (dfs[key]['rt_smeared'] <= 12)]
                lost_ev[card][key].append(dfs[key].shape[0])
                #print(np.min(dfs[key]['zo_smeared']), np.max(dfs[key]['zo_smeared']))

                plt.hist(dfs[key]['rt_smeared'], bins=50, color=f'C{ix}')
                plt.xlabel('t_gamma Smeared [ns]')
                plt.savefig(folder_ims + f'rel_tof_{key}_smeared_cutted.png')
                plt.close()

                plt.hist(dfs[key]['zo_smeared'], bins=50, color=f'C{ix}')
                plt.xlabel('z_origin Smeared [mm]')
                plt.savefig(folder_ims + f'z_origin_{key}_smeared_cutted.png')
                # plt.show()
                plt.close()

                dfs[key]['t_binned'] = np.digitize(dfs[key]['rt_smeared'], t_bins[key])
                dfs[key]['z_binned'] = np.digitize(np.abs(dfs[key]['zo_smeared']),z_bins)

                dfs_neg[key]['t_binned'] = np.digitize(np.abs(dfs_neg[key]['rt_smeared']), t_bins[key])
                dfs_neg[key]['z_binned'] = np.digitize(np.abs(dfs_neg[key]['zo_smeared']), z_bins)
                #print(dfs_neg[key])
                #sys.exit()
                #print(np.min(dfs[key]['t_binned']), np.max(dfs[key]['t_binned']))

                ### z_origin and rel:tof graphs
                dfs[key]['MET_bin1'] = pd.cut(dfs[key]['MET'], bins=edges, labels=met_labels)
                #print(dfs[key]['MET_bin1'])
                g = z0_tof(dfs[key])


                for ind, row in dfs[key].iterrows():
                    #print(ind)
                    bin_matrix[key][row['z_binned'] - 1, row['t_binned'] - 1] +=1
                    #print(row['z_binned'], row['t_binned'])

                for ind, row in dfs_neg[key].iterrows():
                    #print(ind)
                    bin_matrix_neg[key][row['z_binned'] - 1, row['t_binned'] - 1] +=1
                #print(bin_matrix_neg)
                ix += 1

            #print(bin_matrix)
            fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(20, 30))
            plt.subplots_adjust(left=None, bottom=0.05, right=None, top=0.95, wspace=None, hspace=0.3)
            ymax = ymin = []
            for key in bin_matrix.keys():
                nbins = np.array(range(bin_matrix[key].shape[1] + 1)) + 0.5
                ix = int(key[0])-1
                ir = 0
                for row in axs:
                    row[ix].hist(nbins[:-1], bins=nbins, weights=bin_matrix[key][ir]*scale, histtype='step')
                    row[ix].set_yscale('log')
                    row[ix].set_xticks(np.array(range(bin_matrix[key].shape[1])) + 1)
                    row[ix].set_title(f'Dataset {key} ph - bin z {ir + 1}')
                    ymax.append(row[ix].get_ylim()[1])
                    ymin.append(row[ix].get_ylim()[0])
                    ir+=1
            plt.setp(axs, ylim=(min(ymin), max(ymax)))
            fig.savefig(folder_ims + f'{type}_{card}_zbins_tbins_pos.png')
            fig.savefig(paper_ims + f'{type}_{card}_zbins_tbins_BeforeDelphes_pos.png')
            #plt.show()
            plt.close()

            fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(20, 30))
            plt.subplots_adjust(left=None, bottom=0.05, right=None, top=0.95, wspace=None, hspace=0.3)
            ymax = ymin = []
            for key in bin_matrix_neg.keys():
                nbins = np.array(range(bin_matrix_neg[key].shape[1] + 1)) + 0.5
                ix = int(key[0]) - 1
                ir = 0
                for row in axs:
                    row[ix].hist(nbins[:-1], bins=nbins, weights=bin_matrix_neg[key][ir] * scale, histtype='step',
                                 color='C1')
                    row[ix].set_yscale('log')
                    row[ix].set_xticks(np.array(range(bin_matrix_neg[key].shape[1])) + 1)
                    row[ix].set_title(f'Dataset {key} ph - bin z {ir + 1}')
                    ymax.append(row[ix].get_ylim()[1])
                    ymin.append(row[ix].get_ylim()[0])
                    ir += 1
            plt.setp(axs, ylim=(min(ymin), max(ymax)))
            fig.savefig(folder_ims + f'{type}_{card}_zbins_tbins_neg.png')
            fig.savefig(paper_ims + f'{type}_{card}_zbins_tbins_BeforeDelphes_neg.png')
            # plt.show()
            plt.close()
            #sys.exit()
            for key in dfs.keys():
                print(dfs[key].shape)
                dfs[key].to_pickle(f'./data/clean/df_photon_{key}_smeared-{type}_{card}_{tev}_pos.pickle')
            for key in dfs_neg.keys():
                #print(dfs_neg[key])
                dfs_neg[key].to_pickle(f'./data/clean/df_photon_{key}_smeared-{type}_{card}_{tev}_neg.pickle')
            print('dfs saved!')

        message = ''
        for card, value1 in lost_ev.items():
            for ch, value2 in value1.items():
                losses = value2[0] - value2[1]
                #print(losses)
                message += f'{mass[card]}\t{ch:2}\t{round(losses*scale):5}\t{100*losses/value2[0]:.2f} %\n'
            message += '\n'

        with open(folder_txt + 'negative_events.txt', 'w') as file:
            file.write(message)
        with open(paper_txt + 'negative_events.txt', 'w') as file:
            file.write(message)

        #print(message)





