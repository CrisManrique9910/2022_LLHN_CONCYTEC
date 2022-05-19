from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from my_funcs import get_scale
gbins = 50
xlims={i:dict() for i in range(1,4)}

def format_exponent(ax, axis='y',size=13):

    # Change the ticklabel format to scientific format
    ax.ticklabel_format(axis=axis, style='sci', scilimits=(0, 0))

    # Get the appropriate axis
    if axis == 'y':
        ax_axis = ax.yaxis
        x_pos = 0.0
        y_pos = 1.0
        horizontalalignment='left'
        verticalalignment='bottom'
    else:
        ax_axis = ax.xaxis
        x_pos = 1.0
        y_pos = -0.05
        horizontalalignment='right'
        verticalalignment='top'

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
        offset_text = r'x$\mathregular{10^{%d}}$' %expo
    else:
        offset_text = "   "
    # Turn off the offset text that's calculated automatically
    ax_axis.offsetText.set_visible(False)

    # Add in a text box at the top of the y axis
    ax.text(x_pos, y_pos, offset_text, fontsize=size, transform=ax.transAxes,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment)
    
    return ax

def z0_tof():
    for col in ['z_origin1','rel_tof','pt'][:]:
        txmin, txmax = textpos[col]
        cut = cuts[col]
        data = df[(df['select']==True)][[col]]
        data = data[col,1].combine_first(data[col,2])
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

        if col == 'z_origin1':
            gbins = np.linspace(-maxi,maxi,51)
        else:
            gbins = np.linspace(0, maxi, 51)

        #plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
        ax.hist([data[data['MET_bin1']==label][col] for label in met_labels],color=metcolors,
                    bins=gbins,density=True,stacked=True,label=met_labels)
        ax.text(txmin,txmax, f"{names[type]}\n$\mathregular{{M_{{h}}}}$ = {mass[card]} GeV\n{events} Events",
                transform=ax.transAxes,horizontalalignment=halig[col], linespacing=2, fontsize=18)

        if col == 'rel_tof' and maxi > 4:
            ax.axvline(x=4, color='b', linestyle='--')

        ax = format_exponent(ax, axis='y', size=16)
        ax.tick_params(axis='both', labelsize=18,width=1.5,length=7)

        #ax.yaxis.offsetText.set_fontsize(14)
        plt.xlabel(xlabels[col], fontsize=22)
        plt.legend(fontsize=18)
        #plt.show()
        fig.savefig(case + f'{type}_{mass[card]}_{col.replace("_","")}_it{g}.png',
                    bbox_inches='tight')
        #print(data)
        plt.close()
    return g+1

radius = 1775
semilength=4050
hbar=6.582119514*10**(-25)	#GeV*s
#widths= {1:2.272390*10**(-16),2:2.147480*10**(-17),3:1.319880*10**(-19),
#         4:2.2722*10**(-16),5:1.40011*10**(-16),6:1.63114*10**(-16),
#         7:1.06164*10**(-16),8:6.59185*10**(-17),9:4.08258*10**(-17)}	#particle decay width
widths={13:2*10**(-16),14:4*10**(-16),15:1*10**(-15)}
limits_less_z = [0,40,80,120,160,200,2000]
limits_more_z = [0,50,100,150,200,250,2000]
max_zo = limits_more_z[-1]

limits_less_t = [-4,0.5,1.1,1.3,1.5,1.8,4]
limits_more_t = [-4,0.4,1.2,1.4,1.6,1.9,4]

metcolors= ['C3','C2','C1','C0']
met_labels = ['BKG','CR1','CR2','SR']
cuts = {'z_origin1':4000,'rel_tof':7,'pt':np.inf}
xlabels = {'z_origin1':'$\mathregular{{\Delta}z_{\gamma}}$ [mm]',
           'rel_tof':'$\mathregular{t_{\gamma}}$ [ns]', 'pt':'PT [GeV]'}
textpos = {'z_origin1':[0.04,0.74],'rel_tof':[0.98,0.47],'pt':[0.98,0.47]}
halig = {'z_origin1':'left','rel_tof':'right','pt':'right'}
filter = 'rel_tof'
types = ['VBF', 'GF']
names = {'VBF':'Vector Boson Fusion','GF':'Gluon Fusion'}
cards = [13,14,15]
mass = {13:50,14:30,15:10}
tevs = [8, 13]

for tev in tevs[:1]:
    for type in types[:]:
        for card in cards[:]:
            g = 1
            case = f"./paper/{tev}/{type}/{card}/"
            #case = f"./cases/{type}/{card}/{tev}/images/ATLAS"
            Path(case).mkdir(parents=True, exist_ok=True)
            file_in = f'data/clean/photon_df-{type}_{card}_{tev}.xlsx'
            width = widths[card]
            lifetime = (10**9)*hbar/width
            print(f'RUNNING: {type} {card} {tev} ')
            print(f'lifetime of the heavy neutrino: {lifetime} ns\n')
            df = pd.read_excel(file_in, header=[0,1],index_col=0)

            if lifetime < 4:
                table = np.zeros((4,len(limits_less_z)-1,len(limits_less_t)-1)) # rows = z, columns = t
            else:
                table = np.zeros((4,len(limits_more_z)-1,len(limits_more_t)-1))
            #print(table)

            limsup = limits_less_z[-1]

            #print(table_less)

            count0 = df.shape[0]
            df['MET_bin1'] = df['MET_bin',1]
            #print(df['MET_bin1'])
            df['z'] = np.abs(df['z'])
            df['z_origin1',1] = df['z_origin',1]
            df['z_origin1', 2] = df['z_origin', 2]
            df['z_origin'] = np.abs(df['z_origin'])
            df = df.loc[((df.z[1]<1) & (df.z[2]<1))& ((df.r[1]<1) & (df.r[2]<1))]
            count1 = df.shape[0]
            extra1 = df.loc[(df.z_origin[1] > max_zo) | (df.z_origin[2] > max_zo)].shape[0]
            ##### INSIDE THE DETECTOR
            #print(df.shape[0])
            #print(df['MET_bin1'])
            df[('select',1)]= False
            df[('select', 2)] = False
            df.loc[(df['eta'][1] < 1.37) & (df['eta'][2] >= 1.37), ('select', 1)] = True
            df.loc[(df['eta'][2] < 1.37) & (df['eta'][1] >= 1.37), ('select', 2)] = True
            df.loc[
                    (
                        ((df['eta'][1] < 1.37) & (df['eta'][2] < 1.37)) |
                        ((df['eta'][1] >= 1.37) & (df['eta'][2] >= 1.37))
                    )
                     &
                    (df[filter][1] >= df[filter][2])
                    ,
                    ('select', 1)
            ] = True
            df.loc[
                (
                        ((df['eta'][1] < 1.37) & (df['eta'][2] < 1.37)) |
                        ((df['eta'][1] >= 1.37) & (df['eta'][2] >= 1.37))
                )
                &
                (df[filter][1] < df[filter][2])
                ,
                ('select', 2)
            ] = True

            df['eta'] = np.abs(df['eta'])
            df = df.loc[(df.eta[1] < 2.37) & (df.eta[2] < 2.37)]
            count4 = df.shape[0]
            extra4 = df.loc[(df.z_origin[1] > max_zo) | (df.z_origin[2] > max_zo)].shape[0]

            df = df.loc[((df.eta[1]<1.37) | (df.eta[1]>1.52)) & ((df.eta[2]<1.37) | (df.eta[2]>1.52))]
            count5 = df.shape[0]
            extra5 = df.loc[(df.z_origin[1] > max_zo) | (df.z_origin[2] > max_zo)].shape[0]

            df = df.loc[(df.eta[1]<1.37) | (df.eta[2]<1.37)]
            count6 = df.shape[0]
            extra6 = df.loc[(df.z_origin[1] > max_zo) | (df.z_origin[2] > max_zo)].shape[0]
            #print(df.select.sum())
            g = z0_tof()
            #print(df['MET_bin'][1].unique())

            df = df.loc[(df.ET[1]>50) & (df.ET[2] >50)]
            count2 = df.shape[0]
            extra2 = df.loc[(df.z_origin[1] > max_zo) | (df.z_origin[2] > max_zo)].shape[0]
            g = z0_tof()

            df = df.loc[(df.rel_tof[1]<4) & (df.rel_tof[2]<4)]
            count3 = df.shape[0]
            extra3 = df.loc[(df.z_origin[1] > max_zo) | (df.z_origin[2] > max_zo)].shape[0]

            g = z0_tof()
            #print(count6,extra6)

            for index,pair in df.iterrows():
                # Selecting the photon that is going to be measured
                if pair['eta'][1] < 1.37 and pair['eta'][2] >= 1.37:
                    photon = pair[:,1]
                elif pair['eta'][2] < 1.37 and pair['eta'][1] >= 1.37:
                    photon = pair[:,2]
                elif pair[filter][1] >= pair[filter][2]:
                    photon = pair[:,1]
                else:
                    photon = pair[:,2]

                #print(photon)
                # Table depending on the lifetime
                # if photon['t_neu'] < 4:
                abs_z = photon['z_origin']
                rel_t = photon['rel_tof']
                met = photon['MET']
                ix = 4
                row = 0
                col = 0
                if (met > 20) and (met < 50):
                    ix = 1
                elif (met > 50) and (met < 75):
                    ix = 2
                elif met > 75:
                    ix = 3
                else:
                    ix = 0

                if lifetime < 4:
                    limits_z = limits_less_z
                    limits_t = limits_less_t
                else:
                    limits_z = limits_more_z
                    limits_t = limits_more_t
                while limits_z[row + 1] < abs_z and (row + 1) < (len(limits_z) - 1):
                    row += 1
                # print(row)
                while limits_t[col + 1] < rel_t and (col + 1) < (len(limits_t) - 1):
                    col += 1
                table[ix][row, col] += 1

            np.save('photon_dist.npy',table)
            print('                                        # Eventos       # > 2000 mm')
            print(f'                 # Inicial de eventos:{count0: 11}                 -')
            print(f'    Ambos fotones dentro del detector:{count1: 11}{extra1: 18}')
            print(f'  Ambos fotones con ET mayor a 50 GeV:{count2: 11}{extra2: 18}')
            print(f'             TOF relativo mayor a 4ns:{count3: 11}{extra3: 18}')
            print(f'          Pseudorapidity mayor a 2.37:{count4: 11}{extra4: 18}')
            print(f'Caida de 1 en la zona de intersección:{count5: 11}{extra5: 18}')
            print(f'          Caida de ambos en el endcap:{count6: 11}{extra6: 18}')
            # print(outliers)

            table = table*get_scale(tev,type,mass[card])
            events = np.sum(table)
            nbins = np.array(range(table.shape[2]+1))+0.5

            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(14,8))
            #fig.suptitle(f'Using {radius}mm - {semilength}mm and the Max {filter} photon\ntau = {lifetime} ns')
            i = 0
            ymax=[]
            for row in axs:
                for col in row:
                    final = '['
                    if i == 5:
                        final =']'
                    col.hist(nbins[:-1], bins=nbins, weights=table[0][i], histtype='step', color='C3', label='BKG')
                    col.hist(nbins[:-1],bins=nbins,weights=table[1][i], histtype='step', color='C2', label='CR1')
                    col.hist(nbins[:-1], bins=nbins, weights=table[2][i], histtype='step', color='C1', label='CR2')
                    col.hist(nbins[:-1], bins=nbins, weights=table[3][i], histtype='step', color='C0', label='SR')
                    col.set_yscale('log')
                    #col.set_ylabel('Count')
                    col.set_xlabel('$\mathregular{t_{\gamma}}$ bins')
                    col.set_xticks(nbins[1:-1])
                    col.set_xticks(nbins[1:]-0.5,minor=True)
                    col.tick_params(axis='x', which='major', direction='in',
                                    labelbottom=False,width=1.5,length=7)
                    col.tick_params(axis='x', which='minor', length=0)
                    col.xaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
                    col.set_xlim(0.5,6.5)
                    col.text(0.74,0.55,f'z interval:\n[{limits_z[i]},{limits_z[i+1]}{final}',
                             horizontalalignment="left",transform=col.transAxes)
                    col.legend()
                    ymax.append(col.get_ylim()[1])
                    i+= 1
            plt.setp(axs,ylim=(0.8,max(ymax)))
            fig.savefig(case + f'{type}_{mass[card]}_bins_{radius}_{semilength}_max{filter}.png',bbox_inches='tight')
            #plt.show()
            plt.close()

            message = ''
            for idx,region in enumerate(met_labels):
                message += f'{region:3} \t{np.sum(table[idx]):.2f}  \t{100*np.sum(table[idx])/events:.2f} %\n'

            with open(case + 'region_dist.txt','w') as file:
                file.write(message)

