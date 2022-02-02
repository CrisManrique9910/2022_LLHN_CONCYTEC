import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

radius = 1775
semilength=4050
filter = 'rel_tof'
file_in = f'./data/clean/photon_df.xlsx'

width=4.99045323E-16	#particle decay width
hbar=6.582119514*10**(-25)	#GeV*s
lifetime = (10**9)*hbar/width
print(f'lifetime of the heavy neutrino: {lifetime} ns\n')
df = pd.read_excel(file_in, header=[0,1],index_col=0)

limits_less_z = [0,40,80,120,160,200,2000]
limits_more_z = [0,50,100,150,200,250,2000]
max_zo = limits_more_z[-1]

limits_less_t = [-4,0.5,1.1,1.3,1.5,1.8,4]
limits_more_t = [-4,0.4,1.2,1.4,1.6,1.9,4]

if lifetime < 4:
    table = np.zeros((4,len(limits_less_z)-1,len(limits_less_t)-1)) # rows = z, columns = t
else:
    table = np.zeros((4,len(limits_more_z)-1,len(limits_more_t)-1))
#print(table)

limsup = limits_less_z[-1]

#print(table_less)

count0 = df.shape[0]

df['z'] = np.abs(df['z'])
df['z_origin'] = np.abs(df['z_origin'])
df = df.loc[((df.z[1]<semilength) & (df.z[2]<semilength))& ((df.r[1]<radius) & (df.r[2]<radius))]
count1 = df.shape[0]
extra1 = df.loc[(df.z_origin[1] > max_zo) | (df.z_origin[2] > max_zo)].shape[0]

df = df.loc[(df.ET[1]>50) & (df.ET[2] >50)]
count2 = df.shape[0]
extra2 = df.loc[(df.z_origin[1] > max_zo) | (df.z_origin[2] > max_zo)].shape[0]

df = df.loc[(df.rel_tof[1]<4) & (df.rel_tof[2]<4)]
count3 = df.shape[0]
extra3 = df.loc[(df.z_origin[1] > max_zo) | (df.z_origin[2] > max_zo)].shape[0]

df['eta'] = np.abs(df['eta'])
df = df.loc[(df.eta[1]<2.37) & (df.eta[2]<2.37)]
count4 = df.shape[0]
extra4 = df.loc[(df.z_origin[1] > max_zo) | (df.z_origin[2] > max_zo)].shape[0]

df = df.loc[((df.eta[1]<1.37) | (df.eta[1]>1.52)) & ((df.eta[2]<1.37) | (df.eta[2]>1.52))]
count5 = df.shape[0]
extra5 = df.loc[(df.z_origin[1] > max_zo) | (df.z_origin[2] > max_zo)].shape[0]

df = df.loc[(df.eta[1]<1.37) | (df.eta[2]<1.37)]
count6 = df.shape[0]
extra6 = df.loc[(df.z_origin[1] > max_zo) | (df.z_origin[2] > max_zo)].shape[0]

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
#print(table)
nbins = np.array(range(table.shape[2]+1))+0.5

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(14,8))
fig.suptitle(f'Using {radius}mm - {semilength}mm and the Max {filter} photon\ntau = {lifetime} ns')
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
        col.set_xlabel('relative tof Bins')
        col.set_xticks([1,2,3,4,5,6])
        col.text(0.7,0.3,f'z interval:\n[{limits_z[i]},{limits_z[i+1]}{final}', transform=col.transAxes)
        col.legend()
        ymax.append(col.get_ylim()[1])
        i+= 1
plt.setp(axs,ylim=(0.8,max(ymax)))
fig.savefig(f'./images/ATLAS/bins_{radius}_{semilength}_max{filter}.png')
#plt.show()
