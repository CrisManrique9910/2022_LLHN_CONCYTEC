import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


ATLASdet_radius= 1.775
ATLASdet_semilength = 4.050
nbins = 50


destiny_filters = './images/ATLAS/filters/'
destiny_cutflow = './images/ATLAS/cutflow/'
df_in =  f'./data/clean/photon_df.xlsx'
folder_list = ['MET','eta','pt','pz','z_origin','ET','rel_tof']
folder_color = ['Brown','Purple','Orange','Green','Blue','Red','Black']
features =  list(zip(folder_list,folder_color))
met_marks = [0,20,50,75,np.inf]
met_labels = ['BKG','CR1','CR2','SR']
metcolors= ['C3','C2','C1','C0']
Path(destiny_filters).mkdir(parents=True,exist_ok=True)

df = pd.read_excel(df_in, header=[0,1],index_col=0)

df['r'] = (df['r']/1000)/ATLASdet_radius
df['z'] = np.abs(df['z']/1000)/ATLASdet_semilength

events = df.shape[0]
max_ed = int(np.ceil(np.amax(df['r'].to_numpy())))

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
plt.title(f'Filtro 1a: Dentro del radio\n{events} eventos')
plt.savefig(destiny_filters+'filter1a_2D.png')
#plt.show()
plt.close()

################ Filter 1b ####################

df = df.loc[(df.r[1]<1) & (df.r[2]<1)]
events = df.shape[0]
max_ed = int(np.ceil(np.amax(df['z'].to_numpy())))

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
plt.title(f'Filtro 1b: Dentro de la longitud z\n{events} eventos')
plt.savefig(destiny_filters+'filter1b_2D.png')
#plt.show()
plt.close()

################ Filter 2 ####################
df = df.loc[(df.z[1]<1) & (df.z[2]<1)]
events = df.shape[0]
max_ed = np.ceil(np.amax(df['ET'].to_numpy()))

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
#print(juice)

fig, ax = plt.subplots(figsize=(8,8))
for mask,colors in [(mask1,'Reds'),(mask2,'Greens'),(mask3,'Blues')]:
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
plt.title(f'Filtro 2: ET mayor a 50 GeV\n{events} eventos')
plt.savefig(destiny_filters+'filter2_2D.png')
#plt.show()
plt.close()


################### Filtro 3 #################
df = df.loc[(df.ET[1]>50) & (df.ET[2] >50)]
events = df.shape[0]
max_ed = np.ceil(np.amax(df['rel_tof'].to_numpy()))

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

max_ed = np.ceil(np.amax(np.abs(df['eta'].to_numpy())))

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
plt.title(f'Filtro 5 - 6: Ninguno en el intersec y >=1 en el barrel\n{events} eventos')
plt.savefig(destiny_filters+'filter5_2D.png')
#plt.show()
plt.close()


