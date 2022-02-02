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
centers = ['eta','z_origin','pz']
features =  list(zip(folder_list,folder_color))
met_marks = [0,20,50,75,np.inf]
met_labels = ['BKG','CR1','CR2','SR']
metcolors= ['C3','C2','C1','C0']
Path(destiny_filters).mkdir(parents=True,exist_ok=True)

for folder in folder_list:
    Path(destiny_cutflow + folder).mkdir(parents=True, exist_ok=True)

def plot_params_1(filter_num,filter_desc):
    met_max = np.max(df['MET',1])
    secciones = [df[df['MET_bin',1]==i] for i in met_labels]
    for param, color in features[:]:
        if param == 'MET':
            metbins = list(range(0,int(met_max)+5,5))
            fig, ax = plt.subplots(figsize=(10,6))
            ax.hist([dfp[param, 1] for dfp in secciones],color=metcolors,
                    bins=metbins,histtype='step',label = met_labels)
            ax.set_ylim(0.5)
            ax.set_yscale('log')
            plt.legend()
        else:
            pmin = np.amin(df[param].to_numpy())
            pmax = np.amax(df[param].to_numpy())
            if param in centers:
                pmax = np.max(np.abs([pmin, pmax]))
                pmin = -1 * pmax
            param_bins = np.linspace(pmin,pmax + (pmax - pmin)/nbins,nbins)
            #print(pmax)
            ymax = []
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
            for ix, col in enumerate(axs):
                #print(np.max(df[param,ix+1]))
                col.hist(df[param, ix + 1], histtype='step', bins=param_bins, color=color)
                #print(dummy)
                col.set_title(f'Photon {ix + 1}')
                col.set_yscale('log')
                ymax.append(col.get_ylim()[1])
            plt.setp(axs, ylim=(0.8, max(ymax)))
        plt.suptitle(f'After Filter {filter_num}: {filter_desc}\n{param}',fontsize=14)
        plt.savefig(destiny_cutflow+param+f'/{param}_after_filter{filter_num}.png')
        plt.close()
    return

def plot_params_2(filter_num,filter_desc):
    secciones = [df[df['MET_bin',1]==i] for i in met_labels]
    for param, color in features[:]:
        if param != 'MET':
            pmin = np.amin(df[param].to_numpy())
            pmax = np.amax(df[param].to_numpy())
            if param in centers:
                pmax = np.max(np.abs([pmin,pmax]))
                pmin = -1*pmax
            param_bins = np.linspace(pmin, pmax + (pmax - pmin) / nbins, nbins)
            ymax = []
            ymin = []
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
            for ix, col in enumerate(axs):
                #print(np.max(df[param,ix+1]))
                col.hist([dfp[param, ix+1] for dfp in secciones],color=metcolors,
                    bins=param_bins,stacked=True,label = met_labels)
                #print(dummy)
                col.set_title(f'Photon {ix + 1}')
                ymax.append(col.get_ylim()[1])
                ymin.append(col.get_ylim()[0])
                col.legend()
            plt.setp(axs,ylim=(min(ymin),max(ymax)))
            plt.suptitle(f'After Filter {filter_num}: {filter_desc}\n{param}',fontsize=14)
            plt.savefig(destiny_cutflow+param+f'/{param}_after_filter{filter_num}_regions.png')
            plt.close()
    return

df = pd.read_excel(df_in, header=[0,1],index_col=0)
#print(df.columns)
df['MET_bin',1]=pd.cut(df['MET',1],bins=met_marks,labels=met_labels)
df['MET_bin',2]=pd.cut(df['MET',2],bins=met_marks,labels=met_labels)

################### Filter 1a ##############################
df['r'] = (df['r']/1000)/ATLASdet_radius
events = df.shape[0]
df['pair_r']=0
df.loc[(df['r'][1]<1.) & (df['r'][2]>=1.),'pair_r'] = 1
df.loc[(df['r'][2]<1.) & (df['r'][1]>=1.),'pair_r'] = 1
df.loc[(df['r'][1]>=1.) & (df['r'][2]>=1.),'pair_r'] = 2
#print(df.pair_r.unique())

rs=[]
rs.append([df[df.pair_r==i]['r', 1].values for i in range(3)])
rs.append([df[df.pair_r==i]['r', 2].values for i in range(3)])

cuts = 4
delta = 1/cuts
factor = 1
maximum = np.amax(df['r'].values)
#print(maximum)
while factor < maximum:
    factor += 1

rbins = np.arange(0,factor+delta,delta)
rticks = np.arange(0,factor+delta*4,delta*4)

rlabels = range(1,len(rticks))
rlabels = [f'{int(x/(cuts/4))}r' if x%(cuts/4)==0 else f'{(x/(cuts/4))}r' for x in rlabels]
rlabels.insert(0,'0')
#print(rticks)
#print(rlabels)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,8))
for ix,col in enumerate(axs):
    col.hist(rs[ix],bins=rbins,histtype='step',
     label=["Both in", "1 in 1 out", "Both out"],color=['b','g','r'])
    col.set_title(f'Photon {ix+1}')
    col.set_ylim(0.5,events)
    col.set_xticks(rticks)
    col.set_xticklabels(rlabels)
    col.set_yscale('log')
    col.legend()
plt.suptitle(f'Filtro 1a: Dentro del radio\n{events} eventos')
plt.savefig(destiny_filters+'filter1a_1D.png')
#plt.show()
plt.close()

################ Filter 1b ####################

df = df.loc[df.pair_r==0]
df['z'] = (np.abs(df['z'])/1000)/ATLASdet_semilength
events = df.shape[0]
df['pair_z']=0
df.loc[(df['z'][1]<1.) & (df['z'][2]>=1.),'pair_z'] = 1
df.loc[(df['z'][2]<1.) & (df['z'][1]>=1.),'pair_z'] = 1
df.loc[(df['z'][1]>=1.) & (df['z'][2]>=1.),'pair_z'] = 2
#print(df.loc[:,['z','pair_z']])

rs=[]
rs.append([df[df.pair_z==i]['z', 1].values for i in range(3)])
rs.append([df[df.pair_z==i]['z', 2].values for i in range(3)])

cuts = 4
delta = 1/cuts
factor = 1
maximum = np.amax(df['z'].values)
#print(maximum)

while factor < maximum:
    factor += 1

jump = 8
rbins = np.arange(0,factor+delta,delta)
rticks = np.arange(0,factor+delta*jump,delta*jump)
#print(rticks)
rlabels = range(1,len(rticks))
rlabels = [f'{int(x/(cuts/jump))}z' if x%(cuts/jump)==0 else f'{(x/(cuts/jump))}z' for x in rlabels]
rlabels.insert(0,'0')
#print(rticks)
#print(rlabels)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,8))
for ix,col in enumerate(axs):
    col.hist(rs[ix],bins=rbins,histtype='step',
     label=["Both in", "1 in 1 out", "Both out"],color=['b','g','r'])
    col.set_title(f'Photon {ix+1}')
    col.set_ylim(0.5,events)
    col.set_xticks(rticks)
    col.set_xticklabels(rlabels)
    col.set_yscale('log')
    col.legend()
plt.suptitle(f'Filtro 1b: Dentro de la longitud z\n{events} eventos')
plt.savefig(destiny_filters+'filter1b_1D.png')
#plt.show()
plt.close()

#FILTRO 1 - CUTFLOW
df = df.loc[df.pair_z==0]
#df[df.z_origin > 20000].z_origin = 20000
df.z_origin = np.clip(df.z_origin,-5000,5000)
df.pz = np.clip(df.pz,-1000,1000)
events = df.shape[0]
plot_params_1(1,f'Dentro del detector - {events} eventos')
plot_params_2(1,f'Dentro del detector - {events} eventos')


################ Filter 2 ####################
df['pair_et']=0
df.loc[(df['ET'][1]<=50) & (df['ET'][2]>50),'pair_et'] = 1
df.loc[(df['ET'][2]<=50) & (df['ET'][1]>50),'pair_et'] = 1
df.loc[(df['ET'][1]>50) & (df['ET'][2]>50),'pair_et'] = 2
#print(df.loc[:,['ET','pair_et']])

rs=[]
rs.append([df[df.pair_et==i]['ET', 1].values for i in range(3)])
rs.append([df[df.pair_et==i]['ET', 2].values for i in range(3)])

cuts = 1/10
delta = 1/cuts
factor = 1
maximum = np.amax(df['ET'].values)
#print(maximum)

while factor < maximum:
    factor += 1

jump = 5
rbins = np.arange(0,factor+delta,delta)
rticks = np.arange(0,factor+delta*jump,delta*jump)
#print(rticks)
rlabels = range(len(rticks))
rlabels = [f'{(x/(cuts/jump))}' if x > 0 else '0' for x in rlabels]
#print(rticks)
#print(rlabels)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,8))
for ix,col in enumerate(axs):
    col.hist(rs[ix],bins=rbins,histtype='step',
     label=["Both less", "1 more 1 less", "Both more"],color=['r','g','b'])
    col.set_title(f'Photon {ix+1}')
    col.set_ylim(0.5,events)
    col.set_xticks(rticks)
    col.set_xticklabels(rlabels)
    col.set_yscale('log')
    col.legend()

plt.suptitle(f'Filtro 2: ET mayor a 50 GeV\n{events} eventos')
plt.savefig(destiny_filters+'filter2_1D.png')
#plt.show()
plt.close()

#FILTRO 2 - CUTFLOW
df = df.loc[df.pair_et == 2]
events = df.shape[0]
plot_params_1(2,f'ET > a 50 GeV - {events} eventos')
plot_params_2(2,f'ET > a 50 GeV - {events} eventos')

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
maximum = np.amax(df['rel_tof'].values)
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
maximum = np.amax(df['abs_eta'].values)
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
maximum = np.amax(df['abs_eta'].values)
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
print(f'# EVENTOS FINAL: {events}')
