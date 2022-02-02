import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def plot_config(figsize,xtitle,ytitle,xsize, ysize, xtsize, ytsize):
    plt.figure(figsize=figsize)
    plt.ylabel(ytitle, fontsize=ysize)
    plt.xlabel(xtitle, fontsize=xsize)
    plt.xticks(fontsize=xtsize)
    plt.yticks(fontsize=ytsize)
    return

file_in = './data/clean/recollection_v4.json'

destiny_info = './data/clean/'
destiny_ims0 = './images/'

file_out = 'photons_data.json'

CMSdet_radius = 1.29 # meters
CMSdet_semilength = 2.935
ATLASdet_radius= 1.775
ATLASdet_semilength = 4.050

mass_conversion = 1.78266192*10**(-27)	#GeV to kg
p_conversion = 5.344286*10**(-19)	#GeV to kg.m/s
c_speed = 299792458	#m/s

with open(file_in, 'r') as file:
    data = json.load(file)



def pipeline(detec_radius, detec_semilength, detec_name):

    filename = f'{destiny_info}{detec_name}_{file_out}'

    z_origin = []
    pts = []
    pzs = []
    tofs = []
    tofs_b = []
    tofs_e = []
    p_tofs = []
    rel_tofs = []
    nus = []
    tns = []
    tphs = []
    trs = []
    tzs = []
    cofre = []
    photon_rz = []
    counter = 0
    dicts = []
    for event in list(data.keys())[:]:
        print(f'{detec_name} - Event {event}')
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
        photons = []
        pairs = []
        #print(pairs)
        pt_dum = 0
        ix = 1
        if len(holder['a']>2):
            continue
        for photon in holder['a']:
            info = dict()
            info['event'] = event

            vertex = str(photon[-1])
            px, py, pz = [p_scaler*ix for ix in photon[0:3]]
            x, y, z = [d_scaler*ix for ix in holder['v'][vertex][0:3]]
            mass_ph = photon[-2] * p_scaler
            r = np.sqrt(x ** 2 + y ** 2)
            # Calculating transverse momentum
            pt = c
            Et = np.sqrt(mass_ph ** 2 + pt ** 2)
            #
            if ix == 1:
                pt_dum = Et
                info['photon'] = ix
            elif pt_dum < Et:
                info['photon'] = 0
            else:
                info['photon'] = ix

            ix += 1
            # print(mass_ph)
            photons.append([r,z])
            info['r'] = r
            info['z'] = z
            info['pt'] = pt
            info['pz'] = pz
            info['ET'] = Et

            if r >= (r_detec) or abs(z) >= (z_detec):
                 pairs.append(None)
                 info['z_origin'] = np.nan
                 info['rel_tof'] = np.nan
                 info['eta'] = np.nan
                 info['MET'] = np.nan
                 dicts.append(info)
                 continue

            counter += 1

            # Calculating the z_origin of each photon
            v_z = np.array([0, 0, 1])  # point in the z axis
            d_z = np.array([0, 0, 1])  # z axis vector

            v_ph = np.array([x, y, z])
            d_ph = np.array([px, py, pz])

            n = np.cross(d_z, d_ph)

            n_ph = np.cross(d_ph, n)

            c_z = v_z + (((v_ph - v_z) @ n_ph) / (d_z @ n_ph)) * d_z

            # Calculating the time of flight
            # TIme of the neutralino
            vertex_n = str(holder['n5'][vertex][-1])
            mass_n = holder['n5'][vertex][-2] * p_scaler
            # print(mass_n)
            px_n, py_n, pz_n = [p_scaler*ix for ix in holder['n5'][vertex][0:3]]
            x_n, y_n, z_n = [d_scaler*ix for ix in holder['v'][vertex_n][0:3]]
            # print(vertex_n)
            dist_n = np.sqrt((x - x_n) ** 2 + (y - y_n) ** 2 + (z - z_n) ** 2)
            p_n = np.sqrt(px_n ** 2 + py_n ** 2 + pz_n ** 2)

            prev_n = (p_n * p_conversion) / (mass_n * mass_conversion)
            v_n = (prev_n / np.sqrt(1 + (prev_n / c_speed) ** 2)) * 1000  # m/s to mm/s
            t_n = dist_n / v_n  # s
            t_n = t_n * (10 ** 9)  # ns
            # t_n=0
            # Now, time of the photon
            vx = (c_speed * px / np.linalg.norm(d_ph)) * 1000  # mm/s
            vy = (c_speed * py / np.linalg.norm(d_ph)) * 1000  # mm/s
            vz = (c_speed * pz / np.linalg.norm(d_ph)) * 1000  # mm/s

            tr = (-(x * vx + y * vy) + np.sqrt(
                (x * vx + y * vy) ** 2 + (vx ** 2 + vy ** 2) * (r_detec ** 2 - r ** 2))) / (
                     (vx ** 2 + vy ** 2))

            if tr < 0:
                tr = (-(x * vx + y * vy) - np.sqrt(
                    (x * vx + y * vy) ** 2 + (vx ** 2 + vy ** 2) * ((r_detec) ** 2 - r ** 2))) / (
                         (vx ** 2 + vy ** 2))

            tz = (np.sign(vz) * z_detec - z) / vz

            # Now we see which is the impact time
            if tr < tz:
                # rf = r_detec
                rf = r_detec
                zf = z + vz * tr
                t_ph = tr * (10 ** 9)

            elif tz < tr:
                rf = np.sqrt((y + vy * tz) ** 2 + (x + vx * tz) ** 2)
                zf = np.sign(vz) * z_detec
                t_ph = tz * (10 ** 9)

            else:
                rf = r_detec
                zf = np.sign(vz) * z_detec
                t_ph = tz * (10 ** 9)

            tof = t_ph + t_n

            prompt_tof = (10**9)*np.sqrt(rf**2+zf**2)/(c_speed*1000)
            rel_tof = tof - prompt_tof
            # (Pseudo)rapidity

            theta = np.arctan2(rf, zf)
            nu = -np.log(np.tan(theta / 2))

            z_origin.append(c_z[-1])
            pts.append(pt)
            pzs.append(pz)
            tofs.append(tof)
            if abs(nu) < abs(-np.log(np.tan(np.arctan2(r_detec, z_detec) / 2))):
                tofs_b.append(tof)
            else:
                tofs_e.append(tof)
            p_tofs.append(prompt_tof)
            rel_tofs.append(rel_tof)
            nus.append(nu)
            tns.append(t_n)
            tphs.append(t_ph)
            trs.append(tr * (10 ** 9))
            tzs.append(tz * (10 ** 9))

            info['z_origin']=c_z[-1]
            info['rel_tof']=rel_tof
            info['eta']=nu
            info['MET'] = holder['MET']
            pairs.append(info)
            dicts.append(info)

        #pairs = np.array(pairs)
        photon_rz.append(photons)
        cofre.append(pairs)
    #print(cofre)

    with open(filename, 'w') as file:
        json.dump(cofre,file, separators=(',', ':'), sort_keys=True, indent=4)

    print(f'Detected photons in {detec_name}: {counter}')

    with open(destiny_info + "photon_coordinates.json", 'w') as file:
        json.dump(photon_rz,file, separators=(',', ':'), sort_keys=True, indent=4)
    print(f'Photon coordinates saved')

    # Create the directory for the images
    destiny_ims = destiny_ims0 + detec_name
    Path(destiny_ims).mkdir(parents=True,exist_ok=True)


    df = pd.DataFrame(dicts)
    #print(df.event.unique()[:1])
    base = df.loc[df.photon==0,'event'].values
    df.loc[(df.event.isin(base)) & (df.photon == 1), 'photon'] = 2
    df.loc[df.photon==0,'photon'] = 1


    #print(df[['event','photon','pt']])

    #cols = df.columns[2:]
    df = df.pivot(index = 'event',columns='photon')
    print(np.sum(df['ET',1] >= df['ET',2] ))
    #df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)
    #print(df)
    #print(cols)
    df.to_excel(destiny_info+'photon_df.xlsx')
    # Grafiquitos
    lim = 7500
    nbins = 100

    plot_config((12, 8), 'Z_origin [mm]', 'Counts', 16, 16, 14, 14)
    plt.hist(z_origin, bins=nbins, color='C0')
    plt.savefig(f'{destiny_ims}/{detec_name}_photons_zorigin_wo_limits.jpg')
    plt.close()

    plot_config((12, 8), 'Z_origin [mm]', 'Counts', 16, 16, 14, 14)
    plt.hist(z_origin, bins=nbins, range=[-lim, lim], color='C0')
    plt.savefig(f'{destiny_ims}/{detec_name}_photons_zorigin_w_limits.jpg')
    plt.close()

    plot_config((12, 8), 'PT [GeV]', 'Counts', 16, 16, 14, 14)
    plt.hist(pts, bins=nbins, color='C1')
    plt.savefig(f'{destiny_ims}/{detec_name}_photons_pt.jpg')
    plt.close()

    plot_config((12, 8), 'PZ [GeV]', 'Counts', 16, 16, 14, 14)
    plt.hist(pzs, bins=nbins, color='C2')
    plt.savefig(f'{destiny_ims}/{detec_name}_photons_pz.jpg')
    plt.close()

    plot_config((12, 8), 'Time of flight [ns]', 'Counts', 16, 16, 14, 14)
    plt.hist(tofs, bins=nbins, color='C3')
    plt.savefig(f'{destiny_ims}/{detec_name}_photons_tof.jpg')
    plt.close()

    plot_config((12, 8), 'Time of flight [ns]', 'Counts', 16, 16, 14, 14)
    plt.hist([tofs_b,tofs_e], bins=nbins, histtype='step',color=['C3','C4'], label=['Barrel','Endcaps'])
    plt.legend(fontsize=14)
    plt.savefig(f'{destiny_ims}/{detec_name}_photons_tof_divided.jpg')
    plt.close()

    plot_config((12, 8), '(Pseudo)rapidity', 'Counts', 16, 16, 14, 14)
    plt.hist(nus, bins=nbins, color='C4')
    plt.savefig(f'{destiny_ims}/{detec_name}_photons_pseudorapidity.jpg')
    plt.close()

    plot_config((12, 8), 'Time of flight [ns]', '(Pseudo)rapidity', 16, 16, 14, 14)
    plt.scatter(tofs, [abs(x) for x in nus], s=0.05, color='C5')
    plt.scatter(p_tofs, [abs(x) for x in nus], s=0.05, color='C6')
    plt.savefig(f'{destiny_ims}/{detec_name}_photons_tof_vs_nu.jpg')
    plt.close()

    plot_config((12, 8), 'Relative Time of flight [ns]', 'Counts', 16, 16, 14, 14)
    plt.hist(rel_tofs, bins=nbins, color='C7')
    plt.savefig(f'{destiny_ims}/{detec_name}_photons_reltof.jpg')
    plt.close()

    plot_config((12, 8), 'Lifetime of Neutrino [ns]', 'Counts', 16, 16, 14, 14)
    plt.hist(tns, bins=nbins, color='C8')
    plt.savefig(f'{destiny_ims}/{detec_name}_neutrino_lifetime.jpg')
    plt.close()

    return

#pipeline(CMSdet_radius,CMSdet_semilength,'CMS')
pipeline(ATLASdet_radius,ATLASdet_semilength,'ATLAS')