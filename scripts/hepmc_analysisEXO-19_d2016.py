#This calculates H_T, missing energy and others for EXO-19 analysis. If N decays in the detector, then it vector-sums the pT of everything that is not a lepton. If N decays in the detector, then it vector-sums the pT of everything that is not a neutrino.

import math

df = open("hepmc_corrected.txt", "r")
#df = open("hepmc.txt", "r")	#Test file
fileToF=open("dataEXO19v3.txt","w")

vertexid='cualquiercosa'
vertexid2='huevadaymedia'
evento_N=0
evento_N2=0
p_all=0
bigToF=0
smolToF=100
M4=50	#GeV
width=4.99045323E-16	#particle decay width
hbar=6.582119514*10**(-25)	#GeV*s
c_speed=299792458	#m/s
ATLASdet_radius=1.4	#detector radial length to ECAL(m)
CMSdet_radius=1.29
ATLASdet_length=3
CMSdet_length=3
life_time=hbar/width
M_conversion=1.78266192*10**(-27)	#GeV to kg
p_conversion=5.344286*10**(-19)	#GeV to kg.m/s
decay_inside=0
detectable_displaced=0

def time_of_flight(life_time_lab,p_photon,xyz,det_radius,det_length):	#Time between colision and photon getting to ECAL.
	p_photon_mod=math.sqrt(p_photon[0]**2+p_photon[1]**2+p_photon[2]**2)
	if p_photon[2]>=0:
		param_a=(det_length-xyz[2])/p_photon[2]
	else:
		param_a=(-det_length-xyz[2])/p_photon[2]
	x_line=xyz[0]+param_a*p_photon[0]
	y_line=xyz[1]+param_a*p_photon[1]
	deltaZ=abs((xyz[2]-p_photon[2]*(p_photon[0]*xyz[0]+p_photon[1]*xyz[1]+p_photon[2]*xyz[2])/(p_photon_mod**2))/(1-(p_photon[2]**2)/(p_photon_mod**2)))	#photon's reconstructed trajectory distance of closest approach to Z-axis
	decay_in_detector=(math.sqrt(xyz[0]**2+xyz[1]**2)<det_radius and abs(xyz[2])<det_length)	#boolean to check if N decays inside det.
	if math.sqrt(x_line**2+y_line**2)>=det_radius and decay_in_detector:	#ECAL barrel
		at=(c_speed**2)*(p_photon[0]**2+p_photon[1]**2)/(p_photon_mod**2)
		bt=2*c_speed*(xyz[0]*p_photon[0]+xyz[1]*p_photon[1])/(p_photon_mod)
		ct=xyz[0]**2+xyz[1]**2-det_radius**2
		a_alpha_det=p_photon[0]**2+p_photon[1]**2
		b_alpha_det=2*(xyz[0]*p_photon[0]+xyz[1]*p_photon[1])
		c_alpha_det=xyz[0]**2+xyz[1]**2-det_radius**2
		alpha_det=(-b_alpha_det+math.sqrt(b_alpha_det**2-4*a_alpha_det*c_alpha_det))/(2*a_alpha_det)
		theta=math.atan(det_radius/(abs(xyz[2]+alpha_det*p_photon[2])))
		eta_det=-math.log(math.tan(theta/2))	#detector's pseudorapidity
		time_prompt=math.sqrt(det_radius**2+(abs(xyz[2]+alpha_det*p_photon[2]))**2)/c_speed
		return(1,life_time_lab+(-bt+math.sqrt(bt**2-4*at*ct))/(2*at),eta_det,time_prompt,p_photon_mod,p_photon[0]**2+p_photon[1]**2,p_photon[2],math.sqrt(xyz[0]**2+xyz[1]**2+xyz[2]**2),deltaZ)
	elif math.sqrt(x_line**2+y_line**2)<det_radius and decay_in_detector:	#ECAL endcaps
		theta=math.atan((math.sqrt((xyz[0]+param_a*p_photon[0])**2+(xyz[1]+param_a*p_photon[1])**2))/det_length)
		eta_det=-math.log(math.tan(theta/2))
		time_prompt=math.sqrt((math.sqrt((xyz[0]+param_a*p_photon[0])**2+(xyz[1]+param_a*p_photon[1])**2))**2+det_length**2)/c_speed
		return(2,life_time_lab+p_photon_mod*param_a/c_speed,eta_det,time_prompt,p_photon_mod,p_photon[0]**2+p_photon[1]**2,p_photon[2],math.sqrt(xyz[0]**2+xyz[1]**2+xyz[2]**2),deltaZ) #(det_length-xyz[2])/abs(p_photon[2])
	else:
		return (0,0,0,0)

def tof_grand(tof,bigToF):
	if tof>bigToF:
		return tof

def tof_petite(tof,smolToF):
	if tof<smolToF:
		return tof

###########################################################################################################

tof=[0,0]
tof2=[0,0]
pT_photon=0
pT_photon2=0
df.readline()

for line in df:
	if line.split()[0]=='E':	#Esto tiene que resetear todo.
		if (tof[0] or tof2[0]) and not(tof[0] and tof2[0]):	#Si se tiene 1 fotón en el estado final:
			phot_num=1
			et_miss=math.sqrt(p_miss[0]**2+p_miss[1]**2)
			if tof[0]:
				#HT, et_miss, number of photons, pT_photon1, eta_det_photon1, tof_photon1, tof_prompt_photon1, N_dist_vuelo, deltaZ, event_number
				fileToF.write(str(jet_HT)+" "+str(et_miss)+" "+str(phot_num)+" "+str(math.sqrt(tof[5]))+" "+str(tof[2])+" "+str(tof[1])+" "+str(tof[3])+" "+str(tof[7])+" "+str(tof[8])+" "+str(evento_N2)+"\n")
			if tof2[0]:
				#HT, et_miss, number of photons, pT_photon1, eta_det_photon1, tof_photon1, tof_prompt_photon1, N_dist_vuelo, deltaZ, event_number
				fileToF.write(str(jet_HT)+" "+str(et_miss)+" "+str(phot_num)+" "+str(math.sqrt(tof2[5]))+" "+str(tof2[2])+" "+str(tof2[1])+" "+str(tof2[3])+" "+str(tof2[7])+" "+str(tof2[8])+" "+str(evento_N2)+"\n")
		if (tof[0] and tof2[0]): #Si se tiene 2 fotones en el estado final:
			phot_num=2
			et_miss=math.sqrt(p_miss[0]**2+p_miss[1]**2)
			# HT, et_miss, number of photons, pT_photon1, eta_det_photon1, tof_photon1, tof_prompt_photon1, N_dist_vuelo, deltaZ1, pT_photon2, eta_det_photon2, tof_photon2, tof_prompt_photon2, N_dist_vuelo2, deltaZ2, event_number
			fileToF.write(str(jet_HT)+" "+str(et_miss)+" "+str(phot_num)+" "+str(math.sqrt(tof[5]))+" "+str(tof[2])+" "+str(tof[1])+" "+str(tof[3])+" "+str(tof[7])+" "+str(tof[8])+" "+str(math.sqrt(tof2[5]))+" "+str(tof2[2])+" "+str(tof2[1])+" "+str(tof2[3])+" "+str(tof2[7])+" "+str(tof2[8])+" "+str(evento_N2)+"\n")
		evento_N+=1
		evento_N2=line.split()[1]
		flagv1=0	#False
		flagv2=0
		flag2=0
		jet_HT=0
		p_miss=[0,0,0]
		vertexid='cualquiercosa'
		vertexid2='huevadaymedia'
		if (evento_N*100/10000)%5==0:
			print("Process: ",evento_N/100,"%")
		continue
	if line.split()[0]=='P' and line.split()[8]=='1' and not (line.split()[2]=='12' or line.split()[2]=='14' or line.split()[2]=='16' or line.split()[2]=='18' or line.split()[2]=='11' or line.split()[2]=='13' or line.split()[2]=='15' or line.split()[2]=='17' or line.split()[2]=='9900012' or line.split()[2]=='9900016' or line.split()[2]=='9900014'):
		jet_HT=jet_HT+math.sqrt(float(line.split()[3])**2+float(line.split()[4])**2)
	if line.split()[0]=='P' and line.split()[8]=='1' and not (line.split()[2]=='12' or line.split()[2]=='14' or line.split()[2]=='16' or line.split()[2]=='18'):
		p_miss=[p_miss[0]+float(line.split()[3]),p_miss[1]+float(line.split()[4]),p_miss[2]+float(line.split()[5])]
	if line.split()[0]=='P' and (line.split()[2]=='9900014' or line.split()[2]=='9900012' or line.split()[2]=='9900016'):
		if flag2:
			p_N2=[float(line.split()[3]),float(line.split()[4]),float(line.split()[5])]
			#energy2=float(line.split()[6])
			#life_time_lab2=life_time*energy2/M4
			abs_p_N2=math.sqrt(p_N2[0]**2+p_N2[1]**2+p_N2[2]**2)
			p_all=math.sqrt(p_N2[0]**2+p_N2[1]**2+p_N2[2]**2)+p_all
			vertexid2=line.split()[11]
			etaN2=math.atanh(p_N2[2]/abs_p_N2)
			phiN2=math.atan(p_N2[1]/p_N2[0])
			flag2=0
			continue
		if not flag2:
			p_N=[float(line.split()[3]),float(line.split()[4]),float(line.split()[5])]
			#energy=float(line.split()[6])
			#life_time_lab=life_time*energy/M4
			abs_p_N=math.sqrt(p_N[0]**2+p_N[1]**2+p_N[2]**2)
			p_all=math.sqrt(p_N[0]**2+p_N[1]**2+p_N[2]**2)+p_all
			vertexid=line.split()[11]
			etaN=math.atanh(p_N[2]/abs_p_N)
			phiN=math.atan(p_N[1]/p_N[0])
			flag2=1
			continue
		continue
	if line.split()[0]=='V' and line.split()[1]==vertexid:
		xyz_photonvertex=[float(line.split()[3])*0.001,float(line.split()[4])*0.001,float(line.split()[5])*0.001]	#meters
		n_speed2=math.sqrt(((abs_p_N*p_conversion)**2)/((M4*M_conversion)**2+((abs_p_N*p_conversion)**2)/(c_speed**2)))
		life_time_lab_n_speed=math.sqrt(xyz_photonvertex[0]**2+xyz_photonvertex[1]**2+xyz_photonvertex[2]**2)/n_speed2
		flagv1=1	#True
		continue
	if line.split()[0]=='P' and line.split()[2]=='22' and flagv1:
		p_photon=[float(line.split()[3]),float(line.split()[4]),float(line.split()[5])]
		pT_photon=math.sqrt(p_photon[0]**2+p_photon[1]**2)
		p_photon_mod=math.sqrt(p_photon[0]**2+p_photon[1]**2+p_photon[2]**2)
		flagv1=0
		eta=math.atanh(p_photon[2]/p_photon_mod)	#photon's pseudorapidity
		phi=math.atan(p_photon[1]/p_photon[0])
		deltaR=math.sqrt((eta-etaN)**2+(phi-phiN)**2)
		#Calculating time of flight:
		tof=time_of_flight(life_time_lab_n_speed,p_photon,xyz_photonvertex,ATLASdet_radius,ATLASdet_length)
		if tof[0]:	#If TRUE then it decays inside detector.
			#fileToF.write(str(tof[0])+" "+str(tof[1])+" "+str(tof[2])+" "+str(tof[3])+" "+str(tof[4])+" "+str(tof[5])+" "+str(tof[6])+" "+str(deltaR)+"\n")	#write
			decay_inside+=1
			if tof[1]>tof[3]+3E-9:
				detectable_displaced+=1
	if line.split()[0]=='V' and line.split()[1]==vertexid2:
		xyz_photonvertex2=[float(line.split()[3])*0.001,float(line.split()[4])*0.001,float(line.split()[5])*0.001]
		n_speed2=math.sqrt(((abs_p_N2*p_conversion)**2)/((M4*M_conversion)**2+((abs_p_N2*p_conversion)**2)/(c_speed**2)))
		life_time_lab_n_speed2=math.sqrt(xyz_photonvertex2[0]**2+xyz_photonvertex2[1]**2+xyz_photonvertex2[2]**2)/n_speed2
		flagv2=1	#True
		continue
	if line.split()[0]=='P' and line.split()[2]=='22' and flagv2:
		p_photon2=[float(line.split()[3]),float(line.split()[4]),float(line.split()[5])]
		pT_photon2=math.sqrt(p_photon2[0]**2+p_photon2[1]**2)
		p_photon_mod2=math.sqrt(p_photon2[0]**2+p_photon2[1]**2+p_photon2[2]**2)
		flagv2=0
		eta2=math.atanh(p_photon2[2]/p_photon_mod2)	#photon's pseudorapidity
		phi2=math.atan(p_photon2[1]/p_photon2[0])
		deltaR2=math.sqrt((eta2-etaN2)**2+(phi2-phiN2)**2)
		tof2=time_of_flight(life_time_lab_n_speed2,p_photon2,xyz_photonvertex2,ATLASdet_radius,ATLASdet_length)
		if tof2[0]:
			#fileToF.write(str(tof2[0])+" "+str(tof2[1])+" "+str(tof2[2])+" "+str(tof2[3])+" "+str(tof2[4])+" "+str(tof2[5])+" "+str(tof2[6])+" "+str(deltaR2)+"\n")	#write
			decay_inside+=1
			if tof2[1]>tof2[3]+3E-9:
				detectable_displaced+=1


df.close()
fileToF.close()
p_avg=p_all/evento_N
print("Average 3-momentum: ", p_avg, "GeV or ", p_avg*1.602176634*(10**(-10))/(c_speed),"kg.m/s") #201.9273851846905 GeV
print("Number of events: ", evento_N)
print("Largest/Smallest ToF: ", bigToF,"   /   ",smolToF)
print("# of photons decaying inside the detector: ", decay_inside)
print("# of photons time>prompt+3ns: ", detectable_displaced, " --------- Ratio:",detectable_displaced/decay_inside)


