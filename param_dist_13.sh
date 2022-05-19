#Script para asignar un valor de coupling the Higgs a n5 n5 en todos los param cards.
#!/bin/bash


folder_origin='/home/cristian/Desktop/HEP_Jones/param_c'
config_path='/home/cristian/Programs/MG5_aMC_v2_9_2/HN_run_config.txt'

tevs="13"

small="  1e-12 = small_width_treatment"

nevents="  100000 = nevents ! Number of unweighted events requested "
ct="  0 = time_of_flight ! threshold (in mm) below which the invariant livetime is not written (-1 means not written)"
decay="   True  = cut_decays    ! Cut decay products "
pt_lim=" 10.0  = pta       ! minimum pt for the photons "
eta_lim=" 2.37  = etaa    ! max rap for the photons "
ptj1_lim=" 30.0   = ptj1min ! minimum pt for the leading jet in pt"
ptj2_lim=" 30.0   = ptj2min ! minimum pt for the second jet in pt"
ht_lim=" 200.0   = htjmin ! minimum jet HT=Sum(jet pt)"
mi_lim=" 750.0   = mmjj    ! min invariant mass of a jet pair "
deta_lim=" 4.0   = deltaeta ! minimum rapidity for two jets in the WBF case"
drjj_lim=" 0.4 = drjj    ! min distance between jets" 
draa_lim=" 0.2 = draa    ! min distance between gammas" 
draj_lim=" 0.2 = draj    ! min distance between gamma and jet" 
dR_lim="  0.2 = R0gamma ! Radius of isolation code"

###############################
## Primero GF
###############################
#folder_destiny='/home/cristian/Programs/MG5_aMC_v2_9_2/HN_GF/Cards'
#run_path="${folder_destiny}/run_card.dat"

##Cambia el small_width_treatment
#sed -i "17s/.*/${small}/" "${run_path}"

## Agrega los cortes al run_card
#sed -i "59s/.*/${ct}/" "${run_path}"
#sed -i "94s/.*/${decay}/" "${run_path}"
#sed -i "100s/.*/${pt_lim}/" "${run_path}"
#sed -i "112s/.*/${eta_lim}/" "${run_path}"
#sed -i "120s/.*/${draa_lim}/" "${run_path}"

#for tev in $tevs
#	do
#	tev_="$((tev*1000/2))"
#	# Define las energias de los beams en el run_card
#	beam1="     ${tev_}.0     = ebeam1  ! beam 1 total energy in GeV"
#	beam2="     ${tev_}.0     = ebeam2  ! beam 2 total energy in GeV"
#	sed -i "35s/.*/${beam1}/" "${run_path}"
#	sed -i "36s/.*/${beam2}/" "${run_path}"
#	for i in {13..15}
#		do
#		# Copia el param_card correspondiente
#		filename_o="${folder_origin}/param_card_${i}_13.dat"
#		filename_d="${folder_destiny}/param_card.dat"
#		cp "${filename_o}" "${filename_d}" 
#		
#		# Le da el tag apropiado al run
#		tag="  GF_${i}_${tev}     = run_tag ! name of the run "
#		sed -i "21s/.*/${tag}/" "${run_path}"
#		
#		# Correr el run
#		cd "${folder_destiny}"
#		cd ..
#		./bin/madevent "${config_path}"
#	done
#done

#################################
# Ahora para VBF
#################################
folder_destiny='/home/cristian/Programs/MG5_aMC_v2_9_2/HN_VBF/Cards'
run_path="${folder_destiny}/run_card.dat"

#Cambia el small_width_treatment
sed -i "17s/.*/${small}/" "${run_path}"

# Agrega los cortes al run_card
sed -i "26s/.*/${nevents}/" "${run_path}"
sed -i "59s/.*/${ct}/" "${run_path}"
sed -i "94s/.*/${decay}/" "${run_path}"
sed -i "101s/.*/${pt_lim}/" "${run_path}"
sed -i "115s/.*/${eta_lim}/" "${run_path}"
sed -i "123s/.*/${drjj_lim}/" "${run_path}"
sed -i "124s/.*/${draa_lim}/" "${run_path}"
sed -i "125s/.*/${draj_lim}/" "${run_path}"
sed -i "132s/.*/${mi_lim}/" "${run_path}"
sed -i "156s/.*/${ptj1_lim}/" "${run_path}"
sed -i "157s/.*/${ptj2_lim}/" "${run_path}"
sed -i "164s/.*/${ht_lim}/" "${run_path}"
sed -i "174s/.*/${dR_lim}/" "${run_path}"
sed -i "182s/.*/${deta_lim}/" "${run_path}"

for tev in $tevs
	do
	tev_="$((tev*1000/2))"
	# Define las energias de los beams en el run_card
	beam1="     ${tev_}.0     = ebeam1  ! beam 1 total energy in GeV"
	beam2="     ${tev_}.0     = ebeam2  ! beam 2 total energy in GeV"
	sed -i "35s/.*/${beam1}/" "${run_path}"
	sed -i "36s/.*/${beam2}/" "${run_path}"
	for i in {13..15}
		do
		# Copia el param_card correspondiente
		filename_o="${folder_origin}/param_card_${i}_13.dat"
		filename_d="${folder_destiny}/param_card.dat"
		cp "${filename_o}" "${filename_d}" 
		
		# Le da el tag apropiado al run
		tag="  VBF_${i}_${tev}_100k     = run_tag ! name of the run "
		sed -i "21s/.*/${tag}/" "${run_path}"
		
		# Correr el run
		cd "${folder_destiny}"
		cd ..
		./bin/madevent "${config_path}"
	done
done
