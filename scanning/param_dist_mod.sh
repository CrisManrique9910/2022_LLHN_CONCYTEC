#Script para asignar un valor de coupling the Higgs a n5 n5 en todos los param cards.
#!/bin/bash

mass=$1
iter=$2
n=$3
type=$4

#source /home/cristian/anaconda3/etc/profile.d/conda.sh

folder_origin="/home/cristian/Desktop/HEP_Jones/scanning/${mass}/${type}/${iter}/cards"
config_path='/home/cristian/Programs/MG5_aMC_v2_9_2/HN_run_config.txt'

tevs="8"
ct="  0 = time_of_flight ! threshold (in mm) below which the invariant livetime is not written (-1 means not written)"
decay="   True  = cut_decays    ! Cut decay products "
pt_lim=" 10.0  = pta       ! minimum pt for the photons "
eta_lim=" 2.37  = etaa    ! max rap for the photons "

if [ "${type}" == "GF" ]
	then
	##############################
	# Primero GF
	##############################

	folder_destiny='/home/cristian/Programs/MG5_aMC_v2_9_2/HN_GF_scan/Cards'
	run_path="${folder_destiny}/run_card.dat"

	# Agrega los cortes al run_card
	sed -i "59s/.*/${ct}/" "${run_path}"
	sed -i "94s/.*/${decay}/" "${run_path}"
	sed -i "100s/.*/${pt_lim}/" "${run_path}"
	sed -i "112s/.*/${eta_lim}/" "${run_path}"

	for tev in $tevs
		do
		tev_="$((tev*1000/2))"
		# Define las energias de los beams en el run_card
		beam1="     ${tev_}.0     = ebeam1  ! beam 1 total energy in GeV"
		beam2="     ${tev_}.0     = ebeam2  ! beam 2 total energy in GeV"
		sed -i "35s/.*/${beam1}/" "${run_path}"
		sed -i "36s/.*/${beam2}/" "${run_path}"
		
		for ix in $(seq 1 $n)
			do
			echo "${ix}\n"
			# Copia el param_card correspondiente
			filename_o="${folder_origin}/param_card_${ix}.dat"
			filename_d="${folder_destiny}/param_card.dat"
			cp "${filename_o}" "${filename_d}" 

			# Le da el tag apropiado al run
			tag="  GF_iter${iter}_card${ix}_${tev}     = run_tag ! name of the run "
			sed -i "21s/.*/${tag}/" "${run_path}"

			# Correr el run
			cd "${folder_destiny}"
			cd ..
			./bin/madevent "${config_path}"
		done
	done
else
	#################################
	# Ahora para VBF
	#################################
	folder_destiny='/home/cristian/Programs/MG5_aMC_v2_9_2/HN_VBF_scan/Cards'
	run_path="${folder_destiny}/run_card.dat"

	# Agrega los cortes al run_card
	sed -i "59s/.*/${ct}/" "${run_path}"
	sed -i "94s/.*/${decay}/" "${run_path}"
	sed -i "101s/.*/${pt_lim}/" "${run_path}"
	sed -i "115s/.*/${eta_lim}/" "${run_path}"

	for tev in $tevs
		do
		tev_="$((tev*1000/2))"
		# Define las energias de los beams en el run_card
		beam1="     ${tev_}.0     = ebeam1  ! beam 1 total energy in GeV"
		beam2="     ${tev_}.0     = ebeam2  ! beam 2 total energy in GeV"
		sed -i "35s/.*/${beam1}/" "${run_path}"
		sed -i "36s/.*/${beam2}/" "${run_path}"
		
		for ix in $(seq 1 $n)
			do
			# Copia el param_card correspondiente
			filename_o="${folder_origin}/param_card_${ix}.dat"
			filename_d="${folder_destiny}/param_card.dat"
			cp "${filename_o}" "${filename_d}" 

			# Le da el tag apropiado al run
			tag="  VBF_iter${iter}_card${ix}_${tev}     = run_tag ! name of the run "
			sed -i "21s/.*/${tag}/" "${run_path}"

			# Correr el run
			cd "${folder_destiny}"
			cd ..
			./bin/madevent "${config_path}"
		done
	done
fi
