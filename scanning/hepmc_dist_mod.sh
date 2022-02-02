#Script para mandar los hepmc de los runs a la carpeta correspondiente
#!/bin/bash

mass=$1
iter=$2
n=$3
tipo=$4

folder_destiny="/home/cristian/Desktop/HEP_Jones/scanning/${mass}/${tipo}/${iter}/raw"

#declare -a arr
folder_origin="/home/cristian/Programs/MG5_aMC_v2_9_2/HN_${tipo}_scan/Events"
cd ${folder_origin}
#arr=("$(ls -d */)")
#arr=( */ )
for it in $(seq 1 $n)
	do
	run="run_0${it}"
	cd "${run}"
	count="$(ls -1 *.hepmc 2>/dev/null | wc -l)"
	#echo "${count}"
	if [ $count == 0 ]
		then
		#echo "hola"
		file_gz=("$(ls -d *.hepmc.gz)")
		echo "${file_gz}"
		gzip -dk "${file_gz}"
	fi	
	
	file_mc=("$(ls -d *.hepmc)")
	file_final="$(echo "${file_mc}" | sed 's/_pythia8_events//')"
	cp "${file_mc}" "${folder_destiny}/${file_final}"
	cd ..
done

