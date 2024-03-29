#Script para mandar los hepmc de los runs a la carpeta correspondiente
#!/bin/bash

folder_destiny="/home/cristian/Desktop/HEP_Jones/scripts/data/raw"

tipos="VBF GF"
runs="01 02 03"
#runs="02"
for tipo in ${tipos}
	do
	#declare -a arr
	folder_origin="/home/cristian/Programs/MG5_aMC_v2_9_2/HN_${tipo}/Events"
	cd ${folder_origin}
	#arr=("$(ls -d */)")
	#arr=( */ )
	for it in ${runs}
		do
		run="run_${it}"
		cd "${run}"
		count="$(ls -1 *.hepmc 2>/dev/null | wc -l)"
		#echo "${count}"
		if [ $count == 0 ]
			then
			#echo "hola"
			file_gz=("$(ls -d *.hepmc.gz)")
			gzip -dk "${file_gz}"
		fi
		file_mc=("$(ls -d *.hepmc)")
		echo "${file_mc}"
		file_final="$(echo "${file_mc}" | sed 's/_pythia8_events//')"
		cp "${file_mc}" "${folder_destiny}/${file_final}"	
		cd ..
	done
done
