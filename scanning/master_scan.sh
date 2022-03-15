#Script para asignar un valor de coupling the Higgs a n5 n5 en todos los param cards.
#!/bin/bash

mass=10
n=6
iter=4
type='VBF'
tev='13'

if [ "${type}" == "VBF" ]
	then
	/home/cristian/Programs/MG5_aMC_v2_9_2/bin/mg5_aMC mg5_launches_VBF.txt
else
	/home/cristian/Programs/MG5_aMC_v2_9_2/bin/mg5_aMC mg5_launches_GF.txt
fi 

bash param_dist_mod.sh "${mass}" "${iter}" "${n}" "${type}" "${tev}"

bash hepmc_dist_mod.sh "${mass}" "${iter}" "${n}" "${type}" "${tev}"
