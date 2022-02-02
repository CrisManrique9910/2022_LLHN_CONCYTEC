#Script para asignar un valor de coupling the Higgs a n5 n5 en todos los param cards.
#!/bin/bash

cd /home/cristian/Desktop/HEP_Jones/param_c

replace="      4 1.000000e-02 # gnh55"

for i in {1..9}
	do
	filename="param_card_${i}.dat"
	#touch 
	sed -i "16s/.*/${replace}/" "${filename}"
done

