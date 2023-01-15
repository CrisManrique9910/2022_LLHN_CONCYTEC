from pathlib import Path
import pandas as pd
import sys
import os
import glob

types = ['VBF', 'GF']
cards = [13, 14, 15]
tevs = [13]

for type in types[:1]:
    for card in cards[:]:
        for tev in tevs[:]:

            # Programming Parameters

            root = "/home/cristian/Desktop/HEP_Jones/scripts_new/"
            origin = root + f"data/bins/{tev}/{type}/{card}/"
            destiny = root + f"data/bins/{tev}/{type}/{card}/"

            Path(destiny).mkdir(exist_ok=True, parents=True)
            os.system(f'rm {destiny}*.root')

            for in_file in sorted(glob.glob(origin + f"*.hepmc"))[:2]:
                out_file = in_file.replace('.hepmc','.root')
                #print(out_file)
                os.system('cd /home/cristian/Programs/MG5_aMC_v2_9_2/Delphes && ./DelphesHepMC '
                          f'cards/delphes_card_LLHN_ATLAS.tcl {out_file} {in_file}')
                #print(zbns)