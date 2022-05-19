from pathlib import Path
import numpy as np

bosons = [22]
#bosons = [9,21] + list(range(23,26)) + list(range(32,38))
leptons = list(range(11,19))
heavy_n = [9900016,9900014,9900012]
vetos = bosons + leptons + heavy_n
print(vetos)

destiny = "./data/raw/"
types = ['VBF']
cards = [13,14,15]
mass = {13:50,14:30,15:10}
tevs = [13]

for tev in tevs[:]:
    for type in types[:]:
        for card in cards[:]:

            Path(destiny).mkdir(exist_ok=True, parents=True)

            file_in = f"./data/raw/{type}_{card}_{tev}.hepmc"
            file_out = f'prejets-{type}_{card}_{tev}.txt'

            df = open(file_in, "r")
            prej = open(destiny + file_out, 'w')
            i= 0
            it = 0
            while it < 2:
                df.readline()
                it += 1

            #while i<3 :
            #    sentence = df.readline()
            for sentence in df:
                line = sentence.split()
                if line[0] == "E":
                    prej.write(f"Ev{i} px py pz E\n")
                    print(f"{type}_{card}_{tev} Event {i}")
                    i+=1
                elif line[0] == "P":
                    if (abs(int(line[2])) not in vetos) and (line[8] == '1'):
                        data = ' '.join(line[3:7])
                        #print(data)
                        prej.write(f'P {data}\n')

            df.close()
            prej.close()
