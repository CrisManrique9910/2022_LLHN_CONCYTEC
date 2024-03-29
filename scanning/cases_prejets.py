from pathlib import Path
import numpy as np

bosons = [22]
#bosons = [9,21] + list(range(23,26)) + list(range(32,38))
leptons = list(range(11,19))
heavy_n = [9900016,9900014,9900012]
vetos = bosons + leptons + heavy_n
#print(vetos)


n=6
iter=4
types=['VBF']
tevs=[13]
mass=10

cards = list(range(1,n+1))

for tev in tevs[:]:
    for type in types[:]:
        for card in cards[:]:

            destiny = f"./{mass}/{tev}/{type}/{iter}/raw/"
            Path(destiny).mkdir(exist_ok=True, parents=True)

            file_in = f"./{mass}/{tev}/{type}/{iter}/raw/{type}_iter{iter}_card{card}_{tev}.hepmc"
            file_out = f'prejets-{type}_iter{iter}_card{card}_{tev}.txt'

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
                        px, py, pz = [float(i) for i in line[3:6]]
                        pt = np.sqrt(px ** 2 + py ** 2)
                        theta = np.arctan2(pt, pz)
                        nu = -np.log(np.tan(theta / 2))
                        if abs(nu) <= 4.5:
                            #print(nu)
                            data = ' '.join(line[3:7])
                            #print(data)
                            prej.write(f'P {data}\n')

            df.close()
            prej.close()
