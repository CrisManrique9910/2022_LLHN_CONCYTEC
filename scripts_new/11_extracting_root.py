import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import ROOT
from pathlib import Path

print('ROOT FIRST ATTEMPT:',ROOT.gSystem.Load("libDelphes"))
#print('ROOT SECON ATTEMPT:',ROOT.gSystem.Load("libDelphes"))
print('DELPHES CLASSES   :',ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"'))
print('EXRROT TREE READER:',ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"'))

types = ['VBF', 'GF']
cards = [13, 14, 15]
tevs = [13]

for type in types[:1]:
    for card in cards[:]:
        for tev in tevs[:]:

            origin = f"./data/bins/{tev}/{type}/{card}/"
            destiny = f"./data/bins/{tev}/{type}/{card}/"
            destiny_im = f"./cases/{tev}/{type}/{card}/ATLAS/only_Delphes/"

            Path(destiny_im).mkdir(exist_ok=True, parents=True)

            for input_file in sorted(glob.glob(origin + f"*.root")):

                out_file = input_file.replace('.root','.pickle')

                # Create chain of root trees
                chain = ROOT.TChain("Delphes")
                chain.Add(input_file)
    
                # Create object of class ExRootTreeReader
                treeReader = ROOT.ExRootTreeReader(chain)
                numberOfEntries = treeReader.GetEntries()
    
                # Get pointers to branches used in this analysis
                met = treeReader.UseBranch("MissingET")
                branchPhoton = treeReader.UseBranch("Photon")
                branchJet = treeReader.UseBranch("Jet")
    
                # Loop over all events
                photons = []
                jets = []
                print(f"\n{input_file}\nNumber of Entries: {numberOfEntries}")
                for entry in range(numberOfEntries):
                    # Load selected branches with data from specified event
                    treeReader.ReadEntry(entry)
                    miss = met[0].MET
    
                    for ph in branchPhoton:
                        if ph.PT > 10 and (abs(ph.Eta) <= 1.37 or 1.52 <= abs(ph.Eta) <= 2.37):
                        #if True:
                            photons.append({"N": entry, "E":ph.E, "pt":ph.PT, "eta":ph.Eta, 'phi': ph.Phi, 'MET': miss})

                    for jet in branchJet:
                        if jet.PT > 20 and abs(jet.Eta) <= 4.5:
                            jets.append({"N": entry, "pt": jet.PT, "eta": jet.Eta, 'phi': jet.Phi,
                                         'M': jet.Mass, 'MET': miss})
                df = pd.DataFrame(photons)
                df_jets = pd.DataFrame(jets)
                if (df_jets.shape[0] == 0) or (df.shape[0] == 0):
                    continue

                df = df.sort_values(by=['N', 'pt'], ascending=[True, False])
                g = df.groupby('N', as_index=False).cumcount() + 1
                df['id'] = g
                df = df.set_index(['N', 'id'])
                print(f'{100 * df.index.unique(0).size / numberOfEntries:2f} %')
                df.to_pickle(out_file)

                df_jets = df_jets.sort_values(by=['N', 'pt'], ascending=[True, False])
                g = df_jets.groupby('N', as_index=False).cumcount()
                df_jets['id'] = g
                df_jets = df_jets.set_index(['N', 'id'])
                #print(out_file.replace('.pickle','_jets.pickle'))
                #sys.exit()
                df_jets.to_pickle(out_file.replace('.pickle','_jets.pickle'))

                if 'All' in input_file:
                    nbins = 50
                    plt.hist(df.eta, bins=nbins)
                    plt.savefig(destiny_im + f'eta_all.jpg')
                    #plt.show()
                    plt.close()

                    plt.hist(df.pt, bins=nbins)
                    plt.savefig(destiny_im + f'PT_all.jpg')
                    #plt.show()
                    plt.close()

                    plt.hist(df.loc[pd.IndexSlice[:,1],'MET'], bins=nbins)
                    plt.savefig(destiny_im + f'MET_all.jpg')
                    #plt.show()
                    plt.close()


