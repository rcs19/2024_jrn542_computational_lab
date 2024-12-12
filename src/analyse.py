from epic1d import *
from noise import *
from joblib import Parallel, delayed
import pandas as pd
import os

def generate():
    npart_list = [1000,2000,3000,4000,5000,7500,10000,20000,30000,50000,75000,100000,125000,150000,175000,200000]
    for npart in npart_list:
        print(f"npart={npart}")
        Run_and_Save(npart=npart, ncells=20)

def generate_parallel(n_jobs=4):
    L_list = arange(1,20.1,0.5)*np.pi
    timestart=time()
    Parallel(n_jobs=n_jobs)(delayed(Run_and_Save)(npart=20000, cell_length=L, ncells=40) for L in L_list)
    timeend=time()
    print(f"Total Job Runtime (n_jobs={n_jobs}): {timeend-timestart}s")

def save_csv():
    directory = 'savedata/lcell_1pi-20pi_npart20000_ncell40'

    # Dataframe (Table.csv) format
    columns = ['npart','L','ncells','runtime','damp', 'damp_std', 'noise_level', 'omega', 'omega_std']
    rows = []
    
    for filename in os.listdir(directory):
        try:
            filepath = os.path.join(directory, filename).replace('\\','/')   # example string: 'data/P1_B1_Idefault_Ndefault.mat'
        except:
            print(f"Error file not found.")
            break

        print(filename)
        # Unpacking data from filename
        filename_elements = filename.split('_')      
        npart = int(filename_elements[0][1:])
        cell_length = float(filename_elements[1][1:-2])*pi
        ncells = int(filename_elements[2][6:])
        runtime = float(filename_elements[3][1:-4])
        
        # Building one row in table.csv
        row = [npart, cell_length, ncells, runtime]
        row.extend(Signal_vs_Noise(filepath, show_plot=False))  # Add damping rate, damping rate error, and noise level
        row.extend(Find_Frequency(filepath, show_plot=False))   # Add angular frequency and its error
        rows.append(row)
        
    df = pd.DataFrame(rows, columns=columns)
    df = df.sort_values(by=['npart','ncells','L'],ignore_index=True)
    df.to_csv("savedata/lcell_1pi-20pi_npart20000_ncell40.csv", index=False)  

save_csv()