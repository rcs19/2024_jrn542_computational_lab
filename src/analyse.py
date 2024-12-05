from epic1d import *
from noise import *
import pandas as pd
import os

def generate():
    ncells_list = arange(110,201,10)
    for ncells in ncells_list:
        Run_and_Save(npart=5000, ncells=ncells)

def plot_directory():
    directory = 'savedata/session2_variations_05-12-2024'

    # Dataframe (Table.csv) format
    columns = ['npart','L','ncells','runtime','damp', 'damp_std', 'snr', 'omega', 'omega_std']
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
        row.extend(Signal_vs_Noise(filepath, show_plot=False))
        row.extend(Find_Frequency(filepath, show_plot=False))
        rows.append(row)
        
    df = pd.DataFrame(rows, columns=columns)
    df = df.sort_values(by='ncells',ignore_index=True)
    print(df)

plot_directory()