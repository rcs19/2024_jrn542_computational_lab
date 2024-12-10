
from noise import *
import os 

directory = "savedata/"

columns = ['damp', 'damp_std', 'snr', 'omega', 'omega_std']
rows = []

for filename in os.listdir(directory):
    if not filename.endswith(".txt"):
        continue                    
    try:
        filepath = os.path.join(directory, filename).replace('\\','/')   # example string: 'data/P1_B1_Idefault_Ndefault.mat'
    except:
        print(f"Error file not found.")
        break
    
    print(f"\n{filename}")
    
    row = []
    row.extend(Signal_vs_Noise(filepath, show_plot=False))
    row.extend(Find_Frequency(filepath, show_plot=False))
    
    rows.append(row)
    
df = pd.DataFrame(rows, columns=columns)

damp_avg = df['damp'].mean()
damp_avg_std = 1/len(df)*np.abs(np.linalg.norm(df['damp_std']))

omega_avg = df['omega'].mean()
omega_avg_std = 1/len(df)*np.abs(np.linalg.norm(df['omega_std']))

print(f"\nMeans:\nDamping Rate = {damp_avg:.3f} ± {damp_avg_std:.4f}\nAngular Frequency = {omega_avg:.3f} ± {omega_avg_std:.3f}")
