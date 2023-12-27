import pandas as pd
import os
import shutil


input_pickle_path = "/data/nuscenes/infos_val_10sweeps_withvelo_filter_True.pkl"
output_pickle_path = "/data/nuscenes/dump/infos_val_10sweeps_withvelo_filter_True.pkl"

old_dir = '/workspace/centerformer/data/nuscenes/'
new_dir = '/data/nuscenes/'

def dumpFile(file_path):
    new_path = newPath(file_path)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    shutil.copy(file_path, new_path)
    return

def newPath(file_path):
    return file_path.replace(old_dir, new_dir)

if __name__ == "__main__":
    datas = pd.read_pickle(input_pickle_path)
    
    for data in datas:
        for key in data.keys():
            if 'path' in key:
                file_path = data[key]
                #dumpFile(file_path)
                new_path = newPath(file_path)
                data[key] = new_path
                
            elif 'sweeps' == key:
                sweeps = data[key]
                for sweep in sweeps:
                    for key in sweep.keys():
                        if 'path' in key:
                            sweep_path = sweep[key]
                            # dumpFile(sweep_path)
                            new_sweep_path = newPath(sweep_path)
                            sweep[key] = new_sweep_path
    
    pd.to_pickle(datas, output_pickle_path)
    print("Done!")