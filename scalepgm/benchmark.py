from subprocess import check_output
import re
import pandas as pd
import numpy as np

def extract_time(output, section):
    return float(re.search(section + ': ([-+]?[0-9]*\.?[0-9]+)', output, re.IGNORECASE).group(1))

load, min_max, scaling, save, total = [], [], [], [], []
load_stdev, min_max_stdev, scaling_stdev, save_stdev, total_stdev = [], [], [], [], []
for num_cores in range(1, 7):
    t_load, t_min_max, t_scaling, t_save, t_total = [], [], [], [], []
    for i in range(5):
        output = check_output(["srun",
                               "-n 1",
                               "-N 1",
                               "-c " + str(num_cores), 
                               "--cpu_bind=core", 
                               "--constraint=zeus", 
                               "./scalepgm", "huge.pgm", "200", "255", "1"])
        if False:
            output = check_output(["./scalepgm",
                                   "TrueMarble.pgm", "200", "255", "1"])                                
        t_load.append(extract_time(output, "load"))
        t_min_max.append(extract_time(output, "min-max"))
        t_scaling.append(extract_time(output, "scaling"))
        t_save.append(extract_time(output, "save"))
        t_total.append(extract_time(output, "total"))
    load.append(np.mean(t_load))
    min_max.append(np.mean(t_min_max))
    scaling.append(np.mean(t_scaling))
    save.append(np.mean(t_save))
    total.append(np.mean(t_total))
    
    load_stdev.append(np.std(t_load))
    min_max_stdev.append(np.std(t_min_max))
    scaling_stdev.append(np.std(t_scaling))
    save_stdev.append(np.std(t_save))
    total_stdev.append(np.std(t_total))

df = pd.DataFrame({'load' : pd.Series(load), 
                   'load_stdev' : pd.Series(load_stdev), 
                   'min_max' : pd.Series(min_max),
                   'min_max_stdev' : pd.Series(min_max_stdev),
                   'scaling' : pd.Series(scaling),
                   'scaling_stdev' : pd.Series(scaling_stdev),
                   'save' : pd.Series(save),
                   'save_stdev' : pd.Series(save_stdev),
                   'total' : pd.Series(total),
                   'total_stdev' : pd.Series(total_stdev)})
df.to_csv('benchmark.csv')
