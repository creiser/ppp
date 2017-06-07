
import subprocess
import re
import pandas as pd
import numpy as np

def extract_time(output, section):
    return float(re.search(section + ': ([-+]?[0-9]*\.?[0-9]+)', output, re.IGNORECASE).group(1))

def run_benchmark(name, row_ids, cmds, num_iterations):
	execution_time =  []
	execution_time_stdev = []
	for cmd in cmds:
		print(" ".join(cmd))
		t_execution_time = []
		for i in range(num_iterations):
		    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)                           
		    t_execution_time.append(extract_time(output, "vcd"))
		    print("time: " + str(t_execution_time[-1]) + " iteration: " + str(i + 1) + "/" +
		    	str(num_iterations))

		execution_time.append(np.mean(t_execution_time))
		execution_time_stdev.append(np.std(t_execution_time))
		print("mean: " + str(execution_time[-1]) +
			" stdev: " + str(execution_time_stdev[-1]))

	df = pd.DataFrame({'row_id:' : pd.Series(row_ids),
					   'execution_time' : pd.Series(execution_time),
		               'execution_time_stdev' : pd.Series(execution_time_stdev)})
	df.to_csv(name + '.csv')
	
# Sequential as base line
run_benchmark('sequential', [0], [[
	     "srun",
		 "-n 1",
		 "-N 1",
		 "--constraint=zeus", 
		 "./vcd", "in.pgm", "-m 1"]], 20);
	
# This only uses the parallel version, but not the distributed one.
# All cores should be taken from the same node. Zeus machines seem
# to be the only one from the FIM cluster, that can provide 16 cores
# as requested by the task description on a single machine, since Zeus
# machines have 2 processors with 8 cores each.
parallel_row_ids = []		 
parallel_cmds = []
for num_cores in [2, 4, 6, 8, 10, 12, 14, 16]:
	parallel_row_ids.append(num_cores)
	parallel_cmds.append(
		["srun",
		 "-n 1",
		 "-N 1",
		 "-c " + str(num_cores),
		 "--threads-per-core=1",
		 "--cpu_bind=cores",
		 "--constraint=zeus", 
		 "./vcd", "in.pgm", "-m 2"])
run_benchmark('parallel', parallel_row_ids, parallel_cmds, 20);

# Here we also let the algorithm run on multiple nodes of the Zeus cluster.
distributed_row_ids = []
distributed_cmds = []
for num_nodes in [2, 4, 6, 8]:
	for num_cores in [2, 4, 8, 16]:
		distributed_row_ids.append(str(num_nodes) + ";" + str(num_cores))
		distributed_cmds.append(
			["srun",
			 "-n " + str(num_nodes),
			 "-N " + str(num_nodes),
			 "-c " + str(num_cores),
		 	 "--threads-per-core=1",
			 "--cpu_bind=cores",
			 "--constraint=zeus", 
			 "./vcd", "in.pgm", "-m 3"])		  
run_benchmark('distributed', distributed_row_ids, distributed_cmds, 20);
		 
