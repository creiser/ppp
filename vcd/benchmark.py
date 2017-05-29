
import subprocess
import re
import pandas as pd
import numpy as np

def extract_time(output, section):
    return float(re.search(section + ': ([-+]?[0-9]*\.?[0-9]+)', output, re.IGNORECASE).group(1))


def run_benchmark(name, cmds, num_iterations):
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

	df = pd.DataFrame({'execution_time' : pd.Series(execution_time),
		               'execution_time_stdev' : pd.Series(execution_time_stdev)})
	df.to_csv(name + '.csv')
	
distributed_cmds = []
for num_nodes in [2, 4, 6, 8, 16]:
	distributed_cmds.append(
		["srun",
		 "-n " + str(num_nodes),
		 "-N " + str(num_nodes),
		 "--constraint=zeus", 
		 "./vcd", "in.pgm", "-m 3"])
#run_benchmark('distributed', distributed_cmds, 5);

# This only uses the parallel version, but not the distributed one.
# Zeus has only 10 cores, so maybe that's the reason why there is no further speedup
# when requesting more than 10 cores.
# Alternative: Try to use e.g. 8 cores from first node and another 8 cores from second node.		 
parallel_cmds = []
for num_cores in [2, 4, 6, 8, 10, 12, 14, 16]:
	parallel_cmds.append(
		["srun",
		 "-n 1",
		 "-c " + str(num_cores),
		 "--constraint=zeus", 
		 "./vcd", "in.pgm", "-m 2"])
run_benchmark('parallel', parallel_cmds, 10);
		 
