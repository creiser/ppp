from subprocess import call
for num_cores in range(1, 16):
    call(["srun", "-n 1",
                  "-N 1",
                  "-c" + str(num_cores), 
                  "--cpu_bind=core", 
                  "--constraint=zeus", 
                  "./scalepgm", "huge.pgm", "200", "255", "1"])