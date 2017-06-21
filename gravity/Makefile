
all: gravity_simulation.c 
	mpicc -O3 -fopenmp -Wall -o grav gravity_simulation.c \
	-I/scratch/ppp2017/ppp_pnm \
	-L/scratch/ppp2017/ppp_pnm -lppp_pnm -lm 

clean: 
	$(RM) grav gravity_simulation.o out.dat *.pbm
