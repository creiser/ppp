pgm_conv: pgm_conv.o
	mpicc -fopenmp -o pgm_conv pgm_conv.o -L/home/ppp2017/ppp_pnm -lppp_pnm
comp: comp.o
	gcc -fopenmp -o comp comp.o -L/home/ppp2017/ppp_pnm -lppp_pnm
pgm_conv.o: pgm_conv.c
	mpicc -Wall -fopenmp -I/home/ppp2017/ppp_pnm -c pgm_conv.c
comp.o: comp.c
	gcc -Wall -fopenmp -I/home/ppp2017/ppp_pnm -c comp.c
clean_pgm_conv:
	rm -f pgm_conv.o
clean_comp:
	rm -f comp.o
clean_out:
	rm -f *.o
clean_all:
	rm -f pgm_conv.o pgm_conv comp.o comp