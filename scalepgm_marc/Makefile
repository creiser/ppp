pgm_conv: pgm_conv.o
	mpicc -o pgm_conv pgm_conv.o -L/home/ppp2017/ppp_pnm -lppp_pnm -fopenmp
pgm_conv.o: pgm_conv.c
	mpicc -Wall -fopenmp -c pgm_conv.c -I/home/ppp2017/ppp_pnm
clean:
	rm -f pgm_conv.o
clean_all:
	rm -f pgm_conv.o pgm_conv