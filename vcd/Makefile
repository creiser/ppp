vcd: vcd.o
	gcc -O3 -fopenmp -o vcd vcd.o -L/home/ppp2017/ppp_pnm -lppp_pnm -lm
vcd.o: vcd.c
	gcc -Wall -fopenmp -O3 -I/home/ppp2017/ppp_pnm -c vcd.c
clean:
	rm -f vcd.o
clean_all:
	rm -f vcd.o vcd
