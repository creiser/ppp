.PHONY: all
all: libppp_pnm.a

libppp_pnm.a: ppp_pnm.o
	ar r $@ $?

ppp_pnm.o: ppp_pnm.c
	gcc -std=c99 -pedantic -Wall -fPIC -O3 -fomit-frame-pointer -c -o $@ $?

.PHONY: clean
clean:
	rm -f libppp_pnm.a ppp_pnm.o
