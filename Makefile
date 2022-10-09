MPIDIR=$(PWD)
LIBS=-lm -DARMA_DONT_USE_WRAPPER -larmadillo
CC=g++
MPICC=mpic++
MPICFLAGS=-DMPIBART
CCOPS=-Wall -g -O3 -std=c++11







bdsbart: bdsbart.o 
	$(MPICC) -o bdsbart bdsbart.o  $(LIBS)
bdsbart.o: bdsbart.cpp
	$(MPICC) $(MPICFLAGS) $(CCOPS) -c bdsbart.cpp -o bdsbart.o
#--------------------------------------------------
clean:
	rm -f ./bdsbart
	rm -f *.o



