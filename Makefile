build:
	mpicc -c tasks.c utils.c
	mpic++ -o a03 main.c tasks.o utils.o
