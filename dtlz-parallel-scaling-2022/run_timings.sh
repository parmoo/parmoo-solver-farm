#!/bin/bash
mkdir -p parmoo-dtlz2
# serial runs
for SEED in 0 1 2 3 4
do
	for SIZE in 8 16 32
	do
		python3 parmoo_solve_dtlz2_serial.py $SIZE $SEED
	done
done
# 2 thread runs (save to same file, ordered by numthreads)
for SEED in 0 1 2 3 4
do
	for SIZE in 8 16 32
	do
		python3 parmoo_solve_dtlz2_parallel.py $SIZE $SEED --comms local --nworkers 3
	done
done
# 4 thread runs (save to same file, ordered by num threads)
for SEED in 0 1 2 3 4
do
	for SIZE in 8 16 32
	do
		python3 parmoo_solve_dtlz2_parallel.py $SIZE $SEED --comms local --nworkers 5
	done
done
# 8 thread runs (save to same file, ordered by num threads)
for SEED in 0 1 2 3 4
do
	for SIZE in 8 16 32
	do
		python3 parmoo_solve_dtlz2_parallel.py $SIZE $SEED --comms local --nworkers 9
	done
done
# pymoo for comparison
mkdir -p pymoo-dtlz2
for SEED in 0 1 2 3 4
do
	python3 pymoo_solve_dtlz2_serial.py $SEED
done
