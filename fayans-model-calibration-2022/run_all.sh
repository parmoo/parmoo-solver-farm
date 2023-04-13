#!/bin/bash

python3 parmoo_fayans_test.py --comms local --nworkers 4

for SEED in 0 1 2 3 4
do
	python3 parmoo_fayans_blackbox_solver.py --comms local --nworkers 4 --iseed $SEED
	python3 parmoo_fayans_structured_solver.py --comms local --nworkers 4 --iseed $SEED
done
