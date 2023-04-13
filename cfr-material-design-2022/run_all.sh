#!/bin/bash

for SEED in 0 1 2 3 4
do
	python3 parmoo_cfr_blackbox_solver.py --iseed $SEED
	python3 parmoo_cfr_structured_solver.py --iseed $SEED
done
