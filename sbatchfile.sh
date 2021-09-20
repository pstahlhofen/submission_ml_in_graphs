#!/bin/bash

IFS=$'\n'
paroutlog=($(cat params_and_outs.txt))
sbatch run_hyperparameter_search.sh ${paroutlog[@]}
