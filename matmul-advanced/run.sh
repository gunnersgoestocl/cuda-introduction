#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -W group_list=gc64
#PBS -j oe

module purge
module load cuda

MATRIX_SIZE=2048
NUM_GPUS=4

cd ${PBS_O_WORKDIR}
./matmul_multidevice $(MATRIX_SIZE) $(NUM_GPUS)