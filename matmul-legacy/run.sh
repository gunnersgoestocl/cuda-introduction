#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -W group_list=gc64
#PBS -j oe

module purge
module load cuda

cd ${PBS_O_WORKDIR}
# ./matmul_cpu 64
# ./matmul_gpu 64
./matmul_compare 256