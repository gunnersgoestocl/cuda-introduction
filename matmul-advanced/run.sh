#!/bin/bash
#PBS -q debug-g
#PBS -l select=16
#PBS -W group_list=gc64
#PBS -j oe

MATRIX_SIZE_POWER=15
NUM_GPUS=4
NUM_STREAMS=4
NUM_NODES=16

module purge
module load nvidia/24.9
module load cuda/12.6
module load ompi-cuda

cd ${PBS_O_WORKDIR}

# ./matmul_multidevice $MATRIX_SIZE_POWER $NUM_GPUS
# ./matmul_multistream $MATRIX_SIZE_POWER $NUM_STREAMS
mpirun -n ${NUM_NODES} ./matmul_multinode ${MATRIX_SIZE_POWER}