#!/bin/bash
#
#
#PBS -l nodes=1:ppn=16,walltime=12:00:00



# Define working directory  Zappa Hop Alternating 1
export WORK_DIR=$HOME/manifold/zhm200/

# Define executable
export EXE="python script.py"

# Add R module
module add languages/python-anaconda3.8.5-2020.11

# Change into working directory
cd $WORK_DIR

# Do some stuff
echo JOB ID: $PBS_JOBID
echo Working Directory: `pwd`
echo Start Time: `date`

# Execute code
$EXE

echo End Time: `date`



sleep 20
