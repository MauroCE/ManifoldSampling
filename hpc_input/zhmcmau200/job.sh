#!/bin/bash
#
#
#PBS -l nodes=1:ppn=4,walltime=10:00:00



# Define working directory
export WORK_DIR=$HOME/manifold/zhmcmau200/

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
