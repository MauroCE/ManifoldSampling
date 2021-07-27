#!/bin/bash
#
#
#PBS -l nodes=1:ppn=16,walltime=3:00:00



# Define working directory
export WORK_DIR=$HOME/hugjob/

# Define executable
export EXE="python script.py"

# Add R module
module add lang/python/anaconda/3.8.8-2021.05-2.5


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