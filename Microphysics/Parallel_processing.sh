#!/bin/bash
#
# This script loops over idealized MCS cases
#
###########################################
#                        imput section
###########################################

iParallelNr=20

# squeue -u $USER

# loop over simulations
for si in $(seq 1 $iParallelNr)
do

    # prepare the parallel python script
    PythonName='TempPrograms/'$si'_MicrophysicsProperties.py'
    sed "s#>>SIM<<#$si#g" 'Microphysics.py' > $PythonName
    # sed -i "s#>>TC_STOP<<#$iTCstop#g" $PythonName
    chmod 744 $PythonName
/bin/cat <<EOM >'TempPrograms/'$si'_Start.sh'
#!/bin/bash -l
#SBATCH -J $si
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH -t 6:00:00
#SBATCH -A P66770001
#SBATCH -p dav

module load python/2.7.14
ncar_pylib
srun ./$PythonName
EOM

    chmod 744 'TempPrograms/'$si'_Start.sh'
    sbatch 'TempPrograms/'$si'_Start.sh'
    # bsub -n 1 -R "span[ptile=1]" -q geyser -W 24:00 -PP66770001 \
    #     ./$PythonName
    sleep 1

done

exit

