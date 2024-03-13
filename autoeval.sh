#!/bin/bash

# Author: jwher
# Before running this script, ensure you are authorized to access the remote server.
# Also, make sure to select the correct branch of your evaluation codes.
# This script will check for the existence of remote files and perform evaluations.

# set start number
n=3
end=20
domain="hinton3"
rpath="/home/jwher96/EfficientCenterFormer/second/"
hpath="./"
config="configs/nusc/nuscenes_second.py"
cmd="python tools/dist_test.py --config ${config} --checkpoint="

# Initialize start time
start_time=$(date +%s)
echo "[${start_time}] Start"

# Start loop
while :
do
    # Check remote file existence
    ssh ${domain} "ls ${rpath}epoch_${n}.pth" > /dev/null 2>&1
    
    # Check if file exists
    if [ $? -ne 0 ]; then

	current_time=$(date +%s)
	if [ $((current_time - start_time)) -ge 1800 ]; then
            echo "[${current_time}] Waiting for training finish"
	fi

	sleep 60
        continue
    fi

     # File exists, so copy it
     scp ${domain}:${rpath}epoch_${n}.pth ${hpath}
     
     # Evaluate the file or perform desired command
     eval ${cmd}${hpath}epoch_${n}.pth

     start_time=$(date +%s)
     
     # Increase n
     ((n++))
     if [ $n -gt $end ]; then break; fi
done

