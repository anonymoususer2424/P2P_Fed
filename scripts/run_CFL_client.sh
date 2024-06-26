#!/bin/bash
############################################
##               ENV CONFIG               ##
############################################

config=./scripts/config.json
hostslist=($(jq -r '.ADDR.HOSTLIST[]' $config))
using_host=($(jq -r '.ADDR.NODENUM[]' $config))

serveraddr=$(jq -r '.ADDR.SERVER' $config)

root_dir=$(jq -r '.DIR.ROOT_DIR' $config)
proj_dir=$(jq -r '.DIR.PROJ_DIR' $config)
data_dir=$(jq -r '.DIR.DATA_DIR' $config)
echo $proj_dir

############################################

while getopts "MCER" opt; do
	case $opt in
		C)conf="-C"	;;
		E)conf="-E" ;;
		R)conf="-R"	;;
		*) echo "$opt is not the option"
		exit
		;;
	esac
done

i=0
index=0
total=0
for nn in "${using_host[@]}"
do
total=$(( total+nn )) 
done
echo "total | $total"
for hnum in "${using_host[@]}"
do
if [[ $hnum -gt 0 ]]; then
	hosti=${hostslist[$(( i ))]}
	echo "host$i | start with $hnum containers"
	ssh -p 6304 $hosti "hostname"
	ssh -p 6304 $hosti "cd $proj_dir&& bash scripts/CFL_client.sh -S $total -N $hnum -I $index $conf $serveraddr:16000 &"&
	if [ "$conf" = "-C" ] || [ "$conf" = "-R" ]; then
	pids+=($!)
	fi
	if [ "$conf" = "-E" ]; then
		sleep 10 # for reduce server's load
	fi
	index=$(( index+hnum ))
fi
i=$(( i+1 ))
done

if [ "$conf" = "-C" ] || [ "$conf" = "-G" ] || [ "$conf" = "-R" ]; then
for pid in "${pids[@]}"; do
	wait "$pid"
done
fi

echo "All operations completed. Exiting."