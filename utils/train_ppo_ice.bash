cd ..


# $1 - environment folder
# $2 - env num
# $3 - run name(if want to resume)

# folder with environment builded
if [ -z "$1" ]; then
    echo "set environmet in first arg!!!"
    exit 1
else
    build_folder=$1
    echo "environment name: $build_folder"
fi

# choose port to start training
START_PORT=5005
CHECK_COUNT=1000
for i in $(seq 0 $CHECK_COUNT); do
    PORT=$((START_PORT + i))
    
    if ! netstat -tuln | grep -q ":$PORT "; then
        echo "start on port: $PORT"
        break
    fi
done

# set num environments in parallel
if [ -z "$2" ]; then
    echo "set num envs in second arg!!!"
    exit 1
else
    num_envs=$2
    echo "num envs: $num_envs"
fi

# config name
if [ -z "$3" ]; then
    echo "set config name!!!"
    exit 1
else
    config_name=$3
fi

echo "config: $config_name"

# setup folder for logging
dir_name="results" 
base_name="test"
count=1000000
if [ -z "$4" ]; then
    echo "new training"
    for ((i=1; i<=count; i++)); do
        folder_name="${dir_name}/${base_name}_${i}"
        if [ ! -d "$folder_name" ]; then
            exp_foder="${base_name}_${i}"
            echo "exp folder: $exp_foder"
            break
        fi
    done
    mlagents-learn custom_configs/$config_name.yaml --run-id $exp_foder --env=$build_folder --base-port $PORT --num-envs=$num_envs
    
else
    exp_foder=$4
    echo "resume $exp_foder"
    mlagents-learn custom_configs/$config_name.yaml --run-id $exp_foder --env=$build_folder --resume --base-port $PORT --num-envs=$num_envs
fi




