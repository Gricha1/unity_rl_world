cd ..


# $1 - environment folder
# $2 - run name(if want to resume)
# $3 - config name

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
if [ -z "$2" ]; then
    exit 1
else
    exp_foder=$2
    echo "inference $exp_foder"
    mlagents-learn custom_configs/$config_name.yaml --inference --resume --env=$build_folder --run-id $exp_foder --base-port $PORT
fi




