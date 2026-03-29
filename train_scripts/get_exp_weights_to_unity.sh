# Проверка, передан ли аргумент
if [ -z "$1" ]; then
  echo "provide exp name in results/"
  exit 1
fi

if [ -z "$2" ]; then
  steps=""
  echo "load weights: -$steps.onnx"
  echo "exp: $1"
  cd ..
  weights_folder=$1
  cp results/$weights_folder/"My Behavior.onnx" Assets/ML-Agents/weights/$weights_folder.onnx

else
  steps="$2"
  echo "load weights: -$steps.onnx"
  echo "exp: $1"
  cd ..
  weights_folder=$1
  cp results/$weights_folder/"My Behavior"/"My Behavior-$steps.onnx" Assets/ML-Agents/weights/$weights_folder-$steps.onnx
fi


