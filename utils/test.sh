# Начальный порт
START_PORT=5005

# Количество портов для проверки
CHECK_COUNT=1000

for i in $(seq 0 $CHECK_COUNT); do
    PORT=$((START_PORT + i))
    
    # Проверка, занят ли порт
    if ! netstat -tuln | grep -q ":$PORT "; then
        echo "start on port: $PORT"
        exit 0
    fi
done

echo "Все порты заняты от $START_PORT до $(($START_PORT + CHECK_COUNT))"
exit 1