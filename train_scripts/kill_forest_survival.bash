#!/usr/bin/env bash
# Принудительно гасит обучение/инференс ML-Agents и зависшие Unity-сборки проекта forest_survival (WSL/Linux).
# Запуск из корня проекта или откуда угодно:
#   bash train_scripts/kill_forest_survival.bash

set +e

echo "[kill] mlagents-learn..."
pkill -9 -f mlagents-learn 2>/dev/null

echo "[kill] validate_video_watcher..."
pkill -9 -f validate_video_watcher_fixed.bash 2>/dev/null

# Только сборки из build_versions этого проекта (не трогаем скрипт kill_forest_survival.bash)
echo "[kill] Unity-сборки build_versions этого проекта..."
pkill -9 -f "unity_projects/forest_survival/build_versions" 2>/dev/null
pkill -9 -f "unity_projects\\forest_survival\\build_versions" 2>/dev/null

sleep 0.3

echo ""
echo "[kill] осталось что-то с forest_survival в командной строке:"
ps aux 2>/dev/null | grep -E "forest_survival|mlagents-learn" | grep -v grep || echo "  (ничего)"

echo ""
echo "Готово. Если PID всё ещё в htop — добей вручную: kill -9 <PID>"
