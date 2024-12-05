using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Camera : MonoBehaviour
{
    public GameObject player;  // Публичная переменная для хранения ссылки на игровой объект игрока
    private Vector3 offset;  // Частная переменная для хранения смещения расстояния между игроком и камерой

    void Start()
    {
        // Вычисляем и сохраняем значение смещения, получая расстояние между позицией игрока и камеры
        if (player != null) {
            offset = transform.position - player.transform.position;
        }
    }

    void LateUpdate()
    {
        // Устанавливаем положение преобразования камеры таким же, как у игрока, но со смещением на вычисленное расстояние смещения
        if (player != null) {
            transform.position = player.transform.position + offset;
        }
    }
}
