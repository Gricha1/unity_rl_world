using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Спавнит по 1 зомби каждые N секунд в фиксированной точке.
/// У префаба зомби должен быть слой Zombie и скрипты ZombieChase, ZombieHealth.
/// </summary>
public class ZombieSpawner : MonoBehaviour
{
    [Header("Prefab")]
    [SerializeField] private GameObject zombiePrefab;

    [Header("Spawn Settings")]
    [Tooltip("Один зомби появляется каждые столько секунд")]
    [SerializeField] private float spawnInterval = 5f;
    [Tooltip("Максимум зомби на сцене")]
    [SerializeField] private int maxZombies = 15;
    [Tooltip("Точка появления зомби")]
    [SerializeField] private Vector3 spawnPosition = new Vector3(3.25f, -0.03f, -5.93f);

    private List<GameObject> zombies = new List<GameObject>();
    private float nextRespawnTime;

    private void Start()
    {
        nextRespawnTime = Time.time + spawnInterval;
    }

    private void Update()
    {
        if (zombiePrefab == null) return;
        if (Time.time < nextRespawnTime) return;
        nextRespawnTime = Time.time + spawnInterval;

        RemoveDestroyed();
        if (zombies.Count < maxZombies)
            SpawnOne();
    }

    private void RemoveDestroyed()
    {
        zombies.RemoveAll(z => z == null);
    }

    private void SpawnOne()
    {
        GameObject zombie = Instantiate(zombiePrefab, spawnPosition, Quaternion.Euler(0f, Random.Range(0f, 360f), 0f), transform);
        SetZombieLayer(zombie);
        zombies.Add(zombie);
    }

    private void SetZombieLayer(GameObject zombie)
    {
        int layer = LayerMask.NameToLayer("Zombie");
        if (layer >= 0)
            SetLayerRecursively(zombie, layer);
    }

    private void SetLayerRecursively(GameObject go, int layer)
    {
        go.layer = layer;
        foreach (Transform child in go.transform)
            SetLayerRecursively(child.gameObject, layer);
    }

    /// <summary>Очистить всех зомби (например при новом эпизоде).</summary>
    public void ClearZombies()
    {
        foreach (var z in zombies)
        {
            if (z != null)
                Destroy(z);
        }
        zombies.Clear();
    }
}
