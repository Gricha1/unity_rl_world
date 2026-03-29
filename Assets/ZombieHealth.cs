using UnityEngine;

/// <summary>
/// Здоровье зомби. После 2 попаданий (2 выстрела) зомби умирает. Реализует IHasHp для полоски HP.
/// </summary>
public class ZombieHealth : MonoBehaviour, IHasHp
{
    [SerializeField] private int maxHp = 2;
    private int hp;

    public int Hp => hp;
    public int MaxHp => maxHp;

    private void OnEnable()
    {
        hp = maxHp;
    }

    private void Start()
    {
        if (GetComponent<HpBarVisual>() == null)
            gameObject.AddComponent<HpBarVisual>();
    }

    /// <summary>Наносит урон. При HP &lt;= 0 зомби уничтожается.</summary>
    public void TakeDamage(int amount)
    {
        hp -= amount;
        if (hp <= 0)
            Destroy(gameObject);
    }
}
