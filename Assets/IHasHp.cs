/// <summary>
/// Персонаж с HP. Используется для урона от зомби и отображения полоски HP.
/// </summary>
public interface IHasHp
{
    int Hp { get; }
    int MaxHp { get; }
    void TakeDamage(int amount);
}
