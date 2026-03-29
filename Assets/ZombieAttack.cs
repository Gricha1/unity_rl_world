using UnityEngine;

/// <summary>
/// При контакте зомби с персонажем (Jack/Lily) снимает 1/5 его макс. HP. Кулдаун между ударами.
/// Вешать на того же зомби, что и ZombieChase. Нужен CharacterController или коллайдер для контакта.
/// </summary>
[RequireComponent(typeof(Collider))]
public class ZombieAttack : MonoBehaviour
{
    [SerializeField] private float hitCooldown = 1f;
    private float lastHitTime = -999f;

    private void OnControllerColliderHit(ControllerColliderHit hit)
    {
        TryDamage(hit.gameObject);
    }

    private void OnCollisionEnter(Collision collision)
    {
        TryDamage(collision.gameObject);
    }

    private void TryDamage(GameObject other)
    {
        var target = other.GetComponentInParent<IHasHp>();
        if (target == null) return;
        if (Time.time - lastHitTime < hitCooldown) return;

        int damage = Mathf.Max(1, target.MaxHp / 5);
        target.TakeDamage(damage);
        lastHitTime = Time.time;
    }
}
