using UnityEngine;

/// <summary>
/// Зомби идёт к тому, кто ближе — Джек или Лилия. Скорость ниже, чем у них (Jack/Lily = 3, по умолчанию зомби = 1.5).
/// Нужен CharacterController или Rigidbody на объекте.
/// </summary>
public class ZombieChase : MonoBehaviour
{
    [Header("Targets")]
    [SerializeField] private Transform jackTarget;
    [SerializeField] private Transform lilyTarget;

    [Header("Movement")]
    [SerializeField] private float moveSpeed = 0.5f;
    [SerializeField] private float rotationSpeed = 120f;

    [Header("Gravity")]
    [SerializeField] private float gravity = -9.81f;

    private CharacterController controller;
    private Rigidbody rb;
    private Animator animator;
    private float verticalVelocity;
    private bool useRigidbody;

    [Header("Stun / Freeze")]
    [Tooltip("Время, до которого зомби не может двигаться (устанавливается через Stun).")]
    private float stunnedUntilTime = -999f;

    // Счётчик попаданий DO от Джека (нужен, чтобы гарантировать смерть за 2 удара даже без ZombieHealth).
    private int doHitsFromJack;

    public bool IsStunned => Time.time < stunnedUntilTime;

    public void Stun(float seconds)
    {
        if (seconds <= 0f) return;
        stunnedUntilTime = Mathf.Max(stunnedUntilTime, Time.time + seconds);
    }

    public void RegisterJackDoHitAndMaybeDie(int hitsToDie = 2)
    {
        doHitsFromJack++;
        if (hitsToDie > 0 && doHitsFromJack >= hitsToDie)
        {
            Destroy(gameObject);
        }
    }

    private void OnEnable()
    {
        doHitsFromJack = 0;
    }

    private void Start()
    {
        controller = GetComponent<CharacterController>();
        rb = GetComponent<Rigidbody>();
        animator = GetComponent<Animator>();
        useRigidbody = (controller == null && rb != null);

        if (jackTarget == null)
        {
            var jack = FindObjectOfType<AgentGoToHouseDiscrete>();
            if (jack != null) jackTarget = jack.transform;
        }
        if (lilyTarget == null)
        {
            var lily = FindObjectOfType<LilyScript>();
            if (lily != null) lilyTarget = lily.transform;
        }
    }

    private void Update()
    {
        if (Time.time < stunnedUntilTime)
        {
            if (animator != null)
                animator.SetFloat("Speed", 0f);
            return;
        }

        Transform target = GetClosestTarget();
        if (target == null)
        {
            if (animator != null)
                animator.SetFloat("Speed", 0f);
            return;
        }

        Vector3 pos = transform.position;
        Vector3 targetPos = target.position;
        Vector3 delta = targetPos - pos;
        delta.y = 0f;

        if (delta.sqrMagnitude < 0.001f)
        {
            if (animator != null)
                animator.SetFloat("Speed", 0f);
            return;
        }

        Vector3 dir = delta.normalized;

        // Анимация ходьбы (тот же параметр "Speed", что у Jack и Lily)
        if (animator != null)
            animator.SetFloat("Speed", 1f);

        // Поворот в сторону цели
        Quaternion targetRot = Quaternion.LookRotation(dir);
        transform.rotation = Quaternion.RotateTowards(
            transform.rotation,
            targetRot,
            rotationSpeed * Time.deltaTime
        );

        if (controller != null)
        {
            if (controller.isGrounded && verticalVelocity < 0f)
                verticalVelocity = -2f;
            else
                verticalVelocity += gravity * Time.deltaTime;
            Vector3 move = dir * moveSpeed + Vector3.up * verticalVelocity;
            controller.Move(move * Time.deltaTime);
        }
    }

    private void FixedUpdate()
    {
        if (rb == null || !useRigidbody) return;
        if (Time.time < stunnedUntilTime)
        {
            // Останавливаем Rigidbody зомби на время стана
            rb.linearVelocity = Vector3.zero;
            return;
        }
        Transform target = GetClosestTarget();
        if (target == null) return;

        Vector3 delta = target.position - transform.position;
        delta.y = 0f;
        if (delta.sqrMagnitude < 0.001f) return;
        Vector3 dir = delta.normalized;

        Vector3 vel = dir * moveSpeed;
        vel.y = rb.linearVelocity.y + gravity * Time.fixedDeltaTime;
        if (vel.y < -20f) vel.y = -20f;
        rb.linearVelocity = vel;
    }

    /// <summary>Ближайшая цель по горизонтали (XZ).</summary>
    private Transform GetClosestTarget()
    {
        Vector3 pos = transform.position;
        pos.y = 0f;

        float distJack = float.MaxValue;
        if (jackTarget != null && jackTarget.gameObject.activeInHierarchy)
        {
            Vector3 j = jackTarget.position;
            j.y = 0f;
            distJack = Vector3.Distance(pos, j);
        }

        float distLily = float.MaxValue;
        if (lilyTarget != null && lilyTarget.gameObject.activeInHierarchy)
        {
            Vector3 l = lilyTarget.position;
            l.y = 0f;
            distLily = Vector3.Distance(pos, l);
        }

        if (distJack <= distLily && jackTarget != null)
            return jackTarget;
        if (lilyTarget != null)
            return lilyTarget;
        return jackTarget;
    }
}
