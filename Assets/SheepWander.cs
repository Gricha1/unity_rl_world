using UnityEngine;

public class SheepWander : MonoBehaviour
{
    [Header("Movement")]
    [SerializeField] private float moveSpeed = 1.0f;
    [SerializeField] private float rotationSpeed = 180f;

    [Header("Wander")]
    [SerializeField] private float changeDirInterval = 3f;

    [Header("Return to Spawn")]
    [SerializeField] private float maxDistanceFromSpawn = 12f; // если дальше — идём обратно в зону

    [Header("Obstacle Avoidance")]
    [SerializeField] private float obstacleCheckDistance = 0.8f;
    [SerializeField] private LayerMask obstacleLayers;

    [Header("Gravity")]
    [SerializeField] private float gravity = -9.81f;
    [SerializeField] private float groundedForce = -2f;

    private float verticalVelocity;
    private float timer;
    private Vector3 moveDir;
    private CharacterController controller;

    // Зона спавна: задаётся из SheepSpawner при создании овцы
    private bool hasSpawnArea;
    private Vector3 spawnCenterXZ;

    /// <summary>Вызывается из SheepSpawner при спавне — овца будет возвращаться в эту зону, если уйдёт дальше maxDistance.</summary>
    public void SetSpawnArea(Vector3 center, float maxDistance)
    {
        spawnCenterXZ = new Vector3(center.x, 0f, center.z);
        maxDistanceFromSpawn = maxDistance;
        hasSpawnArea = true;
    }

    private void Start()
    {
        controller = GetComponent<CharacterController>();
        PickRandomDirection();
    }

    private void Update()
    {
        timer += Time.deltaTime;

        Vector3 posXZ = new Vector3(transform.position.x, 0f, transform.position.z);

        // Возврат в зону спавна, если ушли слишком далеко
        if (hasSpawnArea && Vector3.Distance(posXZ, spawnCenterXZ) > maxDistanceFromSpawn)
        {
            moveDir = (spawnCenterXZ - posXZ).normalized;
            timer = 0f;
        }
        else
        {
            if (Physics.Raycast(transform.position + Vector3.up * 0.2f,
                         transform.forward,
                         obstacleCheckDistance,
                         obstacleLayers))
            {
                PickRandomDirection();
                timer = 0f;
            }
            else if (timer >= changeDirInterval)
            {
                PickRandomDirection();
                timer = 0f;
            }
        }

        // Плавный поворот
        Quaternion targetRot = Quaternion.LookRotation(moveDir);
        transform.rotation = Quaternion.RotateTowards(
            transform.rotation,
            targetRot,
            rotationSpeed * Time.deltaTime
        );

        // Движение
        Vector3 move = transform.forward * moveSpeed;

        if (controller.isGrounded)
        {
            if (verticalVelocity < 0f)
                verticalVelocity = groundedForce;
        }
        else
        {
            verticalVelocity += gravity * Time.deltaTime;
        }

        move.y = verticalVelocity;

        controller.Move(move * Time.deltaTime);

    }

    private void PickRandomDirection()
    {
        float angle = Random.Range(0f, 360f);
        moveDir = new Vector3(
            Mathf.Sin(angle * Mathf.Deg2Rad),
            0f,
            Mathf.Cos(angle * Mathf.Deg2Rad)
        ).normalized;
    }

#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.red;
        Gizmos.DrawLine(
            transform.position + Vector3.up * 0.2f,
            transform.position + Vector3.up * 0.2f + transform.forward * obstacleCheckDistance
        );
    }
#endif
}
