using UnityEngine;

public class PlayerMovement : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    public float moveSpeed = 3f;      // скорость перемещения
    public float rotationSpeed = 120f; // скорость поворота через A/D

    private CharacterController controller;
    private Animator animator;

    void Start()
    {
        controller = GetComponent<CharacterController>();
        animator = GetComponent<Animator>();
    }

    void Update()
    {
        // Получаем input
        float horizontal = Input.GetAxis("Horizontal"); // A/D
        float vertical = Input.GetAxis("Vertical");     // W/S

        // --- Поворот на месте через A/D ---
        if (Mathf.Abs(horizontal) > 0.01f)
        {
            transform.Rotate(0f, horizontal * rotationSpeed * Time.deltaTime, 0f);
        }

        // --- Движение вперед/назад через W/S ---
        Vector3 forwardMove = transform.forward * vertical;

        if (forwardMove.magnitude > 0.01f)
        {
            controller.Move(forwardMove * moveSpeed * Time.deltaTime);
        }

        // --- Анимация ---
        animator.SetFloat("Speed", Mathf.Abs(vertical));
    }
        
}
