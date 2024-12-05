using UnityEngine;

public class DynamicObstBehaviour : MonoBehaviour
{
    private Vector3 start_pose;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        start_pose = transform.localPosition;
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private void OnTriggerEnter(Collider other) {
        if (other.TryGetComponent(out Wall Wall)) {
            transform.localPosition = start_pose;
        }
    }
}
