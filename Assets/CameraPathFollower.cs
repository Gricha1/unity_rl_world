using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Двигает камеру по точкам (children у CameraPath) в порядке в Hierarchy.
/// Позиция и поворот интерполируются к Transform каждой точки.
/// </summary>
public class CameraPathFollower : MonoBehaviour
{
    [Header("Path")]
    [Tooltip("Объект CameraPath, внутри которого лежат точки (first_point, second_point, ...).")]
    [SerializeField] private Transform pathRoot;

    [Tooltip("Если true — точки берутся из детей pathRoot автоматически (в порядке в Hierarchy).")]
    [SerializeField] private bool autoCollectChildren = true;

    [Tooltip("Явный список точек (если autoCollectChildren = false).")]
    [SerializeField] private List<Transform> points = new List<Transform>();

    [Header("Motion")]
    [Tooltip("Скорость движения между точками (м/с).")]
    [SerializeField] private float moveSpeed = 3.0f;

    [Tooltip("Скорость поворота (чем больше, тем быстрее догоняет ротацию точки).")]
    [SerializeField] private float rotationLerpSpeed = 4.0f;

    [Tooltip("Насколько близко к точке нужно подойти, чтобы считать её достигнутой.")]
    [SerializeField] private float arriveDistance = 0.05f;

    [Tooltip("Пауза на точке (сек).")]
    [SerializeField] private float waitAtPointSeconds = 0.0f;

    [Header("Playback")]
    [SerializeField] private bool playOnStart = true;
    [SerializeField] private bool loop = true;

    private int _index;
    private float _waitLeft;
    private bool _playing;

    private void Start()
    {
        RefreshPoints();
        if (points.Count > 0 && playOnStart)
        {
            // Стартуем ровно в первой точке, чтобы Recorder начинал “как стоит вначале”.
            transform.SetPositionAndRotation(points[0].position, points[0].rotation);
            _index = 1 % points.Count;
            _playing = true;
        }
    }

    [ContextMenu("Refresh Points From PathRoot")]
    public void RefreshPoints()
    {
        if (!autoCollectChildren) return;
        points.Clear();
        if (pathRoot == null) return;
        for (int i = 0; i < pathRoot.childCount; i++)
        {
            var t = pathRoot.GetChild(i);
            if (t != null)
                points.Add(t);
        }
    }

    private void Update()
    {
        if (!_playing) return;
        if (points == null || points.Count == 0) return;

        if (_waitLeft > 0f)
        {
            _waitLeft -= Time.deltaTime;
            return;
        }

        var target = points[Mathf.Clamp(_index, 0, points.Count - 1)];
        if (target == null)
        {
            Advance();
            return;
        }

        // Move
        Vector3 to = target.position - transform.position;
        float dist = to.magnitude;
        if (dist <= arriveDistance)
        {
            transform.position = target.position;
            transform.rotation = target.rotation;
            _waitLeft = Mathf.Max(0f, waitAtPointSeconds);
            Advance();
            return;
        }

        Vector3 step = to / Mathf.Max(0.0001f, dist) * (moveSpeed * Time.deltaTime);
        if (step.magnitude > dist) step = to;
        transform.position += step;

        // Rotate (exponential-ish smoothing)
        float rt = 1f - Mathf.Exp(-rotationLerpSpeed * Time.deltaTime);
        transform.rotation = Quaternion.Slerp(transform.rotation, target.rotation, rt);
    }

    private void Advance()
    {
        _index++;
        if (_index >= points.Count)
        {
            if (loop)
                _index = 0;
            else
                _playing = false;
        }
    }
}

