using UnityEngine;

/// <summary>
/// Рисует полоску HP над персонажем. Вешать на того же объект, что и Jack/Lily/Zombie (IHasHp).
/// Полоска всегда повёрнута к камере (биллборд).
/// </summary>
public class HpBarVisual : MonoBehaviour
{
    [Tooltip("Высота полоски над персонажем. Подбирается под надписи/иконки над головой.")]
    [SerializeField] private float heightAbove = 2.9f;
    [SerializeField] private float barWidth = 2f;
    [SerializeField] private float barHeight = 0.25f;
    [SerializeField] private Color fillColor = new Color(0.2f, 0.8f, 0.2f);
    [SerializeField] private Color backgroundColor = new Color(0.25f, 0.1f, 0.1f);

    private IHasHp source;
    private Transform barPivot;
    private Transform fillTransform;
    private Renderer bgRenderer;
    private Renderer fillRenderer;
    private float currentT = 1f;
    private float tVelocity;

    [Header("Smoothing")]
    [Tooltip("Сколько секунд занимает сглаживание изменения HP (0 = без сглаживания).")]
    [SerializeField] private float smoothTimeSeconds = 0.12f;

    [Header("Rendering")]
    [Tooltip("Sorting order for background (fill uses +1). Helps avoid z-fighting / color smearing.")]
    [SerializeField] private int sortingOrder = 200;
    [Tooltip("Насколько заливка ближе к камере относительно фона (в локальном Z).")]
    [SerializeField] private float fillZOffset = -0.01f;

    private void Start()
    {
        source = GetComponent<IHasHp>();
        if (source == null)
            source = GetComponentInParent<IHasHp>();
        if (source == null)
        {
            Debug.LogWarning("HpBarVisual: на объекте или родителе не найден IHasHp.");
            return;
        }

        barPivot = new GameObject("HpBar").transform;
        barPivot.SetParent(transform);
        barPivot.localPosition = new Vector3(0f, heightAbove, 0f);
        barPivot.localRotation = Quaternion.identity;
        barPivot.localScale = Vector3.one;

        // Фон — Quad, всегда повернут к камере (billboard).
        GameObject bgGo = GameObject.CreatePrimitive(PrimitiveType.Quad);
        bgGo.name = "HpBarBg";
        bgGo.transform.SetParent(barPivot);
        bgGo.transform.localPosition = Vector3.zero;
        bgGo.transform.localRotation = Quaternion.identity;
        bgGo.transform.localScale = new Vector3(barWidth, barHeight, 1f);
        ApplyColor(bgGo, backgroundColor);
        Destroy(bgGo.GetComponent<Collider>());
        bgRenderer = bgGo.GetComponent<Renderer>();
        if (bgRenderer != null)
        {
            bgRenderer.sortingOrder = sortingOrder;
            bgRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            bgRenderer.receiveShadows = false;
        }

        // Заливка — масштаб по hp/maxHp, выравнивание по левому краю
        GameObject fillGo = GameObject.CreatePrimitive(PrimitiveType.Quad);
        fillGo.name = "HpBarFill";
        fillTransform = fillGo.transform;
        fillTransform.SetParent(barPivot);
        fillTransform.localRotation = Quaternion.identity;
        ApplyColor(fillGo, fillColor);
        Destroy(fillGo.GetComponent<Collider>());
        fillRenderer = fillGo.GetComponent<Renderer>();
        if (fillRenderer != null)
        {
            fillRenderer.sortingOrder = sortingOrder + 1;
            fillRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            fillRenderer.receiveShadows = false;
        }

        UpdateBar();
    }

    private static void ApplyColor(GameObject go, Color color)
    {
        var r = go.GetComponent<Renderer>();
        if (r == null) return;
        var shader = Shader.Find("Unlit/Color") ?? Shader.Find("Legacy Shaders/Unlit/Color");
        if (shader != null)
            r.material = new Material(shader);
        r.material.color = color;
    }

    private void LateUpdate()
    {
        if (source == null || fillTransform == null) return;
        // Биллборд: полоска всегда повёрнута к камере.
        if (Camera.main != null)
        {
            // Копируем ротацию камеры: самый стабильный и читаемый вариант для UI-полосок.
            barPivot.rotation = Camera.main.transform.rotation;
        }
        UpdateBar();
    }

    private void UpdateBar()
    {
        int max = source.MaxHp;
        if (max <= 0) return;
        float targetT = Mathf.Clamp01((float)source.Hp / max);
        if (smoothTimeSeconds > 0f)
            currentT = Mathf.SmoothDamp(currentT, targetT, ref tVelocity, smoothTimeSeconds);
        else
            currentT = targetT;

        float w = barWidth * currentT;
        fillTransform.localScale = new Vector3(w, barHeight, 1f);
        fillTransform.localPosition = new Vector3(-(barWidth - w) * 0.5f, 0f, fillZOffset);
    }
}
