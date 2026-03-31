using UnityEngine;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// Runtime HUD: HP bars for Jack and Lily in top-left corner.
/// Created automatically at runtime (no scene wiring required).
/// </summary>
public sealed class HudHpBars : MonoBehaviour
{
    [Header("Layout")]
    [SerializeField] private Vector2 padding = new Vector2(16f, 16f);
    [SerializeField] private float topOffset = 36f;
    [SerializeField] private float barWidth = 140f;
    [SerializeField] private float barHeight = 18f;
    [SerializeField] private float rowSpacing = 10f;

    [Header("Colors")]
    [SerializeField] private Color backgroundColor = new Color(0f, 0f, 0f, 0.55f);
    [SerializeField] private Color jackFill = new Color(0.2f, 0.75f, 1.0f, 0.95f);
    [SerializeField] private Color lilyFill = new Color(1.0f, 0.35f, 0.75f, 0.95f);

    private IHasHp _jack;
    private IHasHp _lily;

    private Image _jackFillImg;
    private RectTransform _jackFillRt;
    private TextMeshProUGUI _jackText;
    private Image _lilyFillImg;
    private RectTransform _lilyFillRt;
    private TextMeshProUGUI _lilyText;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Bootstrap()
    {
        if (FindObjectOfType<HudHpBars>() != null) return;
        var go = new GameObject(nameof(HudHpBars));
        DontDestroyOnLoad(go);
        go.AddComponent<HudHpBars>();
    }

    private void Awake()
    {
        CreateCanvasAndBars();
    }

    private void Update()
    {
        // Resolve references lazily (agents can spawn after scene load).
        if (_jack == null)
            _jack = FindObjectOfType<AgentGoToHouseDiscrete>();
        if (_lily == null)
            _lily = FindObjectOfType<LilyScript>();

        UpdateBar(_jack, _jackFillRt, _jackText, "Jack");
        UpdateBar(_lily, _lilyFillRt, _lilyText, "Lily");
    }

    private void UpdateBar(IHasHp hp, RectTransform fillRt, TextMeshProUGUI label, string name)
    {
        if (fillRt == null || label == null) return;

        if (hp == null || hp.MaxHp <= 0)
        {
            fillRt.sizeDelta = new Vector2(0f, fillRt.sizeDelta.y);
            label.text = $"{name}: --/--";
            return;
        }

        float t = Mathf.Clamp01((float)hp.Hp / hp.MaxHp);
        fillRt.sizeDelta = new Vector2(barWidth * t, fillRt.sizeDelta.y);
        label.text = $"{name}: {hp.Hp}/{hp.MaxHp}";
    }

    private void CreateCanvasAndBars()
    {
        var canvasGo = new GameObject("HudCanvas");
        canvasGo.transform.SetParent(transform, false);

        var canvas = canvasGo.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 1000;

        canvasGo.AddComponent<CanvasScaler>().uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        canvasGo.AddComponent<GraphicRaycaster>();

        var root = new GameObject("TopLeftRoot").AddComponent<RectTransform>();
        root.SetParent(canvasGo.transform, false);
        root.anchorMin = new Vector2(0f, 1f);
        root.anchorMax = new Vector2(0f, 1f);
        root.pivot = new Vector2(0f, 1f);
        root.anchoredPosition = new Vector2(padding.x, -(padding.y + topOffset));
        root.sizeDelta = new Vector2(barWidth + 110f, (barHeight + rowSpacing) * 2f + 10f);

        CreateRow(root, 0, "Jack", jackFill, out _jackFillImg, out _jackFillRt, out _jackText);
        CreateRow(root, 1, "Lily", lilyFill, out _lilyFillImg, out _lilyFillRt, out _lilyText);
    }

    private void CreateRow(RectTransform parent, int rowIndex, string title, Color fillColor,
        out Image fillImg, out RectTransform fillRt, out TextMeshProUGUI text)
    {
        float y = -rowIndex * (barHeight + rowSpacing);

        var row = new GameObject($"{title}Row").AddComponent<RectTransform>();
        row.SetParent(parent, false);
        row.anchorMin = new Vector2(0f, 1f);
        row.anchorMax = new Vector2(0f, 1f);
        row.pivot = new Vector2(0f, 1f);
        row.anchoredPosition = new Vector2(0f, y);
        row.sizeDelta = new Vector2(parent.sizeDelta.x, barHeight);

        // Background
        var bg = new GameObject($"{title}Bg").AddComponent<RectTransform>();
        bg.SetParent(row, false);
        bg.anchorMin = new Vector2(0f, 0f);
        bg.anchorMax = new Vector2(0f, 1f);
        bg.pivot = new Vector2(0f, 0.5f);
        bg.anchoredPosition = Vector2.zero;
        bg.sizeDelta = new Vector2(barWidth, 0f);

        var bgImg = bg.gameObject.AddComponent<Image>();
        bgImg.color = backgroundColor;

        // Fill
        var fill = new GameObject($"{title}Fill").AddComponent<RectTransform>();
        fill.SetParent(bg, false);
        fill.anchorMin = new Vector2(0f, 0f);
        fill.anchorMax = new Vector2(1f, 1f);
        fill.pivot = new Vector2(0f, 0.5f);
        fill.anchoredPosition = Vector2.zero;
        fill.sizeDelta = Vector2.zero;

        fillRt = fill;
        fillImg = fill.gameObject.AddComponent<Image>();
        fillImg.type = Image.Type.Simple;
        fillImg.color = fillColor;
        // We'll control width manually for a "shrinking bar" look.
        fill.anchorMax = new Vector2(0f, 1f);
        fill.sizeDelta = new Vector2(barWidth, 0f);

        // Text
        var labelRt = new GameObject($"{title}Text").AddComponent<RectTransform>();
        labelRt.SetParent(row, false);
        labelRt.anchorMin = new Vector2(0f, 0f);
        labelRt.anchorMax = new Vector2(1f, 1f);
        labelRt.pivot = new Vector2(0f, 0.5f);
        labelRt.anchoredPosition = new Vector2(barWidth + 8f, 0f);
        labelRt.sizeDelta = new Vector2(180f, 0f);

        text = labelRt.gameObject.AddComponent<TextMeshProUGUI>();
        text.fontSize = 16f;
        text.color = Color.white;
        text.alignment = TextAlignmentOptions.Left;
        text.text = $"{title}: --/--";
        text.enableWordWrapping = false;
    }
}

