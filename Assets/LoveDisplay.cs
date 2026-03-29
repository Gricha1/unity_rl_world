using UnityEngine;
using TMPro;

public class LoveDisplay : MonoBehaviour
{
    [SerializeField] private LilyScript lily;
    [SerializeField] private TMP_SpriteAsset spriteAsset;
    private TMP_Text text;

    void Awake()
    {
        text = GetComponent<TMP_Text>();
        if (text != null)
        {
            text.richText = true;
        }
    }

    void Update()
    {
        if (lily == null || text == null) return;

        if (spriteAsset != null && text.spriteAsset != spriteAsset)
            text.spriteAsset = spriteAsset;

        text.text = $"<sprite=0> {lily.Love}";
    }
}
