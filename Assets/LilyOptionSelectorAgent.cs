using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using TMPro;

/// <summary>
/// Селектор опций для Lily: 0 = цветы, 1 = поцелуй Джека, 2 = зомби (стрелять по зомби).
/// Эвристика: клавиша R — переключить опцию (0 → 1 → 2 → 0).
/// </summary>
public class LilyOptionSelectorAgent : Agent
{
    [Header("Low-level Agent")]
    [SerializeField] private LilyScript lilyAgent;

    [Header("UI Display")]
    [SerializeField] private TextMeshPro optionDisplayText;
    [SerializeField] private float displayHeight = 2.5f;

    [Header("Option Selection")]
    private int selectedOption = 0;
    private const int OPTION_CHANGE_INTERVAL = 20;
    private int stepsSinceLastChange = 0;
    private int lastLilyStepCount = 0;
    private int optionChangeCount = 0;
    private bool rKeyPressedLastFrame = false;

    [Header("Reward")]
    [SerializeField] [Tooltip("Доля награды Lily, которую получает селектор при совпадении опции")]
    private float optionRewardScale = 0.5f;
    [SerializeField] [Tooltip("Штраф за смену опции")]
    private float optionChangePenalty = -0.05f;

    public int GetSelectedOption() => selectedOption;

    /// <summary>Lily вызывает при получении награды за задачу — селектор получает награду, если его опция совпадает.</summary>
    public void AddOptionReward(int optionThatGotReward, float amount)
    {
        if (optionThatGotReward == selectedOption)
            AddReward(amount * optionRewardScale);
    }

    public override void Initialize()
    {
        if (lilyAgent == null)
            Debug.LogError("LilyOptionSelectorAgent: lilyAgent не назначен!");
        if (optionDisplayText == null && lilyAgent != null)
            CreateOptionDisplay();
    }

    private void CreateOptionDisplay()
    {
        GameObject textObject = new GameObject("LilyOptionDisplay");
        textObject.transform.SetParent(lilyAgent.transform);
        textObject.transform.localPosition = new Vector3(0, displayHeight, 0);
        textObject.transform.localRotation = Quaternion.identity;
        textObject.transform.localScale = Vector3.one;
        optionDisplayText = textObject.AddComponent<TextMeshPro>();
        optionDisplayText.text = "ЦВЕТЫ";
        optionDisplayText.fontSize = 3;
        optionDisplayText.alignment = TextAlignmentOptions.Center;
        optionDisplayText.color = Color.cyan;
        optionDisplayText.fontStyle = FontStyles.Bold;
        optionDisplayText.sortingOrder = 100;
    }

    public override void OnEpisodeBegin()
    {
        stepsSinceLastChange = 0;
        if (lilyAgent != null)
        {
            lastLilyStepCount = lilyAgent.StepCount;
            selectedOption = lilyAgent.GetCurrentOption();
            if (lilyAgent.CurriculumNoShootNoZombie && selectedOption == 2)
                selectedOption = 0;
            lilyAgent.SetOption(selectedOption);
            UpdateOptionDisplay();
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (lilyAgent == null)
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            return;
        }
        sensor.AddObservation((float)lilyAgent.GetCurrentOption() / 2f); // нормализовано 0..1
        sensor.AddObservation(lilyAgent.GetDistanceToJackNormalized());
        sensor.AddObservation(lilyAgent.GetDistanceToNearestFlowerNormalized());
        sensor.AddObservation(lilyAgent.GetDistanceToNearestZombieNormalized());
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        stepsSinceLastChange++;

        int newOption = actions.DiscreteActions[0];
        if (newOption != 0 && newOption != 1) newOption = selectedOption;
        // Curriculum: опция 2 (зомби) при обучении не используется — подменяем на 0
        if (lilyAgent != null && lilyAgent.CurriculumNoShootNoZombie && newOption == 2)
            newOption = 0;

        if (stepsSinceLastChange >= OPTION_CHANGE_INTERVAL)
        {
            if (newOption != selectedOption)
            {
                selectedOption = newOption;
                stepsSinceLastChange = 0;
                optionChangeCount++;
                AddReward(optionChangePenalty);
                if (lilyAgent != null)
                    lilyAgent.SetOption(selectedOption);
                UpdateOptionDisplay();
            }
            else
                stepsSinceLastChange = 0;
        }

        if (lilyAgent != null)
        {
            int currentLilyStep = lilyAgent.StepCount;
            if (currentLilyStep < lastLilyStepCount && lastLilyStepCount > 0)
                EndEpisode();
            lastLilyStepCount = currentLilyStep;
        }
    }

    private void Update()
    {
        bool rPressed = Input.GetKey(KeyCode.R);
        if (rPressed && !rKeyPressedLastFrame && lilyAgent != null)
        {
            int maxOption = (lilyAgent.CurriculumNoShootNoZombie ? 2 : 3);
            selectedOption = (selectedOption + 1) % maxOption; // curriculum: только 0→1→0
            lilyAgent.SetOption(selectedOption);
            stepsSinceLastChange = 0;
            UpdateOptionDisplay();
        }
        rKeyPressedLastFrame = rPressed;

        if (optionDisplayText != null && lilyAgent != null)
        {
            if (Camera.main != null)
            {
                optionDisplayText.transform.LookAt(Camera.main.transform);
                optionDisplayText.transform.Rotate(0, 180, 0);
            }
            if (optionDisplayText.text != GetOptionDisplayText())
                optionDisplayText.text = GetOptionDisplayText();
        }
    }

    private string GetOptionDisplayText()
    {
        if (selectedOption == 0) return "ЦВЕТЫ";
        if (selectedOption == 1) return "ДЖЕК";
        return "ЗОМБИ";
    }

    private void UpdateOptionDisplay()
    {
        if (optionDisplayText != null)
        {
            optionDisplayText.text = GetOptionDisplayText();
            optionDisplayText.color = selectedOption == 0 ? Color.cyan
                : selectedOption == 1 ? new Color(1f, 0.5f, 0.8f)
                : new Color(0.8f, 0.3f, 0.2f); // зомби — красноватый
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var d = actionsOut.DiscreteActions;
        d[0] = selectedOption;
    }
}
