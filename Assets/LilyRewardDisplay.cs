using UnityEngine;
using TMPro;
using Unity.MLAgents;

public class LilyRewardDisplay : MonoBehaviour
{
    [SerializeField] private Agent agent;
    private TMP_Text text;

    void Start()
    {
        text = GetComponent<TMP_Text>();
    }

    void Update()
    {
        if (agent != null && text != null)
            text.text = $"Reward: {agent.GetCumulativeReward():F2}";
    }
}
