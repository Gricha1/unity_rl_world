using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using TMPro;

/// <summary>
/// High-level agent для выбора опций в HRL системе.
/// Выбирает между 0 (дерево) и 1 (еда) на основе текущего состояния.
/// </summary>
public class OptionSelectorAgent : Agent
{
    [Header("Low-level Agent Reference")]
    [SerializeField] private AgentGoToHouseDiscrete lowLevelAgent;

    [Header("UI Display")]
    [SerializeField] private TextMeshPro optionDisplayText; // текст над головой агента
    [SerializeField] private float displayHeight = 2.5f; // высота над агентом

    [Header("Option Selection")]
    private int selectedOption = 1; // текущая выбранная опция
    private int lastSelectedOption = 1;
    private int stepsSinceLastOptionChange = 0; // шаги с момента последней смены опции
    private const int OPTION_CHANGE_INTERVAL = 20; // опцию можно менять каждые 20 шагов (совпадает с Decision Period)
    
    [Header("Heuristic Control")]
    private int heuristicOption = 1; // опция для heuristic режима
    private bool eKeyPressedLastFrame = false; // для отслеживания нажатия клавиши E

    [Header("Reward Tracking")]
    private float episodeReward = 0f;
    private float lastOptionReward = 0f;
    private int optionChangeCount = 0;
    private float optionStartTime = 0f;
    private const float OPTION_TIMEOUT = 200f; // максимальное время для выполнения опции
    
    [Header("Episode Synchronization")]
    private int lastLowLevelAgentStepCount = 0; // для отслеживания завершения эпизода LowLevelAgent

    public override void Initialize()
    {
        if (lowLevelAgent == null)
        {
            Debug.LogError("OptionSelectorAgent: lowLevelAgent не назначен!");
        }

        // Проверяем Behavior Parameters
        var behaviorParams = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
        if (behaviorParams != null)
        {
            string behaviorName = behaviorParams.BehaviorName;
            Debug.Log($"OptionSelectorAgent: Инициализирован с Behavior Name: '{behaviorName}'");
            
            if (behaviorName != "OptionSelector")
            {
                Debug.LogWarning($"OptionSelectorAgent: Behavior Name '{behaviorName}' не совпадает с конфигом 'OptionSelector'! Это может быть причиной отсутствия обучения.");
            }
            
            // Проверяем, что агент активен
            if (!gameObject.activeInHierarchy)
            {
                Debug.LogError("OptionSelectorAgent: GameObject не активен! Агент не будет обучаться.");
            }
            
            // Проверяем Max Step
            if (MaxStep == 0)
            {
                Debug.LogWarning("OptionSelectorAgent: Max Step = 0. Это нормально для агента, который работает вместе с другим агентом.");
            }
        }
        else
        {
            Debug.LogError("OptionSelectorAgent: BehaviorParameters компонент не найден! Агент не будет обучаться.");
        }

        // Проверяем наличие Decision Requester (критически важно для обучения!)
        var decisionRequester = GetComponent<Unity.MLAgents.DecisionRequester>();
        if (decisionRequester == null)
        {
            Debug.LogError("OptionSelectorAgent: Decision Requester компонент не найден! Агент не будет запрашивать действия от политики и не будет обучаться. Добавьте компонент Decision Requester вручную в Unity Inspector.");
        }
        else
        {
            Debug.Log($"OptionSelectorAgent: Decision Requester найден. Decision Period: {decisionRequester.DecisionPeriod}");
            
            // Рекомендуем установить Decision Period равным OPTION_CHANGE_INTERVAL для эффективности
            if (decisionRequester.DecisionPeriod != OPTION_CHANGE_INTERVAL)
            {
                Debug.LogWarning($"OptionSelectorAgent: Decision Period ({decisionRequester.DecisionPeriod}) не совпадает с OPTION_CHANGE_INTERVAL ({OPTION_CHANGE_INTERVAL}). Рекомендуется установить Decision Period = {OPTION_CHANGE_INTERVAL} для оптимальной производительности.");
            }
        }

        // Создаем текст над головой агента, если он не назначен
        if (optionDisplayText == null && lowLevelAgent != null)
        {
            CreateOptionDisplay();
        }
    }

    private void CreateOptionDisplay()
    {
        // Создаем GameObject для текста
        GameObject textObject = new GameObject("OptionDisplay");
        textObject.transform.SetParent(lowLevelAgent.transform);
        textObject.transform.localPosition = new Vector3(0, displayHeight, 0);
        textObject.transform.localRotation = Quaternion.identity;
        textObject.transform.localScale = Vector3.one;

        // Добавляем TextMeshPro компонент
        optionDisplayText = textObject.AddComponent<TextMeshPro>();
        optionDisplayText.text = "ЕДА";
        optionDisplayText.fontSize = 3;
        optionDisplayText.alignment = TextAlignmentOptions.Center;
        optionDisplayText.color = Color.yellow;
        optionDisplayText.fontStyle = FontStyles.Bold;

        // Настраиваем для 3D отображения
        optionDisplayText.sortingOrder = 100;
    }

    public override void OnEpisodeBegin()
    {
        Debug.Log($"OptionSelectorAgent: OnEpisodeBegin вызван, totalSteps до сброса: {totalSteps}");
        totalSteps = 0; // сбрасываем счетчик для диагностики
        observationCount = 0; // сбрасываем счетчик наблюдений
        rewardCalculationCount = 0; // сбрасываем счетчик наград
        
        selectedOption = Random.Range(0, 2); // случайный выбор в начале эпизода
        lastSelectedOption = selectedOption;
        heuristicOption = selectedOption; // синхронизируем heuristic опцию
        episodeReward = 0f;
        lastOptionReward = 0f;
        optionChangeCount = 0;
        optionStartTime = 0f;
        stepsSinceLastOptionChange = 0; // сбрасываем счетчик шагов
        eKeyPressedLastFrame = false;
        
        // Синхронизируем счетчик шагов LowLevelAgent
        if (lowLevelAgent != null)
        {
            lastLowLevelAgentStepCount = lowLevelAgent.StepCount;
        }

        // ВАЖНО: Передаем опцию низкоуровневому агенту СРАЗУ в начале эпизода
        // Это гарантирует, что опция установлена до первого шага
        if (lowLevelAgent != null)
        {
            lowLevelAgent.SetOption(selectedOption);
            
            // Дополнительная проверка: убеждаемся, что опция установлена правильно
            int verifiedOption = lowLevelAgent.currentOptionTrain;
            if (verifiedOption != selectedOption)
            {
                Debug.LogWarning($"OptionSelectorAgent: Опция не синхронизирована! Установили {selectedOption}, получили {verifiedOption}");
                // Пытаемся установить еще раз
                lowLevelAgent.SetOption(selectedOption);
            }
        }
    }

    private int observationCount = 0; // для диагностики

    public override void CollectObservations(VectorSensor sensor)
    {
        observationCount++;
        
        // Диагностика: логируем первые несколько вызовов
        if (observationCount <= 5 || observationCount % 1000 == 0)
        {
            Debug.Log($"OptionSelectorAgent: CollectObservations вызван {observationCount} раз");
        }
        
        if (lowLevelAgent == null)
        {
            Debug.LogWarning("OptionSelectorAgent: lowLevelAgent не назначен! Отправляем нулевые наблюдения.");
            // Если агент не назначен, отправляем нулевые наблюдения
            for (int i = 0; i < GetObservationSize(); i++)
            {
                sensor.AddObservation(0f);
            }
            return;
        }

        // Наблюдения о состоянии низкоуровневого агента
        sensor.AddObservation((float)lowLevelAgent.wood / lowLevelAgent.maxWoodPublic); // [0,1]
        sensor.AddObservation((float)lowLevelAgent.heat / lowLevelAgent.maxHeatPublic); // [0,1]
        sensor.AddObservation((float)lowLevelAgent.satiety / lowLevelAgent.maxSatietyPublic); // [0,1]

        // Расстояния до важных объектов (нормализованные)
        Vector3 agentPos = lowLevelAgent.transform.position;
        
        // Расстояние до дома
        float distToHouse = Vector3.Distance(
            agentPos,
            lowLevelAgent.houseTargetPublic.position
        );
        sensor.AddObservation(distToHouse / 50f); // нормализация

        // Расстояние до ближайшей овечки
        float distToSheep = lowLevelAgent.GetDistanceToNearestSheep();
        sensor.AddObservation(distToSheep / 50f); // нормализация

        // Расстояние до ближайшего дерева
        float distToTree = lowLevelAgent.GetDistanceToNearestTree();
        sensor.AddObservation(distToTree / 50f); // нормализация

        // Текущая выбранная опция
        sensor.AddObservation((float)selectedOption);

        // Время с момента выбора опции
        sensor.AddObservation(optionStartTime / OPTION_TIMEOUT);

        // Шаги с момента последней смены опции (нормализовано)
        sensor.AddObservation((float)stepsSinceLastOptionChange / OPTION_CHANGE_INTERVAL);
    }

    private int totalSteps = 0; // для диагностики

    public override void OnActionReceived(ActionBuffers actions)
    {
        totalSteps++;
        
        // Диагностика: логируем каждые 1000 шагов
        if (totalSteps % 1000 == 0)
        {
            Debug.Log($"OptionSelectorAgent: Получено действие на шаге {totalSteps}, текущая опция: {selectedOption}");
        }
        
        // Увеличиваем счетчик шагов
        stepsSinceLastOptionChange++;
        optionStartTime += 1f;

        // Получаем выбранную опцию (0 или 1)
        int newOption = actions.DiscreteActions[0];

        // Опцию можно менять только каждые OPTION_CHANGE_INTERVAL шагов
        if (stepsSinceLastOptionChange >= OPTION_CHANGE_INTERVAL)
        {
            // Если опция изменилась, обновляем
            if (newOption != selectedOption)
            {
                lastSelectedOption = selectedOption;
                selectedOption = newOption;
                optionChangeCount++;
                optionStartTime = 0f;
                stepsSinceLastOptionChange = 0; // сбрасываем счетчик

                // Передаем новую опцию низкоуровневому агенту
                if (lowLevelAgent != null)
                {
                    lowLevelAgent.SetOption(selectedOption);
                }

                // Обновляем отображение
                UpdateOptionDisplay();
            }
            else
            {
                // Опция не изменилась, но прошло достаточно шагов - сбрасываем счетчик
                stepsSinceLastOptionChange = 0;
            }
        }
        // Если прошло меньше OPTION_CHANGE_INTERVAL шагов, игнорируем новое действие
        // и продолжаем использовать текущую опцию

        // Reward на основе успешности выполнения опции
        CalculateOptionReward();
        
        // Проверяем, завершил ли LowLevelAgent эпизод
        // Если StepCount LowLevelAgent сбросился (стал меньше предыдущего значения),
        // значит LowLevelAgent начал новый эпизод, и нам нужно завершить наш эпизод
        if (lowLevelAgent != null)
        {
            int currentLowLevelStepCount = lowLevelAgent.StepCount;
            
            // Если StepCount сбросился (начался новый эпизод), завершаем наш эпизод
            if (currentLowLevelStepCount < lastLowLevelAgentStepCount && lastLowLevelAgentStepCount > 0)
            {
                Debug.Log($"OptionSelectorAgent: LowLevelAgent завершил эпизод (StepCount: {lastLowLevelAgentStepCount} -> {currentLowLevelStepCount}), завершаем наш эпизод");
                EndEpisode();
            }
            
            lastLowLevelAgentStepCount = currentLowLevelStepCount;
        }
    }

    private int rewardCalculationCount = 0; // для диагностики

    private void CalculateOptionReward()
    {
        rewardCalculationCount++;
        
        if (lowLevelAgent == null)
        {
            if (rewardCalculationCount % 1000 == 0)
            {
                Debug.LogWarning("OptionSelectorAgent: CalculateOptionReward вызван, но lowLevelAgent = null");
            }
            return;
        }

        float currentReward = 0f;

        // ===== ШТРАФЫ ЗА НИЗКИЕ УРОВНИ РЕСУРСОВ =====
        // Чем ниже уровень ресурса, тем больший штраф получает OptionSelectorAgent
        
        // Штраф за низкий уровень ДЕРЕВА (wood)
        float woodRatio = (float)lowLevelAgent.wood / lowLevelAgent.maxWoodPublic;
        if (woodRatio < 0.3f)
        {
            // Штраф увеличивается по мере уменьшения уровня дерева
            float penalty = (0.3f - woodRatio) * 0.5f; // максимальный штраф ~0.15 при woodRatio = 0
            currentReward -= penalty;
        }
        if (woodRatio == 0f)
        {
            currentReward -= 0.2f; // дополнительный штраф за полное отсутствие дерева
        }

        // Штраф за низкий уровень ТЕПЛА (heat)
        float heatRatio = (float)lowLevelAgent.heat / lowLevelAgent.maxHeatPublic;
        if (heatRatio < 0.3f)
        {
            // Штраф увеличивается по мере уменьшения уровня тепла
            float penalty = (0.3f - heatRatio) * 0.5f; // максимальный штраф ~0.15 при heatRatio = 0
            currentReward -= penalty;
        }
        if (heatRatio == 0f)
        {
            currentReward -= 0.3f; // большой штраф за полное отсутствие тепла (замерзание)
        }

        // Штраф за низкий уровень ПИТАНИЯ (satiety)
        float satietyRatio = (float)lowLevelAgent.satiety / lowLevelAgent.maxSatietyPublic;
        if (satietyRatio < 0.3f)
        {
            // Штраф увеличивается по мере уменьшения уровня питания
            float penalty = (0.3f - satietyRatio) * 0.5f; // максимальный штраф ~0.15 при satietyRatio = 0
            currentReward -= penalty;
        }
        if (satietyRatio == 0f)
        {
            currentReward -= 0.3f; // большой штраф за полное отсутствие питания (голод)
        }

        // ===== ШТРАФЫ ЗА ПОВЕДЕНИЕ =====
        
        // Штраф за слишком частую смену опций
        if (optionChangeCount > 10)
        {
            currentReward -= 0.01f * (optionChangeCount - 10);
        }

        // Штраф за таймаут опции
        if (optionStartTime > OPTION_TIMEOUT)
        {
            currentReward -= 0.5f;
        }

        AddReward(currentReward);
        episodeReward += currentReward;
    }

    /// <summary>
    /// Вызывается низкоуровневым агентом при успешном выполнении задачи
    /// </summary>
    public void OnTaskCompleted(int option, float successReward)
    {
        if (option == selectedOption)
        {
            AddReward(successReward);
            episodeReward += successReward;
        }
    }

    /// <summary>
    /// Получить текущую выбранную опцию
    /// </summary>
    public int GetSelectedOption()
    {
        return selectedOption;
    }

    private int GetObservationSize()
    {
        return 10; // wood, heat, satiety, distToHouse, distToSheep, distToTree, currentOption, timeSinceOption, stepsSinceLastChange
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<int> discreteActions = actionsOut.DiscreteActions;
        
        // В heuristic режиме используем текущую опцию (переключение обрабатывается в Update)
        discreteActions[0] = heuristicOption;
        
        // Дополнительно: можно использовать автоматическую логику как fallback
        if (lowLevelAgent == null)
        {
            discreteActions[0] = heuristicOption;
            return;
        }
        
        // Автоматическая логика выбора опции (если не используется ручное переключение)
        // Если тепло низкое - выбираем дерево
        if (lowLevelAgent.heat < lowLevelAgent.maxHeatPublic * 0.3f)
        {
            heuristicOption = 0;
        }
        // Если сытость низкая - выбираем еду
        else if (lowLevelAgent.satiety < lowLevelAgent.maxSatietyPublic * 0.3f)
        {
            heuristicOption = 1;
        }
        
        discreteActions[0] = heuristicOption;
    }

    private void Update()
    {
        // Проверяем завершение эпизода LowLevelAgent (дополнительная проверка в Update для надежности)
        // Это важно, так как OnActionReceived может вызываться редко из-за Decision Period = 20
        if (lowLevelAgent != null)
        {
            int currentLowLevelStepCount = lowLevelAgent.StepCount;
            
            // Если StepCount сбросился (начался новый эпизод), завершаем наш эпизод
            if (currentLowLevelStepCount < lastLowLevelAgentStepCount && lastLowLevelAgentStepCount > 0)
            {
                Debug.Log($"OptionSelectorAgent (Update): LowLevelAgent завершил эпизод (StepCount: {lastLowLevelAgentStepCount} -> {currentLowLevelStepCount}), завершаем наш эпизод");
                EndEpisode();
                lastLowLevelAgentStepCount = currentLowLevelStepCount;
            }
        }
        
        // Проверяем нажатие клавиши E для переключения опции (работает в любом режиме)
        bool eKeyPressed = Input.GetKey(KeyCode.E);
        
        // Переключаем опцию при нажатии E (только один раз при нажатии, не удерживании)
        if (eKeyPressed && !eKeyPressedLastFrame)
        {
            // Переключаем опцию: 0 -> 1, 1 -> 0
            int newOption = (selectedOption == 0) ? 1 : 0;
            
            // Сразу передаем новую опцию низкоуровневому агенту
            if (lowLevelAgent != null)
            {
                lowLevelAgent.SetOption(newOption);
                selectedOption = newOption;
                heuristicOption = newOption; // синхронизируем heuristic опцию
                optionStartTime = 0f;
                stepsSinceLastOptionChange = 0;
                
                Debug.Log($"OptionSelectorAgent: Опция переключена на {(newOption == 0 ? "ДЕРЕВО" : "ЕДА")}");
                
                // Обновляем отображение
                UpdateOptionDisplay();
            }
        }
        
        eKeyPressedLastFrame = eKeyPressed;

        // Обновляем позицию текста, чтобы он всегда был над головой агента
        if (optionDisplayText != null && lowLevelAgent != null)
        {
            // Поворачиваем текст к камере (billboard effect)
            if (Camera.main != null)
            {
                optionDisplayText.transform.LookAt(Camera.main.transform);
                optionDisplayText.transform.Rotate(0, 180, 0); // разворачиваем, чтобы текст был читаемым
            }

            // Обновляем текст в зависимости от текущей опции
            if (optionDisplayText.text != GetOptionDisplayText())
            {
                optionDisplayText.text = GetOptionDisplayText();
            }
        }
    }

    private string GetOptionDisplayText()
    {
        int currentOpt = (lowLevelAgent != null && lowLevelAgent.currentOptionTrain == selectedOption) 
            ? selectedOption 
            : selectedOption;
        
        return currentOpt == 0 ? "ДЕРЕВО" : "ЕДА";
    }

    private void UpdateOptionDisplay()
    {
        if (optionDisplayText != null)
        {
            optionDisplayText.text = GetOptionDisplayText();
            
            // Меняем цвет в зависимости от опции (желтый для обеих опций для лучшей видимости)
            optionDisplayText.color = selectedOption == 0 ? Color.yellow : new Color(1f, 0.6f, 0.2f);
        }
    }
}
