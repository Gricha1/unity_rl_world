using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

[RequireComponent(typeof(CharacterController))]
[RequireComponent(typeof(Animator))]
public class AgentGoToHouseDiscrete : Agent, IHasHp
{


    [Header("Spawner")]
    [SerializeField] private TreeSpawner treeSpawner;

    [SerializeField] private SheepSpawner sheepSpawner;

    [Header("HRL - Option Selector")]
    [SerializeField] private OptionSelectorAgent optionSelectorAgent;

    [Header("Current Option Icon")]
    [SerializeField] private SpriteRenderer optionIconRenderer;
    [SerializeField] private Sprite optionWoodSprite;
    [SerializeField] private Sprite optionFoodSprite;
    [SerializeField] private Vector3 optionIconOffset = new Vector3(0f, 2.2f, 0f);
    [SerializeField] private float optionWoodIconScale = 0.55f;
    [SerializeField] private float optionFoodIconScale = 0.35f;
    [SerializeField] private int optionIconSortingOrder = 100;
    [SerializeField] private bool optionIconFaceCamera = true;
    [Tooltip("Снять галочку, чтобы скрыть спрайт задачи (дерево/еда) над агентом.")]
    [SerializeField] private bool showOptionTaskIcon = true;

    private bool _lastShowOptionTaskIcon = true;

    [SerializeField] private float eatDistance = 1.2f; // дистанция до овечки
    [SerializeField] private LayerMask sheepLayer;     // слой овечки

    private float prevSheepDist;


    [Header("Fire / House")]
    [SerializeField] private float houseRadius = 1.2f;
    [Tooltip("Сколько секунд горит один заряд дров (одна единица wood) у дома. Больше — дольше сжигание.")]
    [SerializeField] private float burnInterval = 0.35f;
    private int heatPerWood = 2;        // сколько тепла даёт 1 дерево

    private float burnTimer;

    [Header("Fire VFX")]
    [SerializeField] private GameObject fireVfx;
    [SerializeField] private float fireVfxOffDelaySeconds = 2.0f;
    [SerializeField] private Vector3 fireVfxWorldPosition = new Vector3(-24.97772f, 0.1365422f, 0.7556604f);
    private float fireVfxOffTimer;

    [Header("Wood")]
    [SerializeField] private float chopDistance = 1.5f;
    [SerializeField] private float chopReward = 200.0f;
    [SerializeField] private LayerMask treeLayer;
    public int wood;


    [Header("Target")]
    [SerializeField] private Transform houseTarget;
    [SerializeField] private int maxSteps = 1000;

    [Header("Heat")]
    [SerializeField] private float heatDecayInterval = 5.0f; // секунд на 1 единицу тепла
    public int heat;
    private float heatTimer;

    [Header("Hunger / Satiety")]
    [SerializeField] private int maxSatiety = 20;  // максимум сытости
    public int satiety;                               // текущая сытость
    private float satietyTimer;
    [SerializeField] private float satietyDecayInterval = 5.0f; // секунд на 1 единицу сытости

    [Header("Movement")]
    [SerializeField] private float moveSpeed = 3f;
    [SerializeField] private float rotationSpeed = 120f;

    [Header("Reward")]
    [SerializeField] private float reachDistance = 1.2f;
    [SerializeField] private float reachReward = 10f;

    [SerializeField] private int maxWood = 10;
    [SerializeField] private int maxHeat = 30;

    [Header("HP")]
    [SerializeField] private int maxHp = 100;
    public int hp { get; private set; }
    public int Hp => hp;
    public int MaxHp => maxHp;

    public int currentOptionTrain = 1; // 0 = дерево, 1 = еда

    public int currentOption = 1; // 0 = дерево, 1 = еда

    // Публичные свойства для OptionSelectorAgent
    public int maxWoodPublic => maxWood;
    public int maxHeatPublic => maxHeat;
    public int maxSatietyPublic => maxSatiety;
    public Transform houseTargetPublic => houseTarget;

    // Отслеживание наград для каждой опции
    private float lastRewardForOption0 = 0f;
    private float lastRewardForOption1 = 0f;
    private float currentStepReward = 0f;
    private float accumulatedRewardForOption0 = 0f;
    private float accumulatedRewardForOption1 = 0f;

    private float prevTreeDist;

    private int stepCount;

    private CharacterController controller;
    private Animator animator;

    private Vector3 prevPosition;

    
    private float verticalVelocity;
    [SerializeField] private float gravity = -9.81f;

    [Header("Animation")]
    [Tooltip("Имя Trigger в Animator (должен совпадать с параметром в Animator Controller).")]
    [SerializeField] private string doActionAnimTrigger = "Do";
    [Tooltip("Минимум секунд между срабатываниями DO (анимация + попытка добычи).")]
    [SerializeField] private float doActionCooldownSeconds = 0.45f;
    private int _lastChopActionForAnim;
    private float _doCooldownRemaining;

    public override void Initialize()
    {
        controller = GetComponent<CharacterController>();
        animator = GetComponent<Animator>();
        _lastShowOptionTaskIcon = showOptionTaskIcon;
        _lastChopActionForAnim = 0;
        _doCooldownRemaining = 0f;

        // Jack проходит сквозь цветы: коллизия слой Jack ↔ Flower отключена. Lily на другом слое — застревает и собирает.
        // Важно: у Jack в инспекторе должен быть слой, отличный от Lily (напр. Jack = "Player", Lily = "Default").
        int flowerLayerId = LayerMask.NameToLayer("Flower");
        if (flowerLayerId >= 0)
            Physics.IgnoreLayerCollision(gameObject.layer, flowerLayerId, true);

        EnsureOptionIconRenderer();
        UpdateOptionIconVisual();
    }

    private void EnsureOptionIconRenderer()
    {
        if (!showOptionTaskIcon) return;
        if (optionIconRenderer != null) return;
        if (optionWoodSprite == null && optionFoodSprite == null) return;

        var existing = transform.Find("JackOptionIcon");
        if (existing != null)
        {
            optionIconRenderer = existing.GetComponent<SpriteRenderer>();
            if (optionIconRenderer == null)
                optionIconRenderer = existing.gameObject.AddComponent<SpriteRenderer>();
            optionIconRenderer.sortingOrder = optionIconSortingOrder;
            ApplyOptionIconLocalScale();
            return;
        }

        var go = new GameObject("JackOptionIcon");
        go.transform.SetParent(transform, false);
        var sr = go.AddComponent<SpriteRenderer>();
        sr.sortingOrder = optionIconSortingOrder;
        optionIconRenderer = sr;
        ApplyOptionIconLocalScale();
    }

    private void ApplyOptionIconLocalScale()
    {
        if (optionIconRenderer == null) return;
        float s = currentOptionTrain == 0
            ? Mathf.Max(0.01f, optionWoodIconScale)
            : Mathf.Max(0.01f, optionFoodIconScale);
        optionIconRenderer.transform.localScale = Vector3.one * s;
    }

    private void Update()
    {
        if (_doCooldownRemaining > 0f)
            _doCooldownRemaining -= Time.deltaTime;

        if (showOptionTaskIcon != _lastShowOptionTaskIcon)
        {
            _lastShowOptionTaskIcon = showOptionTaskIcon;
            UpdateOptionIconVisual();
        }

        if (Input.GetKeyDown(KeyCode.E))
            SetOption(currentOptionTrain == 0 ? 1 : 0);
    }

    public override void OnEpisodeBegin()
    {
        float minX = -20.78f;
        float maxX = -12.88f;
        float minZ = -7.30f;
        float maxZ = -0.01f;
        float y = 0.42f;

        float randX = Random.Range(minX, maxX);
        float randZ = Random.Range(minZ, maxZ);

        controller.enabled = false;
        transform.position = new Vector3(randX, y, randZ);
        transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
        controller.enabled = true;

        prevPosition = transform.position;
        stepCount = 0;
        _lastChopActionForAnim = 0;
        _doCooldownRemaining = 0f;

        wood = 0;
        satiety = maxSatiety / 2; // стартуем с половины сытости, например
        satietyTimer = 0f;
        heat = maxHeat;
        heatTimer = 0f;

        burnTimer = 0f;
        fireVfxOffTimer = 0f;
        if (fireVfx != null)
        {
            fireVfx.transform.position = fireVfxWorldPosition;
            fireVfx.SetActive(false);
        }

        prevTreeDist = 0f;
        prevSheepDist = 0f;

        hp = maxHp;

        // Устанавливаем опцию в зависимости от наличия OptionSelectorAgent
        if (optionSelectorAgent != null)
        {
            // Если OptionSelectorAgent есть, опция будет установлена через SetOption() из его OnEpisodeBegin
            // НЕ устанавливаем опцию здесь, чтобы не перезаписать опцию, установленную OptionSelectorAgent
            // Но если опция еще не установлена (например, если наш OnEpisodeBegin вызвался раньше),
            // используем текущую опцию или случайную
            if (currentOptionTrain != 0 && currentOptionTrain != 1)
            {
                // Если опция некорректна, устанавливаем случайную
                int randomOption = Random.Range(0, 2);
                currentOptionTrain = randomOption;
                currentOption = randomOption;
            }
        }
        else
        {
            // Если OptionSelectorAgent нет - случайный выбор задачи для обучения: 0 = дерево, 1 = еда
            int randomOption = Random.Range(0, 2);
            currentOptionTrain = randomOption;
            currentOption = randomOption;
        }
        
        // Финальная синхронизация
        currentOption = currentOptionTrain;
        UpdateOptionIconVisual();

        lastRewardForOption0 = 0f;
        lastRewardForOption1 = 0f;
        currentStepReward = 0f;
        accumulatedRewardForOption0 = 0f;
        accumulatedRewardForOption1 = 0f;

        treeSpawner.ResetTrees();
        sheepSpawner.ResetSheep();
    }

    private bool GetNearestSheep(out GameObject nearestSheep, out float distance)
    {
        nearestSheep = null;
        distance = float.MaxValue;

        Collider[] hits = Physics.OverlapSphere(
            transform.position,
            20f,          // радиус поиска овечек
            sheepLayer
        );

        if (hits.Length == 0)
            return false;

        foreach (var hit in hits)
        {
            float d = Vector3.Distance(transform.position, hit.transform.position);
            if (d < distance)
            {
                distance = d;
                nearestSheep = hit.gameObject;
            }
        }

        return true;
    }

    private bool IsTreeNearby()
    {
        Vector3 origin = transform.position;
        Collider[] hits = Physics.OverlapSphere(origin, chopDistance, treeLayer);

        foreach (var c in hits)
        {
            if (c == null) continue;
            if (HarvestReachDistance(origin, c) <= chopDistance)
                return true;
        }

        return false;
    }

    /// <summary>
    /// Публичный метод для OptionSelectorAgent - проверяет наличие деревьев поблизости
    /// </summary>
    public bool HasTreesNearby()
    {
        return IsTreeNearby();
    }

    /// <summary>
    /// Публичный метод для OptionSelectorAgent - проверяет наличие овец поблизости
    /// </summary>
    public bool HasSheepNearby()
    {
        float r = eatDistance * 2f;
        Vector3 origin = transform.position;
        Collider[] hits = Physics.OverlapSphere(origin, r, sheepLayer);
        foreach (var c in hits)
        {
            if (c == null) continue;
            if (HarvestReachDistance(origin, c) <= r)
                return true;
        }
        return false;
    }

    /// <summary>
    /// Получить расстояние до ближайшей овечки (для OptionSelectorAgent)
    /// </summary>
    public float GetDistanceToNearestSheep()
    {
        if (GetNearestSheep(out GameObject sheep, out float distance))
        {
            return distance;
        }
        return 999f; // большое значение, если овец нет
    }

    /// <summary>
    /// Получить расстояние до ближайшего дерева (для OptionSelectorAgent)
    /// </summary>
    public float GetDistanceToNearestTree()
    {
        if (GetNearestTree(out GameObject tree, out float distance))
        {
            return distance;
        }
        return 999f; // большое значение, если деревьев нет
    }

    /// <summary>
    /// Устанавливает опцию от OptionSelectorAgent
    /// </summary>
    public void SetOption(int option)
    {
        // Проверяем валидность опции
        if (option != 0 && option != 1)
        {
            Debug.LogWarning($"AgentGoToHouseDiscrete.SetOption: Некорректная опция {option}, игнорируем");
            return;
        }
        
        currentOptionTrain = option;
        currentOption = option;
        
        // Дополнительная синхронизация для надежности
        if (currentOptionTrain != currentOption)
        {
            currentOption = currentOptionTrain;
        }

        UpdateOptionIconVisual();
    }

    private void LateUpdate()
    {
        if (!showOptionTaskIcon || optionIconRenderer == null) return;

        optionIconRenderer.transform.position = transform.position + optionIconOffset;

        if (optionIconFaceCamera && Camera.main != null)
        {
            // Billboard icon toward camera for readability.
            var camForward = Camera.main.transform.forward;
            if (camForward.sqrMagnitude > 0.0001f)
                optionIconRenderer.transform.rotation = Quaternion.LookRotation(camForward);
        }
    }

    private void UpdateOptionIconVisual()
    {
        if (!showOptionTaskIcon)
        {
            if (optionIconRenderer != null)
                optionIconRenderer.enabled = false;
            return;
        }

        EnsureOptionIconRenderer();
        if (optionIconRenderer == null) return;

        switch (currentOptionTrain)
        {
            case 0:
                optionIconRenderer.sprite = optionWoodSprite;
                optionIconRenderer.enabled = optionWoodSprite != null;
                break;
            case 1:
                optionIconRenderer.sprite = optionFoodSprite;
                optionIconRenderer.enabled = optionFoodSprite != null;
                break;
            default:
                optionIconRenderer.enabled = false;
                break;
        }

        ApplyOptionIconLocalScale();
    }

    public void TakeDamage(int amount)
    {
        hp = Mathf.Max(0, hp - amount);
        if (hp <= 0)
        {
            EvalEpisodeTracker.NotifyEpisodeEnded();
            EndEpisode();
        }
    }

    /// <summary>
    /// Получить последнюю награду для указанной опции
    /// </summary>
    public float GetLastRewardForOption(int option)
    {
        if (option == 0)
            return lastRewardForOption0;
        else
            return lastRewardForOption1;
    }

    /// <summary>
    /// Деревья висят под TreeSpawner; transform.root = спавнер → нельзя Destroy(root).
    /// Нужен один инстанс — прямой ребёнок спавнера.
    /// </summary>
    private GameObject GetTreeInstanceRoot(Collider hit)
    {
        Transform t = hit.transform;
        if (treeSpawner != null)
        {
            Transform sp = treeSpawner.transform;
            for (; t != null; t = t.parent)
            {
                if (t.parent == sp)
                    return t.gameObject;
            }
        }
        return hit.gameObject;
    }

    private GameObject GetSheepInstanceRoot(Collider hit)
    {
        var wander = hit.GetComponentInParent<SheepWander>();
        if (wander != null)
            return wander.gameObject;
        Transform t = hit.transform;
        if (sheepSpawner != null)
        {
            Transform sp = sheepSpawner.transform;
            for (; t != null; t = t.parent)
            {
                if (t.parent == sp)
                    return t.gameObject;
            }
        }
        return hit.gameObject;
    }

    /// <summary>
    /// Дистанция добычи: до ближайшей точки на коллайдере, а не до pivot (центр дерева недостижим).
    /// </summary>
    private static float HarvestReachDistance(Vector3 from, Collider c)
    {
        return Vector3.Distance(from, c.ClosestPoint(from));
    }

    private bool TryEatSheep()
    {
        Vector3 origin = transform.position;
        Collider[] hits = Physics.OverlapSphere(origin, eatDistance, sheepLayer);

        GameObject bestRoot = null;
        float bestDist = float.MaxValue;
        foreach (var c in hits)
        {
            if (c == null) continue;
            float d = HarvestReachDistance(origin, c);
            if (d > eatDistance) continue;
            GameObject root = GetSheepInstanceRoot(c);
            if (d < bestDist)
            {
                bestDist = d;
                bestRoot = root;
            }
        }

        if (bestRoot == null)
            return false;

        satiety = Mathf.Min(maxSatiety, satiety + 2);
        Destroy(bestRoot);
        return true;
    }


    private bool GetNearestTree(out GameObject nearestTree, out float distance)
    {
        nearestTree = null;
        distance = float.MaxValue;

        Collider[] hits = Physics.OverlapSphere(
            transform.position,
            20f, // радиус поиска больше дистанции рубки
            treeLayer
        );

        if (hits.Length == 0)
            return false;

        foreach (var hit in hits)
        {
            float d = Vector3.Distance(transform.position, hit.transform.position);
            if (d < distance)
            {
                distance = d;
                nearestTree = hit.gameObject;
            }
        }

        return true;
    }


    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.position);
        sensor.AddObservation(transform.forward);

        sensor.AddObservation((float)wood / maxWood); // [0,1]
        sensor.AddObservation((float)heat / maxHeat); // [0,1]
        sensor.AddObservation((float)satiety / maxSatiety); // сытость [0,1]

        bool onHouse =
        Vector3.Distance(transform.position, houseTarget.position) <= houseRadius;

        sensor.AddObservation(onHouse ? 1f : 0f);
        // рядом ли дерево
        bool nearTree = IsTreeNearby();
        sensor.AddObservation(nearTree ? 1f : 0f);

        // One-hot опции: [1,0] = дерево, [0,1] = еда — даёт сети явное разделение режимов
        sensor.AddObservation(currentOption == 0 ? 1f : 0f);
        sensor.AddObservation(currentOption == 1 ? 1f : 0f);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Дополнительная проверка: если OptionSelectorAgent есть, синхронизируем опцию на первом шаге
        // Это защита на случай, если OnEpisodeBegin вызывался в неправильном порядке
        if (optionSelectorAgent != null && stepCount == 0)
        {
            int optionFromSelector = optionSelectorAgent.GetSelectedOption();
            if (optionFromSelector == 0 || optionFromSelector == 1)
            {
                if (currentOptionTrain != optionFromSelector)
                {
                    Debug.LogWarning($"AgentGoToHouseDiscrete: Опция не синхронизирована в начале! Исправляем: {currentOptionTrain} -> {optionFromSelector}");
                    currentOptionTrain = optionFromSelector;
                    currentOption = optionFromSelector;
                }
            }
        }
        // Если OptionSelectorAgent нет - опция уже установлена случайно в OnEpisodeBegin
        
        int moveAction = actions.DiscreteActions[0];
        int rotateAction = actions.DiscreteActions[1];
        int chopAction   = actions.DiscreteActions[2];

        bool chopJustPressed = chopAction == 1 && _lastChopActionForAnim != 1;
        bool doReady = chopJustPressed && _doCooldownRemaining <= 0f;

        if (doReady && doActionAnimTrigger.Length > 0)
            animator.SetTrigger(doActionAnimTrigger);

        float moveInput = 0f;
        float rotateInput = 0f;

        // --- Движение ---
        if (moveAction == 1) moveInput = 1f;
        else if (moveAction == 3) moveInput = -1f;

        // --- Поворот ---
        if (rotateAction == 1) rotateInput = 1f;
        else if (rotateAction == 3) rotateInput = -1f;

        // --- Поворот ---
        transform.Rotate(0f, rotateInput * rotationSpeed * Time.deltaTime, 0f);

        // --- Гравитация ---
        if (controller.isGrounded)
        {
            if (verticalVelocity < 0f)
                verticalVelocity = -2f; // прижимаем к земле
        }
        else
        {
            verticalVelocity += gravity * Time.deltaTime;
        }

        // --- Итоговое движение ---
        Vector3 move = transform.forward * moveInput * moveSpeed +
                    Vector3.up * verticalVelocity;

        controller.Move(move * Time.deltaTime);

        // --- Анимация ---
        animator.SetFloat("Speed", Mathf.Abs(moveInput));

        // --- Reward: прогресс к дому ---
        float prevDist = Vector3.Distance(prevPosition, houseTarget.position);
        float currDist = Vector3.Distance(transform.position, houseTarget.position);

        currentStepReward = 0f; // сбрасываем награду за шаг

        // Добыча: нарастающий фронт DO + кулдаун; только цель текущей опции; дистанция по коллайдеру.
        int currentOptionSnapshot = currentOptionTrain;

        bool choppedTree = false;
        bool ateSheep = false;

        if (doReady)
        {
            if (currentOptionSnapshot == 0)
                choppedTree = TryChopTree();
            else if (currentOptionSnapshot == 1)
                ateSheep = TryEatSheep();

            _doCooldownRemaining = Mathf.Max(0f, doActionCooldownSeconds);
        }
        
        // Reward за рубку дерева - только если опция = дерево (0)
        // Используем snapshot опции для защиты от изменения во время выполнения
        if (choppedTree && currentOptionSnapshot == 0)
        {
            float reward = 10.0f;
            AddReward(reward);
            currentStepReward += reward;
            accumulatedRewardForOption0 += reward;
            lastRewardForOption0 = accumulatedRewardForOption0;
        }
        
        // Reward за поедание овцы - только если опция = еда (1)
        // Используем snapshot опции для защиты от изменения во время выполнения
        if (ateSheep && currentOptionSnapshot == 1)
        {
            float reward = 10.0f;
            AddReward(reward);
            currentStepReward += reward;
            accumulatedRewardForOption1 += reward;
            lastRewardForOption1 = accumulatedRewardForOption1;
        }

        if (currentOptionTrain == 0)
        {
            if (wood >= maxWood)
            {
                float reward = prevDist - currDist;
                AddReward(reward);
                currentStepReward += reward;
                accumulatedRewardForOption0 += reward;
                lastRewardForOption0 = accumulatedRewardForOption0;
            }
            else
            {
                if (GetNearestTree(out GameObject tree, out float currTreeDist))
                {
                    if (prevTreeDist > 0f)
                    {
                        float delta = prevTreeDist - currTreeDist;
                        float reward = delta * 0.5f; // коэффициент подбирается
                        AddReward(reward);
                        currentStepReward += reward;
                        accumulatedRewardForOption0 += reward;
                        lastRewardForOption0 = accumulatedRewardForOption0;
                    }

                    prevTreeDist = currTreeDist;
                }
                else
                {
                    prevTreeDist = 0f;
                }
            }
        }


        if (currentOptionTrain == 1)
        {
            if (GetNearestSheep(out GameObject sheep, out float currSheepDist))
            {
                if (prevSheepDist > 0f)
                {
                    float delta = prevSheepDist - currSheepDist;

                    // подошёл ближе → reward +
                    float reward = delta * 0.5f;
                    AddReward(reward);
                    currentStepReward += reward;
                    accumulatedRewardForOption1 += reward;
                    lastRewardForOption1 = accumulatedRewardForOption1;
                }

                prevSheepDist = currSheepDist;
            }
            else
            {
                prevSheepDist = 0f;
            }
        }

        prevPosition = transform.position;
        
        stepCount++;
        satietyTimer += Time.deltaTime;
        if (satietyTimer >= satietyDecayInterval)
        {
            satiety = Mathf.Max(0, satiety - 1);
            satietyTimer = 0f;

            // штраф, если сытость упала до нуля
            if (satiety == 0)
            {
                if (currentOptionTrain == 1)
                {
                    float penalty = -0.1f;
                    AddReward(penalty);
                    currentStepReward += penalty;
                    accumulatedRewardForOption1 += penalty;
                    lastRewardForOption1 = accumulatedRewardForOption1;
                }
            }
        }

        heatTimer += Time.deltaTime;
        if (heatTimer >= heatDecayInterval)
        {
            heat = Mathf.Max(0, heat - 1);
            heatTimer = 0f;

            if (heat == 0)
            {
                if (currentOptionTrain == 0)
                {
                    float penalty = -0.1f;
                    AddReward(penalty);
                    currentStepReward += penalty;
                    accumulatedRewardForOption0 += penalty;
                    lastRewardForOption0 = accumulatedRewardForOption0;
                }
            }
        }

        bool onHouse = Vector3.Distance(transform.position, houseTarget.position) <= houseRadius;

        if (onHouse && wood > 0)
        {
            if (fireVfx != null)
            {
                fireVfx.transform.position = fireVfxWorldPosition;
                fireVfx.SetActive(true);
                fireVfxOffTimer = fireVfxOffDelaySeconds;
            }

            burnTimer += Time.deltaTime;

            if (burnTimer >= burnInterval)
            {
                burnTimer = 0f;
                wood--;

                if (heat < maxHeat)
                {
                    heat = Mathf.Min(maxHeat, heat + heatPerWood);
                }
                if (currentOptionTrain == 0)
                {
                    float reward = 5.0f;
                    AddReward(reward);
                    currentStepReward += reward;
                    accumulatedRewardForOption0 += reward;
                    lastRewardForOption0 = accumulatedRewardForOption0;

                    // Уведомляем OptionSelectorAgent о успешном выполнении задачи
                    if (optionSelectorAgent != null)
                    {
                        optionSelectorAgent.OnTaskCompleted(0, reward);
                    }
                }
            }
        }
        else
        {
            burnTimer = 0f;
        }

        if (fireVfx != null && !(onHouse && wood > 0) && fireVfx.activeSelf)
        {
            // Stop effect after a delay once we stopped burning.
            if (fireVfxOffTimer <= 0f)
                fireVfxOffTimer = fireVfxOffDelaySeconds;

            fireVfxOffTimer -= Time.deltaTime;
            if (fireVfxOffTimer <= 0f)
                fireVfx.SetActive(false);
        }


        if (stepCount >= maxSteps)
        {
            var statsRecorder = Academy.Instance.StatsRecorder;
            statsRecorder.Add("collision", 0.0f);
            
            // Уведомляем OptionSelectorAgent о завершении эпизода
            // Важно: OptionSelectorAgent должен завершить эпизод синхронно с LowLevelAgent
            // чтобы ML-Agents мог правильно логировать статистику
            if (optionSelectorAgent != null)
            {
                // Вызываем EndEpisode для OptionSelectorAgent перед завершением нашего эпизода
                // Это гарантирует синхронизацию завершения эпизодов
                optionSelectorAgent.EndEpisode();
            }
            
            EvalEpisodeTracker.NotifyEpisodeEnded();
            EndEpisode();
        }

        // Уведомляем OptionSelectorAgent о награде за шаг (если используется HRL)
        if (optionSelectorAgent != null && currentStepReward != 0f)
        {
            // Награда уже добавлена, просто обновляем отслеживание
        }

        _lastChopActionForAnim = chopAction;
    }

    private bool TryChopTree()
    {
        if (wood >= maxWood)
            return false;

        Vector3 origin = transform.position;
        Collider[] hits = Physics.OverlapSphere(origin, chopDistance, treeLayer);

        GameObject bestRoot = null;
        float bestDist = float.MaxValue;
        foreach (var c in hits)
        {
            if (c == null) continue;
            float d = HarvestReachDistance(origin, c);
            if (d > chopDistance) continue;
            GameObject root = GetTreeInstanceRoot(c);
            if (d < bestDist)
            {
                bestDist = d;
                bestRoot = root;
            }
        }

        if (bestRoot == null)
            return false;

        wood++;
        Destroy(bestRoot);
        return true;
    }


    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<int> discreteActions = actionsOut.DiscreteActions;

        // --- Движение только W / S (не стрелки) ---
        int moveAction = 2; // стоять
        if (Input.GetKey(KeyCode.W))
            moveAction = 1;   // вперёд
        else if (Input.GetKey(KeyCode.S))
            moveAction = 3;   // назад

        // --- Поворот только A / D (не стрелки) ---
        int rotateAction = 2; // не крутиться
        if (Input.GetKey(KeyCode.D))
            rotateAction = 1; // вправо
        else if (Input.GetKey(KeyCode.A))
            rotateAction = 3; // влево

        int chopAction = Input.GetMouseButton(0) ? 1 : 0;

        discreteActions[0] = moveAction;
        discreteActions[1] = rotateAction;
        discreteActions[2] = chopAction;
    }

}
