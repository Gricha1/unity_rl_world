using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

[RequireComponent(typeof(CharacterController))]
[RequireComponent(typeof(Animator))]
public class LilyScript : Agent, IHasHp
{
    /// <summary>0 = цветы, 1 = поцелуй Джека, 2 = зомби (стрелять по зомби, штраф за попадание в Джека).</summary>
    private int currentOption;

    [Header("Option Selector (HRL)")]
    [SerializeField] private LilyOptionSelectorAgent optionSelectorAgent;

    [Header("Current Option Icon")]
    [SerializeField] private SpriteRenderer optionIconRenderer;
    [SerializeField] private Sprite optionFlowerSprite;
    [SerializeField] private Sprite optionKissSprite;
    [SerializeField] private Sprite optionZombieSprite;
    [SerializeField] private Vector3 optionIconOffset = new Vector3(0f, 2.2f, 0f);
    [SerializeField] private float optionFlowerIconScale = 0.45f;
    [SerializeField] private float optionKissIconScale = 0.45f;
    [SerializeField] private float optionZombieIconScale = 0.45f;
    [SerializeField] private int optionIconSortingOrder = 100;
    [SerializeField] private bool optionIconFaceCamera = true;
    [Tooltip("Снять галочку, чтобы скрыть иконку задачи над агентом.")]
    [SerializeField] private bool showOptionTaskIcon = true;

    private bool _lastShowOptionTaskIcon = true;

    [Header("Jack (опция «поцелуй»)")]
    [SerializeField] private Transform jackTarget;
    [SerializeField] private LayerMask jackLayer;
    [SerializeField] private float kissDistance = 2.5f;
    [SerializeField] private float kissReward = 10f;
    [SerializeField] private float moveTowardsJackRewardScale = 0.3f;

    [Header("Flower Spawner")]
    [SerializeField] private FlowerSpawner flowerSpawner;
    [SerializeField] private LayerMask flowerLayer;

    [Header("Collect")]
    [SerializeField] private float collectDistance = 1.8f;
    [SerializeField] private float collectReward = 10f;

    [Header("Animation (DO — сбор / поцелуй)")]
    [Tooltip("Trigger в Animator Lily (как у Jack).")]
    [SerializeField] private string doActionAnimTrigger = "Do";
    [Tooltip("Пауза между срабатываниями DO (сбор цветка или поцелуй).")]
    [SerializeField] private float collectActionCooldownSeconds = 0.45f;
    private int _lastCollectAction;
    private float _doCooldownRemaining;

    [Header("Movement")]
    [SerializeField] private float moveSpeed = 3f;
    [SerializeField] private float rotationSpeed = 120f;

    [Header("Rewards")]
    [SerializeField] private float moveTowardsFlowerRewardScale = 0.3f;
    [SerializeField] private float stepPenalty = -0.001f;

    [Header("Счётчики (растут от действий, со временем падают)")]
    [SerializeField] private int maxFlowerCount = 20;
    [SerializeField] private float flowerDecayInterval = 8f;
    [SerializeField] private int maxLove = 100;
    [SerializeField] private float loveDecayInterval = 8f;

    [Header("Shoot")]
    [SerializeField] private int bulletDamage = 10;
    [SerializeField] private float bulletSpeed = 15f;
    [SerializeField] private float shootCooldown = 0.5f;
    private float shootCooldownTimer;

    [Header("Zombie (опция «зомби»)")]
    [SerializeField] private LayerMask zombieLayer;
    [SerializeField] private float zombieHitReward = 10f;
    [SerializeField] private float jackHitPenalty = -5f;

    [Header("HP")]
    [SerializeField] private int maxHp = 100;
    public int hp { get; private set; }

    [Header("Curriculum (последовательное обучение)")]
    [Tooltip("Включить для первой стадии: выстрел ничего не делает, опция «зомби» не выбирается. Action space остаётся 4, наблюдения те же.")]
    [SerializeField] private bool curriculumNoShootNoZombie = false;

    [Header("Episode")]
    [SerializeField] private int maxSteps = 1500;

    /// <summary>True — режим curriculum: выстрел отключён, опция зомби недоступна.</summary>
    public bool CurriculumNoShootNoZombie => curriculumNoShootNoZombie;

    private CharacterController controller;
    private Animator animator;
    private float verticalVelocity;
    [SerializeField] private float gravity = -9.81f;
    private float prevFlowerDist = -1f;
    private float prevJackDist = -1f;
    private int stepCount;
    private float flowerDecayTimer;
    private float loveDecayTimer;

    public int FlowerCount { get; private set; }
    public int Love { get; private set; }
    public int Hp => hp;
    public int MaxHp => maxHp;

    private const float MaxFlowerDistForObs = 50f;
    private const float MaxJackDistForObs = 50f;
    private const float MaxZombieDistForObs = 50f;

    public int StepCount => stepCount;
    public int GetCurrentOption() => currentOption;

    public float GetDistanceToJackNormalized()
    {
        float d = GetDistanceToJack(out _);
        return Mathf.Clamp01(d / MaxJackDistForObs);
    }

    public float GetDistanceToNearestFlowerNormalized()
    {
        if (GetNearestFlower(out _, out float dist))
            return Mathf.Clamp01(dist / MaxFlowerDistForObs);
        return 1f;
    }

    public float GetDistanceToNearestZombieNormalized()
    {
        if (GetNearestZombie(out _, out float dist))
            return Mathf.Clamp01(dist / MaxZombieDistForObs);
        return 1f;
    }

    public void SetOption(int option)
    {
        if (option < 0 || option > 2)
            return;
        if (curriculumNoShootNoZombie && option == 2)
            option = 0;
        currentOption = option;
        UpdateOptionIconVisual();
    }

    /// <summary>Вызывается пулей при попадании в объект на слое Zombie.</summary>
    public void OnBulletHitZombie()
    {
        AddReward(zombieHitReward);
        optionSelectorAgent?.AddOptionReward(2, zombieHitReward);
    }

    /// <summary>Вызывается пулей при попадании в Джека (слой Jack). Штраф при опции «Зомби».</summary>
    public void OnBulletHitJack()
    {
        AddReward(jackHitPenalty);
        optionSelectorAgent?.AddOptionReward(2, jackHitPenalty);
    }

    public override void Initialize()
    {
        controller = GetComponent<CharacterController>();
        animator = GetComponent<Animator>();
        _lastCollectAction = 0;
        _doCooldownRemaining = 0f;
        _lastShowOptionTaskIcon = showOptionTaskIcon;
        EnsureOptionIconRenderer();
        UpdateOptionIconVisual();
    }

    private void EnsureOptionIconRenderer()
    {
        if (!showOptionTaskIcon) return;
        if (optionIconRenderer != null) return;
        if (optionFlowerSprite == null && optionKissSprite == null && optionZombieSprite == null) return;

        var existing = transform.Find("LilyOptionIcon");
        if (existing != null)
        {
            optionIconRenderer = existing.GetComponent<SpriteRenderer>();
            if (optionIconRenderer == null)
                optionIconRenderer = existing.gameObject.AddComponent<SpriteRenderer>();
            optionIconRenderer.sortingOrder = optionIconSortingOrder;
            ApplyOptionIconLocalScale();
            return;
        }

        var go = new GameObject("LilyOptionIcon");
        go.transform.SetParent(transform, false);
        var sr = go.AddComponent<SpriteRenderer>();
        sr.sortingOrder = optionIconSortingOrder;
        optionIconRenderer = sr;
        ApplyOptionIconLocalScale();
    }

    private void ApplyOptionIconLocalScale()
    {
        if (optionIconRenderer == null) return;
        float s = currentOption switch
        {
            0 => Mathf.Max(0.01f, optionFlowerIconScale),
            1 => Mathf.Max(0.01f, optionKissIconScale),
            2 => Mathf.Max(0.01f, optionZombieIconScale),
            _ => 0.35f
        };
        optionIconRenderer.transform.localScale = Vector3.one * s;
    }

    private void LateUpdate()
    {
        if (!showOptionTaskIcon || optionIconRenderer == null) return;

        optionIconRenderer.transform.position = transform.position + optionIconOffset;

        if (optionIconFaceCamera && Camera.main != null)
        {
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

        switch (currentOption)
        {
            case 0:
                optionIconRenderer.sprite = optionFlowerSprite;
                optionIconRenderer.enabled = optionFlowerSprite != null;
                break;
            case 1:
                optionIconRenderer.sprite = optionKissSprite;
                optionIconRenderer.enabled = optionKissSprite != null;
                break;
            case 2:
                optionIconRenderer.sprite = optionZombieSprite;
                optionIconRenderer.enabled = optionZombieSprite != null;
                break;
            default:
                optionIconRenderer.enabled = false;
                break;
        }

        ApplyOptionIconLocalScale();
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

        if (Input.GetKeyDown(KeyCode.T))
        {
            int next = (currentOption + 1) % 3;
            if (curriculumNoShootNoZombie && next == 2)
                next = 0;
            SetOption(next);
        }
    }

    public void TakeDamage(int amount)
    {
        hp = Mathf.Max(0, hp - amount);
        if (hp <= 0)
        {
            optionSelectorAgent?.EndEpisode();
            EvalEpisodeTracker.NotifyEpisodeEnded();
            EndEpisode();
        }
    }

    public override void OnEpisodeBegin()
    {
        stepCount = 0;
        prevFlowerDist = -1f;
        prevJackDist = -1f;
        FlowerCount = 0;
        Love = 0;
        flowerDecayTimer = 0f;
        loveDecayTimer = 0f;
        shootCooldownTimer = 0f;
        hp = maxHp;
        _lastCollectAction = 0;
        _doCooldownRemaining = 0f;

        // Опция: от селектора или случайная (если селектора нет)
        if (optionSelectorAgent != null)
            currentOption = optionSelectorAgent.GetSelectedOption();
        else
            currentOption = Random.Range(0, 3);
        if (curriculumNoShootNoZombie && currentOption == 2)
            currentOption = 0;

        // Те же координаты спавна, что у JackScript
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

        if (flowerSpawner != null)
            flowerSpawner.ResetFlowers();

        UpdateOptionIconVisual();
    }

    private bool GetNearestFlower(out GameObject nearestFlower, out float distance)
    {
        nearestFlower = null;
        distance = float.MaxValue;

        Collider[] hits = Physics.OverlapSphere(transform.position, MaxFlowerDistForObs, flowerLayer);
        foreach (var hit in hits)
        {
            if (hit == null || !hit.gameObject.activeInHierarchy) continue;
            float d = Vector3.Distance(transform.position, hit.transform.position);
            if (d < distance)
            {
                distance = d;
                nearestFlower = hit.gameObject;
            }
        }
        return nearestFlower != null;
    }

    private float GetDistanceToJack(out Vector3 dirToJack)
    {
        dirToJack = Vector3.zero;
        if (jackTarget == null) return float.MaxValue;
        Vector3 delta = jackTarget.position - transform.position;
        delta.y = 0f;
        float d = delta.magnitude;
        if (d > 0.001f) dirToJack = delta.normalized;
        return d;
    }

    private bool IsJackInKissRange()
    {
        if (jackTarget == null) return false;
        Vector3 p = transform.position;
        Vector3 j = jackTarget.position;
        p.y = 0f;
        j.y = 0f;
        float d = Vector3.Distance(p, j);
        if (d <= kissDistance) return true;
        Collider[] hits = Physics.OverlapSphere(transform.position, kissDistance, jackLayer);
        foreach (var h in hits)
            if (h != null && (h.transform == jackTarget || h.transform.IsChildOf(jackTarget)))
                return true;
        return false;
    }

    private bool GetNearestZombie(out GameObject nearestZombie, out float distance)
    {
        nearestZombie = null;
        distance = float.MaxValue;
        if (zombieLayer == 0) return false;
        Collider[] hits = Physics.OverlapSphere(transform.position, MaxZombieDistForObs, zombieLayer);
        foreach (var hit in hits)
        {
            if (hit == null || !hit.gameObject.activeInHierarchy) continue;
            float d = Vector3.Distance(transform.position, hit.transform.position);
            if (d < distance)
            {
                distance = d;
                nearestZombie = hit.gameObject;
            }
        }
        return nearestZombie != null;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.position);
        sensor.AddObservation(transform.forward);

        // One-hot опции: 0 = цветы, 1 = поцелуй Джека, 2 = зомби
        sensor.AddObservation(currentOption == 0 ? 1f : 0f);
        sensor.AddObservation(currentOption == 1 ? 1f : 0f);
        sensor.AddObservation(currentOption == 2 ? 1f : 0f);

        // Счётчики (нормализованные [0,1])
        sensor.AddObservation(maxFlowerCount > 0 ? (float)FlowerCount / maxFlowerCount : 0f);
        sensor.AddObservation(maxLove > 0 ? (float)Love / maxLove : 0f);

        // Цветы
        if (GetNearestFlower(out GameObject flower, out float dist))
        {
            sensor.AddObservation(Mathf.Clamp01(dist / MaxFlowerDistForObs));
            Vector3 dir = (flower.transform.position - transform.position).normalized;
            sensor.AddObservation(dir.x);
            sensor.AddObservation(dir.z);
            sensor.AddObservation(dist <= collectDistance ? 1f : 0f);
        }
        else
        {
            sensor.AddObservation(1f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }

        // Джек и зомби — только через лидар (компонент на агенте), координаты сюда не подаём
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (optionSelectorAgent != null && stepCount == 0)
        {
            int opt = optionSelectorAgent.GetSelectedOption();
            if (opt >= 0 && opt <= 2)
            {
                if (curriculumNoShootNoZombie && opt == 2)
                    opt = 0;
                currentOption = opt;
                UpdateOptionIconVisual();
            }
        }

        int moveAction = actions.DiscreteActions[0];
        int rotateAction = actions.DiscreteActions[1];
        int collectAction = actions.DiscreteActions[2];
        int shootAction = actions.DiscreteActions[3];

        bool collectJustPressed = collectAction == 1 && _lastCollectAction != 1;
        bool collectDoRelevant = currentOption == 0 || currentOption == 1;
        bool collectReady = collectJustPressed && _doCooldownRemaining <= 0f && collectDoRelevant;

        if (collectReady && animator != null && doActionAnimTrigger.Length > 0)
            animator.SetTrigger(doActionAnimTrigger);

        shootCooldownTimer -= Time.deltaTime;
        if (shootAction == 1 && shootCooldownTimer <= 0f)
        {
            if (!curriculumNoShootNoZombie)
                SpawnBullet();
            shootCooldownTimer = shootCooldown;
        }

        float moveInput = 0f;
        if (moveAction == 2) moveInput = 1f;
        else if (moveAction == 0) moveInput = -1f;

        float rotateInput = 0f;
        if (rotateAction == 2) rotateInput = 1f;
        else if (rotateAction == 0) rotateInput = -1f;

        transform.Rotate(0f, rotateInput * rotationSpeed * Time.deltaTime, 0f);

        if (controller.isGrounded)
            verticalVelocity = verticalVelocity < 0f ? -2f : verticalVelocity;
        else
            verticalVelocity += gravity * Time.deltaTime;

        Vector3 move = transform.forward * moveInput * moveSpeed + Vector3.up * verticalVelocity;
        controller.Move(move * Time.deltaTime);

        if (animator != null)
            animator.SetFloat("Speed", Mathf.Abs(moveInput));

        AddReward(stepPenalty);

        // Затухание счётчиков со временем
        flowerDecayTimer += Time.deltaTime;
        if (flowerDecayTimer >= flowerDecayInterval)
        {
            flowerDecayTimer = 0f;
            if (FlowerCount > 0) FlowerCount--;
        }
        loveDecayTimer += Time.deltaTime;
        if (loveDecayTimer >= loveDecayInterval)
        {
            loveDecayTimer = 0f;
            if (Love > 0) Love--;
        }

        if (currentOption == 0)
        {
            // Опция: собирать цветы — один раз на фронте DO + кулдаун
            bool collected = false;
            if (collectReady)
                collected = TryCollectFlower();

            if (collected)
            {
                FlowerCount = Mathf.Min(maxFlowerCount, FlowerCount + 1);
                AddReward(collectReward);
                optionSelectorAgent?.AddOptionReward(0, collectReward);
                prevFlowerDist = -1f;
            }
            else if (GetNearestFlower(out _, out float currDist))
            {
                if (prevFlowerDist > 0f)
                    AddReward((prevFlowerDist - currDist) * moveTowardsFlowerRewardScale);
                prevFlowerDist = currDist;
            }
            else
                prevFlowerDist = -1f;
            prevJackDist = -1f;
        }
        else if (currentOption == 1)
        {
            // Опция: поцелуй — один раз на фронте DO + кулдаун, в радиусе поцелуя
            float currJackDist = GetDistanceToJack(out _);
            if (collectReady && IsJackInKissRange())
            {
                Love = Mathf.Min(maxLove, Love + 1);
                AddReward(kissReward);
                optionSelectorAgent?.AddOptionReward(1, kissReward);
                prevJackDist = -1f;
            }
            else if (jackTarget != null)
            {
                if (prevJackDist > 0f)
                    AddReward((prevJackDist - currJackDist) * moveTowardsJackRewardScale);
                prevJackDist = currJackDist;
            }
            else
                prevJackDist = -1f;
            prevFlowerDist = -1f;
        }
        // Опция 2 (зомби): награда/штраф выдаются в OnBulletHitZombie / OnBulletHitJack из пули

        if (collectReady)
            _doCooldownRemaining = Mathf.Max(0f, collectActionCooldownSeconds);

        _lastCollectAction = collectAction;

        stepCount++;
        if (stepCount >= maxSteps)
        {
            optionSelectorAgent?.EndEpisode();
            EvalEpisodeTracker.NotifyEpisodeEnded();
            EndEpisode();
        }
    }

    private void SpawnBullet()
    {
        GameObject bullet = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        bullet.name = "LilyBullet";
        bullet.transform.position = transform.position + Vector3.up * 0.5f + transform.forward * 0.5f;
        bullet.transform.localScale = Vector3.one * 0.2f;

        var mat = bullet.GetComponent<Renderer>()?.material;
        if (mat != null)
            mat.color = Color.black;

        var rb = bullet.AddComponent<Rigidbody>();
        rb.useGravity = false;
        var col = bullet.GetComponent<Collider>();
        if (col != null)
            col.isTrigger = true;

        var bulletScript = bullet.AddComponent<LilyBullet>();
        var jackAgent = jackTarget != null ? jackTarget.GetComponent<AgentGoToHouseDiscrete>() : null;
        bulletScript.Init(transform.forward, jackTarget, jackAgent, bulletDamage, bulletSpeed, this, zombieLayer, jackLayer);
    }

    private static float HarvestReachDistance(Vector3 from, Collider c)
    {
        return Vector3.Distance(from, c.ClosestPoint(from));
    }

    private GameObject GetFlowerInstanceRoot(Collider hit)
    {
        Transform t = hit.transform;
        if (flowerSpawner != null)
        {
            Transform sp = flowerSpawner.transform;
            for (; t != null; t = t.parent)
            {
                if (t.parent == sp)
                    return t.gameObject;
            }
        }
        return hit.gameObject;
    }

    private bool TryCollectFlower()
    {
        Vector3 origin = transform.position;
        Collider[] hits = Physics.OverlapSphere(origin, collectDistance, flowerLayer);

        GameObject bestRoot = null;
        float bestDist = float.MaxValue;
        foreach (var c in hits)
        {
            if (c == null || !c.gameObject.activeInHierarchy) continue;
            float d = HarvestReachDistance(origin, c);
            if (d > collectDistance) continue;
            GameObject root = GetFlowerInstanceRoot(c);
            if (d < bestDist)
            {
                bestDist = d;
                bestRoot = root;
            }
        }

        if (bestRoot == null)
            return false;

        Destroy(bestRoot);
        return true;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var d = actionsOut.DiscreteActions;
        int moveAction = 1;
        int rotateAction = 1;
        int collectAction = Input.GetMouseButton(0) ? 1 : 0;  // ЛКМ — действие DO (собрать/поцеловать)
        int shootAction = Input.GetMouseButton(1) ? 1 : 0;     // ПКМ — выстрел

        // Только стрелки управляют движением (при любой опции; без авто-движения к Джеку)
        if (Input.GetKey(KeyCode.UpArrow)) moveAction = 2;
        else if (Input.GetKey(KeyCode.DownArrow)) moveAction = 0;
        if (Input.GetKey(KeyCode.RightArrow)) rotateAction = 2;
        else if (Input.GetKey(KeyCode.LeftArrow)) rotateAction = 0;

        d[0] = moveAction;
        d[1] = rotateAction;
        d[2] = collectAction;
        d[3] = shootAction;
    }
}
