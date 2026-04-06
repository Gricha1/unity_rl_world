using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

[RequireComponent(typeof(CharacterController))]
[RequireComponent(typeof(Animator))]
public class LilyScript : Agent, IHasHp
{
    /// <summary>0 = цветы, 1 = поцелуй Джека.</summary>
    private int currentOption;

    [Header("Option Sampling (Flowers/Kiss)")]
    [Tooltip("Если true, опция (цветы/поцелуй) выбирается по utility+softmax sampling каждые 20 шагов.")]
    [SerializeField] private bool useUtilitySoftmaxSampling = false;
    [Tooltip("Температура softmax (0.2 = почти жёстко, 0.7 = заметно случайно).")]
    [SerializeField] [Range(0.05f, 2.0f)] private float tau = 0.2f;
    [Tooltip("Шум eps ~ Uniform[-noise, +noise], добавляется в utility.")]
    [SerializeField] [Range(0f, 2f)] private float noise = 0.15f;
    [Tooltip("Макс. дистанция для access (в метрах). Ближе = 1, дальше = 0.")]
    [SerializeField] [Range(1f, 100f)] private float accessMaxDistance = 20f;
    [Tooltip("Липкость: насколько выгодно не переключаться без причины (добавка к utility текущей опции).")]
    [SerializeField] [Range(0f, 2f)] private float stickinessBonus = 0.5f;

    [Header("Current Option Icon")]
    [SerializeField] private SpriteRenderer optionIconRenderer;
    [SerializeField] private Sprite optionFlowerSprite;
    [SerializeField] private Sprite optionKissSprite;
    [SerializeField] private Vector3 optionIconOffset = new Vector3(0f, 2.2f, 0f);
    [SerializeField] private float optionFlowerIconScale = 0.45f;
    [SerializeField] private float optionKissIconScale = 0.45f;
    [SerializeField] private int optionIconSortingOrder = 100;
    [SerializeField] private bool optionIconFaceCamera = true;
    [Tooltip("Если задано — иконка разворачивается к этой камере; иначе MainCamera или камера с максимальным depth.")]
    [SerializeField] private Camera optionIconBillboardCamera;
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
    [Tooltip("Сглаживание параметра Speed в Animator (0 = без сглаживания).")]
    [SerializeField] private float walkAnimSpeedDamp = 0f;
    private int _lastCollectAction;
    private float _doCooldownRemaining;
    private float _lastPlanarMoveInput;

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

    [Header("Zombie (только наблюдения / окружение)")]
    [SerializeField] private LayerMask zombieLayer;

    [Header("HP")]
    [SerializeField] private int maxHp = 100;
    public int hp { get; private set; }

    [Header("Curriculum (последовательное обучение)")]
    [Tooltip("Включить для первой стадии: у Lily остаются только опции 0 (цветы) и 1 (поцелуй).")]
    [SerializeField] private bool curriculumNoShootNoZombie = false;

    [Header("Episode")]
    [SerializeField] private int maxSteps = 1500;

    /// <summary>True — режим curriculum: у Lily только опции 0/1.</summary>
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
        if (option < 0 || option > 1)
            return;
        currentOption = option;
        UpdateOptionIconVisual();
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
        if (optionFlowerSprite == null && optionKissSprite == null) return;

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
            _ => 0.35f
        };
        optionIconRenderer.transform.localScale = Vector3.one * s;
    }

    private void LateUpdate()
    {
        ApplyWalkAnimatorSpeed();

        if (!showOptionTaskIcon || optionIconRenderer == null) return;

        Vector3 iconPos = transform.position + optionIconOffset;
        optionIconRenderer.transform.position = iconPos;

        if (optionIconFaceCamera)
        {
            var cam = BillboardIconCamera.Resolve(iconPos, optionIconBillboardCamera);
            if (cam != null)
            {
                Vector3 toCam = cam.transform.position - iconPos;
                if (toCam.sqrMagnitude > 1e-6f)
                    optionIconRenderer.transform.rotation = Quaternion.LookRotation(toCam.normalized, cam.transform.up);
            }
        }
    }

    private void ApplyWalkAnimatorSpeed()
    {
        if (animator == null || controller == null) return;

        Vector3 v = controller.velocity;
        v.y = 0f;
        float velNorm = moveSpeed > 1e-4f ? Mathf.Clamp01(v.magnitude / moveSpeed) : 0f;
        float target = Mathf.Max(Mathf.Abs(_lastPlanarMoveInput), velNorm);
        if (target < 0.02f)
            target = 0f;

        if (walkAnimSpeedDamp > 0f)
            animator.SetFloat("Speed", target, walkAnimSpeedDamp, Time.deltaTime);
        else
            animator.SetFloat("Speed", target);
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
            int next = (currentOption + 1) % 2;
            SetOption(next);
        }
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

    public override void OnEpisodeBegin()
    {
        stepCount = 0;
        prevFlowerDist = -1f;
        prevJackDist = -1f;
        FlowerCount = 0;
        Love = 0;
        flowerDecayTimer = 0f;
        loveDecayTimer = 0f;
        hp = maxHp;
        _lastCollectAction = 0;
        _doCooldownRemaining = 0f;

        // Опция: либо utility+softmax, либо случайно
        if (useUtilitySoftmaxSampling)
            currentOption = SampleOptionUtilitySoftmax(currentOption);
        else
            currentOption = Random.Range(0, 2);

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

    // Опции "зомби" у Lily нет.

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

        // One-hot опции: 0 = цветы, 1 = поцелуй Джека
        sensor.AddObservation(currentOption == 0 ? 1f : 0f);
        sensor.AddObservation(currentOption == 1 ? 1f : 0f);

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
        // Utility sampling: пересэмпливаем каждые 20 шагов (если включено)
        if (useUtilitySoftmaxSampling && stepCount > 0 && (stepCount % 20) == 0)
        {
            currentOption = SampleOptionUtilitySoftmax(currentOption);
            UpdateOptionIconVisual();
        }

        int moveAction = actions.DiscreteActions[0];
        int rotateAction = actions.DiscreteActions[1];
        int collectAction = actions.DiscreteActions[2];

        bool collectJustPressed = collectAction == 1 && _lastCollectAction != 1;
        // DO актуален для обеих опций: 0 (цветы), 1 (поцелуй)
        bool collectDoRelevant = currentOption == 0 || currentOption == 1;
        bool collectReady = collectJustPressed && _doCooldownRemaining <= 0f && collectDoRelevant;

        if (collectReady && animator != null && doActionAnimTrigger.Length > 0)
            animator.SetTrigger(doActionAnimTrigger);

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

        _lastPlanarMoveInput = moveInput;

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
        if (collectReady)
            _doCooldownRemaining = Mathf.Max(0f, collectActionCooldownSeconds);

        _lastCollectAction = collectAction;

        stepCount++;
        if (stepCount >= maxSteps)
        {
            EvalEpisodeTracker.NotifyEpisodeEnded();
            EndEpisode();
        }
    }

    private int SampleOptionUtilitySoftmax(int currentOpt)
    {
        // need: чем меньше прогресс по "цветам/любви", тем выше потребность
        float flowerRatio = maxFlowerCount > 0 ? (float)FlowerCount / maxFlowerCount : 0f;
        float loveRatio = maxLove > 0 ? (float)Love / maxLove : 0f;
        float needFlowers = Mathf.Clamp01(1f - flowerRatio);
        float needKiss = Mathf.Clamp01(1f - loveRatio);

        // access: ближе цель -> больше access
        float distFlower = GetNearestFlower(out _, out float dF) ? dF : 999f;
        float distJack = GetDistanceToJack(out _);
        float accessFlowers = DistanceToAccess(distFlower);
        float accessKiss = DistanceToAccess(distJack);

        float stickFlowers = currentOpt == 0 ? 1f : 0f;
        float stickKiss = currentOpt == 1 ? 1f : 0f;

        float eps0 = Random.Range(-noise, noise);
        float eps1 = Random.Range(-noise, noise);

        float u0 = 2.5f * needFlowers + 1.0f * accessFlowers + stickinessBonus * stickFlowers + eps0;
        float u1 = 2.5f * needKiss + 1.0f * accessKiss + stickinessBonus * stickKiss + eps1;

        return SoftmaxSample2(u0, u1, Mathf.Max(0.0001f, tau));
    }

    private float DistanceToAccess(float distance)
    {
        if (float.IsNaN(distance) || float.IsInfinity(distance)) return 0f;
        if (accessMaxDistance <= 0.0001f) return 0f;
        float t = Mathf.Clamp01(distance / accessMaxDistance);
        return 1f - t;
    }

    private static int SoftmaxSample2(float u0, float u1, float temperature)
    {
        float a0 = u0 / temperature;
        float a1 = u1 / temperature;
        float m = Mathf.Max(a0, a1);
        float e0 = Mathf.Exp(a0 - m);
        float e1 = Mathf.Exp(a1 - m);
        float p0 = e0 / (e0 + e1);
        return Random.value < p0 ? 0 : 1;
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
        int collectAction = Input.GetMouseButton(1) ? 1 : 0;  // ПКМ — действие DO (собрать/поцеловать)

        // Только стрелки управляют движением (при любой опции; без авто-движения к Джеку)
        if (Input.GetKey(KeyCode.UpArrow)) moveAction = 2;
        else if (Input.GetKey(KeyCode.DownArrow)) moveAction = 0;
        if (Input.GetKey(KeyCode.RightArrow)) rotateAction = 2;
        else if (Input.GetKey(KeyCode.LeftArrow)) rotateAction = 0;

        d[0] = moveAction;
        d[1] = rotateAction;
        d[2] = collectAction;
    }
}
