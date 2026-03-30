using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Unity.MLAgents;
using UnityEngine;

/// <summary>
/// Optional runtime-only helper for evaluation runs launched from CLI.
/// Enables frame capture (PNG sequence) and auto-quit after N Academy steps.
///
/// Usage (passed via ML-Agents --env-args):
///   --capture-dir <path>         Directory for PNG frames
///   --capture-dir-b <path>       Directory for 2nd camera PNG frames
///   --capture-every <n>          Capture every n rendered frames (default: 1)
///   --capture-width <w>          Camera capture width (default: 1920)
///   --capture-height <h>         Camera capture height (default: 1080)
///   --capture-camera-a <name>    Camera A name (optional)
///   --capture-camera-b <name>    Camera B name (optional)
///   --quit-after-steps <n>       Quit after n Academy steps (default: disabled)
///   --quit-after-episodes <n>    Quit after n completed episodes (requires agent integration)
///   --quit-after-seconds <s>     Quit after s real-time seconds (works even if Academy doesn't step)
///   --quit-delay-seconds <s>     Delay before quitting to allow final frame writes (default: 0.5)
/// </summary>
public sealed class CommandLineEvalCapture : MonoBehaviour
{
    private string _captureDir;
    private string _captureDirB;
    private int _captureEvery = 1;
    private int _captureWidth = 1920;
    private int _captureHeight = 1080;
    private string _cameraAName;
    private string _cameraBName;
    private long _quitAfterSteps = -1;
    private int _quitAfterEpisodes = -1;
    private float _quitAfterSeconds = -1f;
    private float _quitDelaySeconds = 0.5f;

    private int _frameIndexA;
    private int _frameIndexB;
    private int _renderFrameCounter;
    private long _startStep = -1;
    private bool _quitScheduled;
    private float _startRealtime = -1f;
    private Camera _cameraA;
    private Camera _cameraB;
    private Texture2D _captureTexture;
    private RenderTexture _captureRt;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Bootstrap()
    {
        try
        {
            var args = Environment.GetCommandLineArgs();
            if (args == null || args.Length == 0) return;

            // Only create the helper if any of the supported flags are present.
            for (var i = 0; i < args.Length; i++)
            {
                if (args[i] == "--capture-dir" || args[i] == "--capture-dir-b" || args[i] == "--quit-after-steps")
                {
                    var go = new GameObject(nameof(CommandLineEvalCapture));
                    DontDestroyOnLoad(go);
                    go.AddComponent<CommandLineEvalCapture>();
                    return;
                }
            }
        }
        catch
        {
            // Best-effort: never break a build due to eval utilities.
        }
    }

    private void Awake()
    {
        ParseArgs(Environment.GetCommandLineArgs());

        if (!string.IsNullOrWhiteSpace(_captureDir))
        {
            Directory.CreateDirectory(_captureDir);
            try
            {
                File.WriteAllText(Path.Combine(_captureDir, "capture_started.txt"), DateTime.UtcNow.ToString("O"));
            }
            catch
            {
                // ignore
            }
        }

        if (!string.IsNullOrWhiteSpace(_captureDirB))
        {
            Directory.CreateDirectory(_captureDirB);
            try
            {
                File.WriteAllText(Path.Combine(_captureDirB, "capture_started.txt"), DateTime.UtcNow.ToString("O"));
            }
            catch
            {
                // ignore
            }
        }

        InitCaptureResources();

        if (_quitAfterEpisodes > 0)
        {
            EvalEpisodeTracker.Reset();
        }
    }

    private void Update()
    {
        if (_startRealtime < 0f) _startRealtime = Time.realtimeSinceStartup;

        if (!string.IsNullOrWhiteSpace(_captureDir) || !string.IsNullOrWhiteSpace(_captureDirB))
        {
            _renderFrameCounter++;
            if (_renderFrameCounter % Math.Max(1, _captureEvery) == 0)
            {
                CaptureFrames();
            }
        }

        if (_quitAfterSteps > 0 && !_quitScheduled)
        {
            var currentStep = Academy.Instance.StepCount;
            if (_startStep < 0) _startStep = currentStep;

            if (currentStep - _startStep >= _quitAfterSteps)
            {
                _quitScheduled = true;
                Invoke(nameof(Quit), Mathf.Max(0f, _quitDelaySeconds));
            }
        }

        if (_quitAfterEpisodes > 0 && !_quitScheduled)
        {
            if (EvalEpisodeTracker.EndedEpisodes >= _quitAfterEpisodes)
            {
                _quitScheduled = true;
                Invoke(nameof(Quit), Mathf.Max(0f, _quitDelaySeconds));
            }
        }

        if (_quitAfterSeconds > 0f && !_quitScheduled)
        {
            if (Time.realtimeSinceStartup - _startRealtime >= _quitAfterSeconds)
            {
                _quitScheduled = true;
                Invoke(nameof(Quit), Mathf.Max(0f, _quitDelaySeconds));
            }
        }
    }

    private void Quit()
    {
        Application.Quit(0);
    }

    private void OnDestroy()
    {
        if (_captureRt != null)
        {
            _captureRt.Release();
            Destroy(_captureRt);
        }
        if (_captureTexture != null)
        {
            Destroy(_captureTexture);
        }
    }

    private void InitCaptureResources()
    {
        _captureWidth = Math.Max(64, _captureWidth);
        _captureHeight = Math.Max(64, _captureHeight);
        _captureRt = new RenderTexture(_captureWidth, _captureHeight, 24, RenderTextureFormat.ARGB32);
        _captureTexture = new Texture2D(_captureWidth, _captureHeight, TextureFormat.RGB24, false);
        ResolveCameras();
    }

    private void ResolveCameras()
    {
        var cameras = Camera.allCameras
            .Where(c => c != null && c.enabled)
            .OrderBy(c => c.depth)
            .ToArray();

        if (!string.IsNullOrWhiteSpace(_cameraAName))
            _cameraA = cameras.FirstOrDefault(c => c.name == _cameraAName);
        if (!string.IsNullOrWhiteSpace(_cameraBName))
            _cameraB = cameras.FirstOrDefault(c => c.name == _cameraBName);

        if (_cameraA == null)
            _cameraA = Camera.main ?? cameras.FirstOrDefault();

        if (_cameraB == null)
            _cameraB = cameras.FirstOrDefault(c => c != _cameraA);
    }

    private void CaptureFrames()
    {
        // Some cameras may spawn after scene load.
        if (_cameraA == null || (!string.IsNullOrWhiteSpace(_captureDirB) && _cameraB == null))
            ResolveCameras();

        if (!string.IsNullOrWhiteSpace(_captureDir))
        {
            if (_cameraA != null)
                CaptureCameraToPng(_cameraA, _captureDir, ref _frameIndexA);
            else
                CaptureScreenFallback(_captureDir, ref _frameIndexA);
        }

        if (!string.IsNullOrWhiteSpace(_captureDirB))
        {
            if (_cameraB != null)
                CaptureCameraToPng(_cameraB, _captureDirB, ref _frameIndexB);
        }
    }

    private void CaptureCameraToPng(Camera camera, string dir, ref int index)
    {
        var prevTarget = camera.targetTexture;
        var prevActive = RenderTexture.active;

        camera.targetTexture = _captureRt;
        camera.Render();
        RenderTexture.active = _captureRt;
        _captureTexture.ReadPixels(new Rect(0, 0, _captureWidth, _captureHeight), 0, 0);
        _captureTexture.Apply(false, false);

        camera.targetTexture = prevTarget;
        RenderTexture.active = prevActive;

        var bytes = _captureTexture.EncodeToPNG();
        var path = Path.Combine(dir, $"frame_{index:D06}.png");
        File.WriteAllBytes(path, bytes);
        index++;
    }

    private static void CaptureScreenFallback(string dir, ref int index)
    {
        var path = Path.Combine(dir, $"frame_{index:D06}.png");
        ScreenCapture.CaptureScreenshot(path, 1);
        index++;
    }

    private void ParseArgs(string[] args)
    {
        var dict = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var i = 0; i < args.Length; i++)
        {
            var key = args[i];
            if (!key.StartsWith("--", StringComparison.Ordinal)) continue;
            if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
            {
                dict[key] = args[i + 1];
                i++;
            }
            else
            {
                dict[key] = "true";
            }
        }

        if (dict.TryGetValue("--capture-dir", out var captureDir) && !string.IsNullOrWhiteSpace(captureDir))
        {
            _captureDir = captureDir;
        }

        if (dict.TryGetValue("--capture-dir-b", out var captureDirB) && !string.IsNullOrWhiteSpace(captureDirB))
        {
            _captureDirB = captureDirB;
        }

        if (dict.TryGetValue("--capture-every", out var captureEveryStr) && int.TryParse(captureEveryStr, out var captureEvery))
        {
            _captureEvery = Mathf.Max(1, captureEvery);
        }

        if (dict.TryGetValue("--capture-width", out var widthStr) && int.TryParse(widthStr, out var width))
        {
            _captureWidth = Math.Max(64, width);
        }

        if (dict.TryGetValue("--capture-height", out var heightStr) && int.TryParse(heightStr, out var height))
        {
            _captureHeight = Math.Max(64, height);
        }

        if (dict.TryGetValue("--capture-camera-a", out var cameraAName) && !string.IsNullOrWhiteSpace(cameraAName))
        {
            _cameraAName = cameraAName;
        }

        if (dict.TryGetValue("--capture-camera-b", out var cameraBName) && !string.IsNullOrWhiteSpace(cameraBName))
        {
            _cameraBName = cameraBName;
        }

        if (dict.TryGetValue("--quit-after-steps", out var quitAfterStepsStr) && long.TryParse(quitAfterStepsStr, out var quitAfterSteps))
        {
            _quitAfterSteps = Math.Max(1, quitAfterSteps);
        }

        if (dict.TryGetValue("--quit-after-episodes", out var quitAfterEpisodesStr) && int.TryParse(quitAfterEpisodesStr, out var quitAfterEpisodes))
        {
            _quitAfterEpisodes = Mathf.Max(1, quitAfterEpisodes);
        }

        if (dict.TryGetValue("--quit-after-seconds", out var quitAfterSecondsStr) && float.TryParse(quitAfterSecondsStr, out var quitAfterSeconds))
        {
            _quitAfterSeconds = Mathf.Max(0.1f, quitAfterSeconds);
        }

        if (dict.TryGetValue("--quit-delay-seconds", out var quitDelayStr) && float.TryParse(quitDelayStr, out var quitDelay))
        {
            _quitDelaySeconds = Mathf.Max(0f, quitDelay);
        }
    }
}
