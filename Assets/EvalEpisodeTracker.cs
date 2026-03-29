using System;
using System.Threading;

/// <summary>
/// Minimal cross-agent episode counter for evaluation runs.
/// Agents call NotifyEpisodeEnded() right before EndEpisode().
/// </summary>
public static class EvalEpisodeTracker
{
    private static int _endedEpisodes;

    public static int EndedEpisodes => _endedEpisodes;

    public static void Reset() => Interlocked.Exchange(ref _endedEpisodes, 0);

    public static void NotifyEpisodeEnded()
    {
        Interlocked.Increment(ref _endedEpisodes);
    }
}

