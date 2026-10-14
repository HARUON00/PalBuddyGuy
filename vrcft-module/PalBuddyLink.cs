using System.Net.Sockets;
using System.Text;
using Microsoft.Extensions.Logging;
using VRCFaceTracking.Core.Params.Data;
using VRCFaceTracking.Core.Params.Expressions;

namespace PalBuddyGuy.VRCFT;

/// <summary>
/// TCP link to the Pal Buddy Guy app (the app listens, this module connects and reconnects).
///
/// Protocol 2 (little framing, all multi-byte values big endian):
///   module -> app : "PBG2" version:u8                     hello, sent on connect
///   app -> module : 0x05 mode:u8 count:u8 {len:u8 name}*   target table: Unified Expression names
///                                                          mode 0 = replace, 1 = max(base, ours)
///   app -> module : 0x03 count:u8 {slot:u8 weight:u16}*    weights (0..65535 = 0..1) for table slots;
///                                                          count 0 = tracking stopped, clear overrides
/// Overrides older than <see cref="StaleAfter"/> are dropped, so a frozen or closed app
/// never leaves a shape stuck.
/// </summary>
public sealed class PalBuddyLink : IDisposable
{
    public const byte Version = 2;
    public static readonly TimeSpan StaleAfter = TimeSpan.FromMilliseconds(500);

    private readonly string _host;
    private readonly int _port;
    private readonly ILogger _logger;
    private readonly Thread _thread;
    private volatile bool _stop;
    private TcpClient? _client;

    private readonly object _lock = new();
    private int[] _slotToShape = [];            // table slot -> UnifiedExpressions index (-1 = unknown name)
    private readonly float[] _weights = new float[(int)UnifiedExpressions.Max];
    private readonly bool[] _active = new bool[(int)UnifiedExpressions.Max];
    private readonly bool[] _touched = new bool[(int)UnifiedExpressions.Max];  // shapes we must reset on clear
    private bool _maxMode;
    private long _lastValuesTicks;

    public bool Connected { get; private set; }
    public long PacketsReceived { get; private set; }

    public PalBuddyLink(string host, int port, ILogger logger)
    {
        _host = host;
        _port = port;
        _logger = logger;
        _thread = new Thread(Run) { IsBackground = true, Name = "PalBuddyGuy link" };
    }

    public void Start() => _thread.Start();

    private void Run()
    {
        var backoff = 250;
        var loggedFailure = false;
        while (!_stop)
        {
            try
            {
                using var client = new TcpClient { NoDelay = true };
                var connect = client.ConnectAsync(_host, _port);
                if (!connect.Wait(TimeSpan.FromSeconds(2)) || !client.Connected)
                {
                    throw new SocketException((int)SocketError.TimedOut);
                }
                _client = client;
                var stream = client.GetStream();
                stream.Write([(byte)'P', (byte)'B', (byte)'G', (byte)'2', Version]);
                Connected = true;
                loggedFailure = false;
                backoff = 250;
                _logger.LogInformation("Connected to Pal Buddy Guy at {host}:{port}", _host, _port);
                ReadLoop(stream);
            }
            catch (Exception e) when (e is SocketException or IOException or AggregateException or ObjectDisposedException)
            {
                if (!loggedFailure && !_stop)
                {
                    _logger.LogInformation("Pal Buddy Guy app not reachable on {host}:{port} ({msg}); retrying in the background",
                        _host, _port, e.GetBaseException().Message);
                    loggedFailure = true;
                }
            }
            finally
            {
                if (Connected)
                {
                    _logger.LogInformation("Disconnected from Pal Buddy Guy");
                }
                Connected = false;
                _client = null;
                Clear();
            }
            if (!_stop)
            {
                Thread.Sleep(backoff);
                backoff = Math.Min(backoff * 2, 3000);
            }
        }
    }

    private static byte ReadByte(Stream s)
    {
        var b = s.ReadByte();
        if (b < 0) throw new IOException("connection closed");
        return (byte)b;
    }

    private void ReadLoop(NetworkStream stream)
    {
        var buf = new byte[256 * 3];
        while (!_stop)
        {
            var type = ReadByte(stream);
            switch (type)
            {
                case 0x05:
                {
                    var mode = ReadByte(stream);
                    var count = ReadByte(stream);
                    var table = new int[count];
                    for (var i = 0; i < count; i++)
                    {
                        var len = ReadByte(stream);
                        stream.ReadExactly(buf, 0, len);
                        var name = Encoding.ASCII.GetString(buf, 0, len);
                        if (Enum.TryParse<UnifiedExpressions>(name, out var shape) && shape != UnifiedExpressions.Max)
                        {
                            table[i] = (int)shape;
                        }
                        else
                        {
                            table[i] = -1;
                            _logger.LogWarning("Pal Buddy Guy target '{name}' is not a Unified Expression in this VRCFT version", name);
                        }
                    }
                    lock (_lock)
                    {
                        ClearLocked();
                        _slotToShape = table;
                        _maxMode = mode == 1;
                    }
                    _logger.LogInformation("Pal Buddy Guy drives {count} shapes ({mode})", count, mode == 1 ? "max" : "replace");
                    break;
                }
                case 0x03:
                {
                    var count = ReadByte(stream);
                    stream.ReadExactly(buf, 0, count * 3);
                    lock (_lock)
                    {
                        if (count == 0)
                        {
                            ClearLocked();
                            break;
                        }
                        Array.Clear(_active);
                        for (var i = 0; i < count; i++)
                        {
                            var slot = buf[i * 3];
                            if (slot >= _slotToShape.Length || _slotToShape[slot] < 0) continue;
                            var shape = _slotToShape[slot];
                            var weight = ((buf[i * 3 + 1] << 8) | buf[i * 3 + 2]) / 65535f;
                            // several slots may map to the same shape: keep the strongest
                            _weights[shape] = _active[shape] ? Math.Max(_weights[shape], weight) : weight;
                            _active[shape] = true;
                            _touched[shape] = true;
                        }
                        _lastValuesTicks = Environment.TickCount64;
                    }
                    PacketsReceived++;
                    break;
                }
                default:
                    throw new IOException($"unknown message type {type} from Pal Buddy Guy");
            }
        }
    }

    private void Clear()
    {
        lock (_lock) ClearLocked();
    }

    private void ClearLocked()
    {
        Array.Clear(_active);
        _lastValuesTicks = 0;
    }

    /// <summary>Write shapes to <paramref name="output"/>: our override where we have a fresh
    /// one, otherwise the wrapped module's value (<paramref name="baseWeights"/>, may be null).
    /// Each output shape is written once, so VRCFT never sees an intermediate value. A shape we
    /// stop overriding that the wrapped module doesn't provide is set to 0 once (an unset value
    /// would make VRCFT keep our last one).</summary>
    public void Compose(float[]? baseWeights, UnifiedExpressionShape[] output)
    {
        lock (_lock)
        {
            var fresh = _lastValuesTicks != 0 && Environment.TickCount64 - _lastValuesTicks < StaleAfter.TotalMilliseconds;
            var n = Math.Min(output.Length, _weights.Length);
            for (var i = 0; i < n; i++)
            {
                var baseWeight = baseWeights != null && i < baseWeights.Length ? baseWeights[i] : InnerData.Unset;
                if (fresh && _active[i])
                {
                    var b = InnerData.IsUnset(baseWeight) ? 0f : baseWeight;
                    output[i].Weight = _maxMode ? Math.Max(b, _weights[i]) : _weights[i];
                }
                else if (_touched[i])
                {
                    output[i].Weight = InnerData.IsUnset(baseWeight) ? 0f : baseWeight;
                    _touched[i] = false;
                }
                else if (baseWeights != null)
                {
                    output[i].Weight = baseWeight;
                }
            }
        }
    }

    public void Dispose()
    {
        _stop = true;
        try { _client?.Close(); } catch { /* ignore */ }
        _thread.Join(TimeSpan.FromSeconds(3));
    }
}
