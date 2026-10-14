using Microsoft.Extensions.Logging;
using System.Text.Json;
using VRCFaceTracking;

namespace PalBuddyGuy.VRCFT;

/// <summary>
/// VRCFaceTracking v6 module: SRanipal tracking (via the wrapped SRanipal module) with the
/// shapes trained in Pal Buddy Guy laid on top.
///
/// With the SRanipal module available: provides eye + expression tracking and can override
/// any Unified Expression, including eye-area shapes like brows.
/// Without it: provides expressions only (Pal Buddy Guy's shapes), e.g. next to another
/// eye-tracking module when no facial tracker is present.
///
/// Settings are read from palbuddyguy.json next to this DLL (written by the app):
///   { "host": "127.0.0.1", "port": 26421, "innerModule": "C:\\...\\SRanipal.dll" }
/// </summary>
public class PalBuddyGuyModule : ExtTrackingModule
{
    private ExtTrackingModule? _inner;
    private InnerData? _innerData;
    private PalBuddyLink? _link;
    private bool _innerEye, _innerExpression;

    public override (bool SupportsEye, bool SupportsExpression) Supported => (true, true);

    internal static string OwnDirectory => Path.GetDirectoryName(typeof(PalBuddyGuyModule).Assembly.Location) ?? ".";

    public override (bool eyeSuccess, bool expressionSuccess) Initialize(bool eyeAvailable, bool expressionAvailable)
    {
        var settings = LoadSettings();
        var host = settings.TryGetValue("host", out var h) && h.ValueKind == JsonValueKind.String ? h.GetString()! : "127.0.0.1";
        var port = settings.TryGetValue("port", out var p) && p.TryGetInt32(out var pv) ? pv : 26421;
        var innerPath = settings.TryGetValue("innerModule", out var m) && m.ValueKind == JsonValueKind.String ? m.GetString() : null;
        if (string.IsNullOrWhiteSpace(innerPath) || !File.Exists(innerPath))
        {
            var customLibs = Directory.GetParent(OwnDirectory)?.FullName ?? OwnDirectory;
            innerPath = InnerModule.FindSRanipalModule(customLibs, OwnDirectory, Logger);
        }

        if (innerPath != null && InnerModule.Load(innerPath, Logger) is var (module, data))
        {
            _inner = module;
            _innerData = data;
        }
        if (_inner != null)
        {
            try
            {
                (_innerEye, _innerExpression) = _inner.Initialize(eyeAvailable, expressionAvailable);
                Logger.LogInformation("SRanipal: eye {eye}, expression {expr}", _innerEye, _innerExpression);
            }
            catch (Exception e)
            {
                Logger.LogError("SRanipal module failed to initialise: {e}", e);
                _inner = null;
            }
        }
        else
        {
            Logger.LogWarning("SRanipal module not found: Pal Buddy Guy will only provide its own expression shapes. " +
                              "Install the SRanipal module in VRCFaceTracking, then run Install again in the Pal Buddy Guy app.");
        }

        _link = new PalBuddyLink(host, port, Logger);
        _link.Start();

        var info = ModuleInformation;
        info.Name = _inner != null ? "PalBuddyGuy + SRanipal" : "PalBuddyGuy";
        if (_inner?.ModuleInformation.StaticImages is { Count: > 0 } images)
        {
            info.StaticImages = images;
        }
        ModuleInformation = info;

        // Eye tracking only comes from SRanipal. Expressions: SRanipal's (if any) plus ours;
        // without a facial tracker we still claim expressions so our shapes reach VRChat.
        return (_innerEye, expressionAvailable);
    }

    private Dictionary<string, JsonElement> LoadSettings()
    {
        var path = Path.Combine(OwnDirectory, "palbuddyguy.json");
        try
        {
            if (File.Exists(path))
            {
                return JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(File.ReadAllText(path)) ?? new();
            }
        }
        catch (Exception e)
        {
            Logger.LogWarning("Ignoring invalid {path}: {msg}", path, e.Message);
        }
        return new();
    }

    public override void Update()
    {
        var inner = _inner;
        if (inner != null)
        {
            // keep the wrapped module's view of its state in sync with what VRCFT told us
            inner.Status = Status;
            var info = inner.ModuleInformation;
            info.UsingEye = ModuleInformation.UsingEye;
            info.UsingExpression = ModuleInformation.UsingExpression;
            inner.ModuleInformation = info;
            try
            {
                inner.Update();  // writes the private data (and paces the loop)
            }
            catch (Exception e)
            {
                Logger.LogError("SRanipal module update failed: {msg}", e.Message);
                Thread.Sleep(100);
            }
            _innerData!.CopyEyeAndHead(UnifiedTracking.Data);
            _link?.Compose(_innerData.Shapes, UnifiedTracking.Data.Shapes);
        }
        else
        {
            Thread.Sleep(10);  // ~100 Hz, like VRCFT's update requests
            _link?.Compose(null, UnifiedTracking.Data.Shapes);
        }
    }

    public override void Teardown()
    {
        _link?.Dispose();
        _link = null;
        try
        {
            _inner?.Teardown();
        }
        catch (Exception e)
        {
            Logger.LogWarning("SRanipal module teardown failed: {msg}", e.Message);
        }
    }
}
