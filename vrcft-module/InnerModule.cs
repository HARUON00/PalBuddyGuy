using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using Microsoft.Extensions.Logging;
using System.Text.Json;
using VRCFaceTracking;

namespace PalBuddyGuy.VRCFT;

/// <summary>
/// Loads another VRCFT module (normally the SRanipal module) into this module process.
///
/// VRCFaceTracking v6 gives eye tracking to exactly one module and expressions to exactly one
/// module, whichever initialises first. To add Pal Buddy Guy's shapes on top of SRanipal
/// deterministically, this module runs the SRanipal module itself and overrides shapes
/// after each of its updates; the standalone SRanipal module is disabled while
/// Pal Buddy Guy is installed (the app renames its module.json, which VRCFT then skips).
/// </summary>
public static class InnerModule
{
    public const string DisabledSuffix = ".palbuddyguy-disabled";

    /// <summary>Find an SRanipal module under VRCFT's CustomLibs folder (disabled ones first).</summary>
    public static string? FindSRanipalModule(string customLibs, string ownDirectory, ILogger logger)
    {
        if (!Directory.Exists(customLibs)) return null;
        string? enabled = null;
        foreach (var dir in Directory.GetDirectories(customLibs))
        {
            if (Path.GetFullPath(dir).TrimEnd(Path.DirectorySeparatorChar) ==
                Path.GetFullPath(ownDirectory).TrimEnd(Path.DirectorySeparatorChar)) continue;
            foreach (var (file, isDisabled) in new[] { (Path.Combine(dir, "module.json" + DisabledSuffix), true),
                                                        (Path.Combine(dir, "module.json"), false) })
            {
                if (!File.Exists(file)) continue;
                try
                {
                    using var json = JsonDocument.Parse(File.ReadAllText(file));
                    var name = Str(json.RootElement, "ModuleName");
                    var dll = Str(json.RootElement, "DllFileName");
                    if (!name.Contains("SRanipal", StringComparison.OrdinalIgnoreCase) &&
                        !dll.Contains("SRanipal", StringComparison.OrdinalIgnoreCase)) continue;
                    var path = Path.Combine(dir, dll);
                    if (!File.Exists(path)) continue;
                    if (isDisabled) return path;
                    enabled ??= path;
                }
                catch (Exception e)
                {
                    logger.LogDebug("Skipping {file}: {msg}", file, e.Message);
                }
            }
        }
        if (enabled != null)
        {
            logger.LogWarning("The SRanipal module is also installed and enabled. VRCFaceTracking will start both and " +
                              "only one can own eye/expression tracking. Disable it from the Pal Buddy Guy app " +
                              "(Settings > VRCFaceTracking module > Install).");
        }
        return enabled;
    }

    private static string Str(JsonElement e, string key) =>
        e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() ?? "" : "";

    /// <summary>Load the module with its own copy of VRCFaceTracking.Core (see <see cref="InnerData"/>).</summary>
    public static (ExtTrackingModule module, InnerData data)? Load(string dllPath, ILogger logger)
    {
        try
        {
            var context = new InnerLoadContext(dllPath);
            var privateCore = context.LoadFromAssemblyName(typeof(UnifiedTracking).Assembly.GetName());
            var privateUnified = privateCore.GetType(typeof(UnifiedTracking).FullName!, throwOnError: true)!;
            if (privateUnified == typeof(UnifiedTracking))
            {
                throw new InvalidOperationException("VRCFaceTracking.Core was not isolated");
            }
            var data = new InnerData(privateUnified);
            data.InitialiseUnset();
            var assembly = context.LoadFromAssemblyPath(Path.GetFullPath(dllPath));
            Type[] types;
            try
            {
                types = assembly.GetTypes();
            }
            catch (ReflectionTypeLoadException e)
            {
                types = e.Types.Where(t => t != null).ToArray()!;
            }
            var type = types.FirstOrDefault(t => t.IsSubclassOf(typeof(ExtTrackingModule)) && !t.IsAbstract);
            if (type == null)
            {
                logger.LogError("{dll} contains no VRCFaceTracking module", dllPath);
                return null;
            }
            var module = (ExtTrackingModule)Activator.CreateInstance(type)!;
            module.Logger = logger;
            logger.LogInformation("Loaded inner module {type} from {dll}", type.FullName, dllPath);
            return (module, data);
        }
        catch (Exception e)
        {
            logger.LogError("Could not load inner module {dll}: {e}", dllPath, e);
            return null;
        }
    }

    /// <summary>Resolves the inner module's own dependencies from its folder and gives it a
    /// private VRCFaceTracking.Core (so its UnifiedTracking.Data is separate), while sharing
    /// VRCFaceTracking.SDK (ExtTrackingModule) and framework assemblies (ILogger) with the host.</summary>
    private sealed class InnerLoadContext(string mainAssemblyPath)
        : AssemblyLoadContext("PalBuddyGuy inner module", isCollectible: false)
    {
        private readonly string _dir = Path.GetDirectoryName(Path.GetFullPath(mainAssemblyPath))!;

        private static readonly string CorePath = typeof(UnifiedTracking).Assembly.Location;
        private static readonly string CoreName = typeof(UnifiedTracking).Assembly.GetName().Name!;

        private static bool IsShared(string name) =>
            name.StartsWith("VRCFaceTracking", StringComparison.OrdinalIgnoreCase) ||
            name.StartsWith("Microsoft.Extensions.", StringComparison.OrdinalIgnoreCase) ||
            name.StartsWith("System.", StringComparison.OrdinalIgnoreCase) ||
            name.Equals("Newtonsoft.Json", StringComparison.OrdinalIgnoreCase) ||
            name.Equals("netstandard", StringComparison.OrdinalIgnoreCase) ||
            name.Equals("mscorlib", StringComparison.OrdinalIgnoreCase);

        protected override Assembly? Load(AssemblyName assemblyName)
        {
            var name = assemblyName.Name ?? "";
            if (name == CoreName) return LoadFromAssemblyPath(CorePath);  // private copy
            if (IsShared(name)) return null;  // default context
            var path = Path.Combine(_dir, name + ".dll");
            return File.Exists(path) ? LoadFromAssemblyPath(path) : null;
        }

        protected override IntPtr LoadUnmanagedDll(string unmanagedDllName)
        {
            foreach (var candidate in new[] { unmanagedDllName, unmanagedDllName + ".dll",
                                              "lib" + unmanagedDllName + ".so", unmanagedDllName + ".so" })
            {
                var path = Path.Combine(_dir, candidate);
                if (File.Exists(path) && NativeLibrary.TryLoad(path, out var handle)) return handle;
            }
            return IntPtr.Zero;  // default probing
        }
    }
}
