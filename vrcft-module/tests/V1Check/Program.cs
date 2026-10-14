// stdin: one JSON object per line: {"id": "...", "shapes": {"UnifiedName": weight, ...}}
// stdout: one JSON object per line: {"id": "...", "v1": {"ParamName": value, ...}}
// Values come from the float parameter functions in UnifiedTracking.AllParameters_v1, i.e.
// exactly what VRCFaceTracking would send for v1 avatar parameters.
using System.Reflection;
using System.Text.Json;
using VRCFaceTracking;
using VRCFaceTracking.Core.OSC.DataTypes;
using VRCFaceTracking.Core.Params.Data;
using VRCFaceTracking.Core.Params.Expressions;

var floatParams = new List<(string name, Func<UnifiedTrackingData, float> func)>();
void Collect(object p)
{
    var t = p.GetType();
    if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(BaseParam<>) && t.GetGenericArguments()[0] == typeof(float))
    {
        var name = (string)t.GetField("_paramName", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(p)!;
        var func = (Func<UnifiedTrackingData, float>)t.GetField("_getValueFunc", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(p)!;
        if (floatParams.All(f => f.name != name)) floatParams.Add((name, func));
        return;
    }
    var inner = t.GetField("_parameter", BindingFlags.NonPublic | BindingFlags.Instance);
    if (inner?.GetValue(p) is Array children)
        foreach (var c in children) Collect(c!);
}
foreach (var p in UnifiedTracking.AllParameters_v1) Collect(p);

string? line;
while ((line = Console.ReadLine()) != null)
{
    if (string.IsNullOrWhiteSpace(line)) continue;
    using var doc = JsonDocument.Parse(line);
    var data = new UnifiedTrackingData();
    foreach (var s in doc.RootElement.GetProperty("shapes").EnumerateObject())
        data.Shapes[(int)Enum.Parse<UnifiedExpressions>(s.Name)].Weight = s.Value.GetSingle();
    var result = new Dictionary<string, float>();
    foreach (var (name, func) in floatParams)
    {
        var v = func(data);
        result[name] = float.IsFinite(v) ? v : 0f;
    }
    Console.WriteLine(JsonSerializer.Serialize(new { id = doc.RootElement.GetProperty("id").GetString(), v1 = result }));
}
