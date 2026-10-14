// Prints a JSON object {"base": [...], "binary": [...]} with the parameter names of every
// parameter in UnifiedTracking.AllParameters. VRCFaceTracking drives an avatar parameter whose
// address equals a base name or ends with "/<name>"; binary parameters additionally match
// "<name><digits>" (bools) and "<name>Negative".
using System.Collections;
using System.Reflection;
using System.Text.Json;
using VRCFaceTracking;
using VRCFaceTracking.Core.OSC.DataTypes;
using VRCFaceTracking.Core.Params;

var baseNames = new SortedSet<string>(StringComparer.Ordinal);
var binaryNames = new SortedSet<string>(StringComparer.Ordinal);
var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);

IEnumerable<FieldInfo> AllFields(Type t)
{
    for (var c = t; c != null && c != typeof(object); c = c.BaseType)
        foreach (var f in c.GetFields(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.DeclaredOnly))
            yield return f;
}

void Walk(object? p)
{
    if (p == null || !seen.Add(p)) return;
    var t = p.GetType();
    var nameField = AllFields(t).FirstOrDefault(f => f.Name == "_paramName" && f.FieldType == typeof(string));
    if (nameField != null)
    {
        var name = (string)nameField.GetValue(p)!;
        (p is BinaryBaseParameter ? binaryNames : baseNames).Add(name);
    }
    foreach (var f in AllFields(t))
    {
        var v = f.GetValue(p);
        if (v is Parameter child) Walk(child);
        else if (v is IEnumerable seq and not string)
            foreach (var item in seq) if (item is Parameter c2) Walk(c2);
    }
}
foreach (var p in UnifiedTracking.AllParameters) Walk(p);
Console.WriteLine(JsonSerializer.Serialize(new { @base = baseNames, binary = binaryNames }));
