using System.Reflection;
using System.Runtime.CompilerServices;
using VRCFaceTracking;
using VRCFaceTracking.Core.Params.Data;

namespace PalBuddyGuy.VRCFT;

/// <summary>
/// The wrapped module's private tracking data.
///
/// The wrapped (SRanipal) module gets its own copy of VRCFaceTracking.Core, so its writes to
/// UnifiedTracking.Data land in a private object instead of the one VRCFT reads. Modules
/// typically write their values and then sleep inside Update(); if both wrote to the shared
/// object, VRCFT would read SRanipal's value for most of each cycle and our overrides would
/// only flash briefly. Instead, after every inner update the eye/head data and the shapes
/// (with overrides merged per shape) are written to the shared object once.
/// </summary>
public sealed class InnerData
{
    public const float Unset = 0xFFFFFFFF;  // VRCFT's "module didn't provide this" marker

    private readonly FieldInfo _dataField;
    private readonly Dictionary<(Type, string), FieldInfo?> _fieldCache = new();

    public InnerData(Type privateUnifiedTracking)
    {
        _dataField = privateUnifiedTracking.GetField("Data", BindingFlags.Public | BindingFlags.Static)
                     ?? throw new MissingFieldException("UnifiedTracking.Data");
    }

    private object Data => _dataField.GetValue(null)!;

    private FieldInfo? Field(Type type, string name)
    {
        if (!_fieldCache.TryGetValue((type, name), out var f))
        {
            f = type.GetField(name, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            _fieldCache[(type, name)] = f;
        }
        return f;
    }

    private object? Get(object obj, string name) => Field(obj.GetType(), name)?.GetValue(obj);

    /// <summary>Weights of the private shapes, as floats (UnifiedExpressionShape is a single float).</summary>
    public float[] Shapes
    {
        get
        {
            var shapes = (Array)Get(Data, "Shapes")!;
            return Unsafe.As<float[]>(shapes);
        }
    }

    /// <summary>Mark everything as unset, like VRCFT's module process does for the shared data.</summary>
    public void InitialiseUnset()
    {
        var shapes = Shapes;
        Array.Fill(shapes, Unset);
        var eye = Get(Data, "Eye")!;
        foreach (var side in new[] { "Left", "Right" })
        {
            var f = Field(eye.GetType(), side)!;
            var single = f.GetValue(eye)!;  // boxed struct
            SetFloat(single, "Openness", Unset);
            SetFloat(single, "PupilDiameter_MM", Unset);
            var gazeField = Field(single.GetType(), "Gaze")!;
            var gaze = gazeField.GetValue(single)!;
            SetFloat(gaze, "x", Unset);
            SetFloat(gaze, "y", Unset);
            gazeField.SetValue(single, gaze);
            f.SetValue(eye, single);
        }
        SetFloat(eye, "_maxDilation", Unset);
        SetFloat(eye, "_minDilation", Unset);
    }

    private void SetFloat(object target, string name, float value) => Field(target.GetType(), name)?.SetValue(target, value);

    /// <summary>Copy eye and head data to the shared (VRCFT-visible) object.</summary>
    public void CopyEyeAndHead(UnifiedTrackingData shared)
    {
        var data = Data;
        var eye = Get(data, "Eye")!;
        shared.Eye.Left = CopyStruct<UnifiedSingleEyeData>(Get(eye, "Left")!);
        shared.Eye.Right = CopyStruct<UnifiedSingleEyeData>(Get(eye, "Right")!);
        shared.Eye._maxDilation = (float)Get(eye, "_maxDilation")!;
        shared.Eye._minDilation = (float)Get(eye, "_minDilation")!;
        shared.Eye._leftDiameter = (float)Get(eye, "_leftDiameter")!;
        shared.Eye._rightDiameter = (float)Get(eye, "_rightDiameter")!;
        shared.Head = CopyStruct<UnifiedHeadData>(Get(data, "Head")!);
    }

    /// <summary>Field-by-field copy between same-shaped structs from two copies of VRCFT.Core.</summary>
    private T CopyStruct<T>(object source) where T : struct => (T)CopyStruct(source, typeof(T));

    private object CopyStruct(object source, Type targetType)
    {
        var target = Activator.CreateInstance(targetType)!;  // boxed
        foreach (var f in targetType.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
        {
            var sf = Field(source.GetType(), f.Name);
            if (sf == null) continue;
            var value = sf.GetValue(source);
            if (value == null) continue;
            f.SetValue(target, f.FieldType.IsPrimitive || f.FieldType.IsEnum ? value : CopyStruct(value, f.FieldType));
        }
        return target;
    }

    public static bool IsUnset(float v) => v == Unset || float.IsNaN(v);
}
