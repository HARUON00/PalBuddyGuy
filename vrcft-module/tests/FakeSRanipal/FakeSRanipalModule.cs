using Microsoft.Extensions.Logging;
using VRCFaceTracking;
using VRCFaceTracking.Core.Params.Expressions;

namespace FakeSRanipal;

public class FakeSRanipalModule : ExtTrackingModule
{
    public override (bool SupportsEye, bool SupportsExpression) Supported => (true, true);

    public override (bool eyeSuccess, bool expressionSuccess) Initialize(bool eyeAvailable, bool expressionAvailable)
    {
        Logger.LogInformation("FakeSRanipal initialised (eye {eye}, expr {expr})", eyeAvailable, expressionAvailable);
        return (eyeAvailable, expressionAvailable);
    }

    public override void Update()
    {
        var d = UnifiedTracking.Data;
        d.Eye.Left.Openness = 0.8f;
        d.Eye.Right.Openness = 0.8f;
        d.Eye.Left.Gaze = new VRCFaceTracking.Core.Types.Vector2(0.1f, -0.2f);
        d.Eye.Right.Gaze = new VRCFaceTracking.Core.Types.Vector2(0.1f, -0.2f);
        d.Shapes[(int)UnifiedExpressions.JawOpen].Weight = 0.3f;
        d.Shapes[(int)UnifiedExpressions.MouthCornerPullLeft].Weight = 0.25f;
        d.Shapes[(int)UnifiedExpressions.EyeWideLeft].Weight = 0.4f;
        Thread.Sleep(10);
    }

    public override void Teardown() => Logger.LogInformation("FakeSRanipal teardown");
}
