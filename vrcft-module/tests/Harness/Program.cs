// Usage: Harness <ModuleProcess.dll> <module.dll> <seconds>
// Prints one JSON line of tracking state every 100 ms to stdout; module logs go to stderr.
using System.Diagnostics;
using System.Globalization;
using Microsoft.Extensions.Logging.Abstractions;
using VRCFaceTracking;
using VRCFaceTracking.Core.Library;
using VRCFaceTracking.Core.Params.Expressions;
using VRCFaceTracking.Core.Sandboxing;
using VRCFaceTracking.Core.Sandboxing.IPC;

var moduleProcess = args[0];
var modulePath = args[1];
var seconds = double.Parse(args[2], CultureInfo.InvariantCulture);

var server = new VrcftSandboxServer(NullLoggerFactory.Instance, [9000, 9001]);
bool usingEye = false, usingExpr = false, initDone = false;
string moduleName = "";
var modulePort = -1;
var updates = 0;
var stateLock = new object();
CancellationTokenSource updateCts = new();

server.OnPacketReceived = (in IpcPacket packet, in int port) =>
{
    switch (packet.GetPacketType())
    {
        case IpcPacket.PacketType.Handshake:
            modulePort = port;
            server.SendData(new EventInitGetSupported(), port);
            break;
        case IpcPacket.PacketType.EventLog:
            Console.Error.WriteLine("MODULE: " + ((EventLogPacket)packet).Message);
            break;
        case IpcPacket.PacketType.ReplyGetSupported:
            server.SendData(new EventInitPacket { eyeAvailable = true, expressionAvailable = true }, port);
            break;
        case IpcPacket.PacketType.ReplyInit:
        {
            var reply = (ReplyInitPacket)packet;
            lock (stateLock)
            {
                usingEye = reply.eyeSuccess;
                usingExpr = reply.expressionSuccess;
                moduleName = reply.ModuleInformationName;
                initDone = true;
            }
            server.SendData(new EventStatusUpdatePacket { ModuleState = ModuleState.Active, UsingEye = usingEye, UsingExpression = usingExpr }, port);
            var p = port;
            new Thread(() =>
            {
                var update = new EventUpdatePacket();
                while (!updateCts.IsCancellationRequested)
                {
                    Thread.Sleep(10);
                    server.SendData(update, p);
                }
            }) { IsBackground = true }.Start();
            break;
        }
        case IpcPacket.PacketType.ReplyUpdate:
        {
            var reply = (ReplyUpdatePacket)packet;
            lock (stateLock)
            {
                if (usingEye) reply.UpdateGlobalEyeState();
                if (usingExpr) reply.UpdateGlobalExpressionState();
                updates++;
            }
            break;
        }
    }
};

var proc = Process.Start(new ProcessStartInfo("dotnet",
    $"\"{moduleProcess}\" --port {server.Port} --module-path \"{modulePath}\" --parent-pid {Environment.ProcessId}")
{
    RedirectStandardOutput = true,
    RedirectStandardError = true,
})!;
proc.OutputDataReceived += (_, e) => { if (e.Data != null) Console.Error.WriteLine("PROC: " + e.Data); };
proc.ErrorDataReceived += (_, e) => { if (e.Data != null) Console.Error.WriteLine("PROC: " + e.Data); };
proc.BeginOutputReadLine();
proc.BeginErrorReadLine();

string F(float v) => v.ToString("0.0000", CultureInfo.InvariantCulture);
var end = DateTime.UtcNow.AddSeconds(seconds);
while (DateTime.UtcNow < end)
{
    Thread.Sleep(100);
    lock (stateLock)
    {
        var d = UnifiedTracking.Data;
        float S(UnifiedExpressions e) => d.Shapes[(int)e].Weight;
        Console.WriteLine("{" +
            $"\"init\":{(initDone ? "true" : "false")},\"name\":\"{moduleName}\",\"eye\":{(usingEye ? "true" : "false")}," +
            $"\"expr\":{(usingExpr ? "true" : "false")},\"updates\":{updates}," +
            $"\"EyeOpenLeft\":{F(d.Eye.Left.Openness)},\"JawOpen\":{F(S(UnifiedExpressions.JawOpen))}," +
            $"\"MouthCornerPullLeft\":{F(S(UnifiedExpressions.MouthCornerPullLeft))}," +
            $"\"EyeWideLeft\":{F(S(UnifiedExpressions.EyeWideLeft))}," +
            $"\"BrowLowererLeft\":{F(S(UnifiedExpressions.BrowLowererLeft))}," +
            $"\"MouthPressLeft\":{F(S(UnifiedExpressions.MouthPressLeft))}" + "}");
        Console.Out.Flush();
    }
}
updateCts.Cancel();
if (modulePort > 0) server.SendData(new EventTeardownPacket(), modulePort);
Thread.Sleep(500);
try { proc.Kill(true); } catch { }
Environment.Exit(0);
