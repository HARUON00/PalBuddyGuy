"""VRCFaceTracking v6 module integration.

ProtocolTwoTests always run (a fake protocol-2 client in Python).

EndToEndTests drive the real chain - Python app -> TCP -> PalBuddyGuy module ->
real VRCFaceTracking module process -> sandbox IPC -> a host emulator - and need
.NET build outputs. Set PALBUDDY_VRCFT_TEST_DIR to a folder containing:
  mp/VRCFaceTracking.ModuleProcess.dll   (VRCFaceTracking.ModuleProcess build)
  harness/Harness.dll                    (vrcft-module/tests/Harness build)
  fake/FakeSRanipal.dll                  (vrcft-module/tests/FakeSRanipal build)
  v1check/V1Check.dll                    (vrcft-module/tests/V1Check build)
See vrcft-module/README.md for the build commands.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from palbuddy import vrcft_install  # noqa: E402
from palbuddy.config import Config, ExpressionClass  # noqa: E402
from palbuddy.net import recv_exact  # noqa: E402
from palbuddy.params import (LIP_SHAPES, UNIFIED_EXPRESSIONS, V1_ONLY, V1_UNSUPPORTED,  # noqa: E402
                             unified_weights)
from palbuddy.vrcft import VRCFTServer  # noqa: E402


def wait_for(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def read_message(sock):
    """Decode one app -> module protocol-2 message."""
    kind = recv_exact(sock, 1)[0]
    if kind == 5:
        mode, count = recv_exact(sock, 2)
        names = []
        for _ in range(count):
            n = recv_exact(sock, 1)[0]
            names.append(bytes(recv_exact(sock, n)).decode("ascii"))
        return ("table", mode, names)
    if kind == 3:
        count = recv_exact(sock, 1)[0]
        raw = recv_exact(sock, count * 3)
        return ("weights", [(raw[i * 3], ((raw[i * 3 + 1] << 8) | raw[i * 3 + 2]) / 65535) for i in range(count)])
    raise AssertionError("unknown message %d" % kind)


class ProtocolTwoTests(unittest.TestCase):
    def setUp(self):
        self.srv = VRCFTServer("127.0.0.1", 0).start()
        self.addCleanup(self.srv.stop)

    def connect_v2(self):
        c = socket.create_connection(("127.0.0.1", self.srv.port))
        self.addCleanup(c.close)
        c.sendall(b"PBG2\x02")
        c.settimeout(3)
        self.assertTrue(wait_for(lambda: self.srv.protocol == 2))
        return c

    def test_table_and_weights(self):
        c = self.connect_v2()
        # SRanipal name expands to several Unified shapes; Unified names pass through
        self.assertTrue(self.srv.send_targets([("MouthPout", 1.0), ("BrowLowererLeft", 0.0)]))
        kind, mode, names = read_message(c)
        self.assertEqual((kind, mode), ("table", 0))
        self.assertEqual(names, ["LipPuckerUpperRight", "LipPuckerUpperLeft", "LipPuckerLowerRight",
                                 "LipPuckerLowerLeft", "BrowLowererLeft"])
        self.assertTrue(all(n in UNIFIED_EXPRESSIONS for n in names))
        kind, weights = read_message(c)
        self.assertEqual([s for s, _ in weights], [0, 1, 2, 3, 4])
        self.assertAlmostEqual(weights[0][1], 1.0, places=4)   # v=+1 -> weight 1
        self.assertAlmostEqual(weights[4][1], 0.5, places=4)   # v=0 -> weight 0.5
        # the table is sent once per connection; a subset of targets reuses it
        self.assertTrue(self.srv.send_targets([("BrowLowererLeft", -1.0)]))
        kind, weights = read_message(c)
        self.assertEqual(kind, "weights")
        self.assertEqual(weights, [(4, 0.0)])
        self.srv.clear()
        self.assertEqual(read_message(c), ("weights", []))

    def test_legacy_module_detected_by_silence_or_data(self):
        c = socket.create_connection(("127.0.0.1", self.srv.port))
        self.addCleanup(c.close)
        self.assertTrue(wait_for(lambda: self.srv.connected))
        self.assertFalse(self.srv.send_targets([("JawOpen", 1.0)]))  # nothing sent before we know
        self.assertTrue(wait_for(lambda: self.srv.protocol == 1, timeout=3))
        self.assertTrue(self.srv.send_targets([("JawOpen", 1.0), ("BrowLowererLeft", 1.0)]))
        c.settimeout(3)
        self.assertEqual(recv_exact(c, 5), bytearray([2, 1, 3, 255, 254]))  # only the SRanipal shape

    def test_engine_sends_protocol_two(self):
        import numpy as np
        import torch
        from palbuddy.engine import Engine
        from palbuddy.model import BuddyNet
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cfg = Config(face_port=0, eye_port=0, vrcft_port=0, infer_engine="pytorch", infer_device="cpu",
                     model_path=os.path.join(tmp.name, "m.pt"), input_mode="face",
                     classes=[ExpressionClass("n"), ExpressionClass("frown", target="BrowLowererLeft")])
        engine = Engine(cfg).start()
        self.addCleanup(engine.stop)
        engine.model = BuddyNet(2, "face")
        c = socket.create_connection(("127.0.0.1", engine.vrcft.port))
        self.addCleanup(c.close)
        c.sendall(b"PBG2\x02")
        c.settimeout(3)
        self.assertTrue(wait_for(lambda: engine.vrcft.protocol == 2))
        engine.start_inference()
        engine.hub.push(None, np.full(25600, 0.5, np.float32).tobytes())
        self.assertEqual(read_message(c)[2], ["BrowLowererLeft"])
        kind, weights = read_message(c)
        self.assertEqual(kind, "weights")
        engine.stop_inference()
        while True:  # the last message after stopping is "clear"
            msg = read_message(c)
            if msg == ("weights", []):
                break
        del torch


class MappingTests(unittest.TestCase):
    def test_transforms(self):
        self.assertEqual(unified_weights("TongueLongStep1", 1.0), [("TongueOut", 0.5)])
        self.assertEqual(unified_weights("TongueLongStep2", 0.5), [("TongueOut", 0.75)])
        self.assertEqual(unified_weights("TongueLongStep2", 0.0), [("TongueOut", 0.0)])  # inactive = no push
        self.assertEqual(unified_weights("BrowDownLeft", 0.4), [("BrowPinchLeft", 0.4), ("BrowLowererLeft", 0.4)])
        self.assertEqual(unified_weights("MouthSadLeft", 0.3), [("MouthStretchLeft", 0.3)])
        self.assertEqual(unified_weights("JawOpen", 2.0), [("JawOpen", 1.0)])  # clamped
        for name in list(LIP_SHAPES) + list(V1_ONLY):
            for u, _ in unified_weights(name, 0.5):
                self.assertIn(u, UNIFIED_EXPRESSIONS)


class CompiledSendTests(unittest.TestCase):
    def test_send_path_equals_unified_weights(self):
        """VRCFTServer precompiles the mapping; what goes over the wire must equal
        params.unified_weights for every target (up to the 16-bit wire resolution)."""
        from palbuddy.params import all_target_names
        srv = VRCFTServer("127.0.0.1", 0).start()
        self.addCleanup(srv.stop)
        c = socket.create_connection(("127.0.0.1", srv.port))
        self.addCleanup(c.close)
        c.sendall(b"PBG2\x02")
        c.settimeout(3)
        self.assertTrue(wait_for(lambda: srv.protocol == 2))
        names = all_target_names()
        table = None
        for v in (-1.0, -0.4, 0.0, 0.3, 1.0):
            self.assertTrue(srv.send_targets([(n, v) for n in names]))
            msg = read_message(c)
            if msg[0] == "table":
                table = msg[2]
                msg = read_message(c)
            got = {}
            for slot, w in msg[1]:
                got[table[slot]] = max(got.get(table[slot], 0.0), w)  # module keeps the max per shape
            expected = {}
            for n in names:
                for u, uw in unified_weights(n, (v + 1.0) / 2.0):
                    expected[u] = max(expected.get(u, 0.0), uw)
            self.assertEqual(set(got), set(expected))
            for u in expected:
                self.assertAlmostEqual(got[u], expected[u], delta=1 / 65535 + 1e-9, msg=(u, v))


class InstallerTests(unittest.TestCase):
    def test_install_uninstall(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        libs = os.path.join(tmp.name, "CustomLibs")
        sran = os.path.join(libs, "some-guid")
        os.makedirs(sran)
        with open(os.path.join(sran, "module.json"), "w") as f:
            json.dump({"ModuleName": "SRanipal", "DllFileName": "SRanipalExtTrackingModule.dll"}, f)
        open(os.path.join(sran, "SRanipalExtTrackingModule.dll"), "wb").close()
        cfg = Config(vrcft_custom_libs=libs, vrcft_port=26499)

        steps = vrcft_install.install(cfg)
        self.assertTrue(any("Disabled" in s for s in steps))
        target = os.path.join(libs, vrcft_install.MODULE_ID)
        self.assertTrue(os.path.exists(os.path.join(target, "PalBuddyGuy.VRCFT.dll")))
        self.assertFalse(os.path.exists(os.path.join(sran, "module.json")))
        with open(os.path.join(target, "palbuddyguy.json")) as f:
            settings = json.load(f)
        self.assertEqual(settings["port"], 26499)
        self.assertTrue(settings["innerModule"].endswith("SRanipalExtTrackingModule.dll"))
        st = vrcft_install.status(cfg)
        self.assertTrue(st["installed"])
        self.assertTrue(st["sranipal"][0]["disabled"])

        vrcft_install.uninstall(cfg)
        self.assertFalse(os.path.exists(target))
        self.assertTrue(os.path.exists(os.path.join(sran, "module.json")))
        self.assertFalse(vrcft_install.status(cfg)["installed"])


TEST_DIR = os.environ.get("PALBUDDY_VRCFT_TEST_DIR")


@unittest.skipUnless(TEST_DIR and os.path.isdir(os.path.join(TEST_DIR, "v1check")),
                     "set PALBUDDY_VRCFT_TEST_DIR to run (needs .NET builds)")
class V1RoundTripTests(unittest.TestCase):
    """For avatars using VRCFaceTracking's v1 (SRanipal) parameters: the Unified weights we
    send must make VRCFT v6 compute the v1 parameter equal to the trained value. Checked with
    VRCFT's own parameter functions (vrcft-module/tests/V1Check)."""

    @classmethod
    def setUpClass(cls):
        cases = []
        for name in list(LIP_SHAPES) + list(V1_ONLY):
            for w in (0.2, 0.55, 1.0):
                cases.append({"id": "%s|%s" % (name, w), "shapes": dict(unified_weights(name, w))})
        out = subprocess.run(["dotnet", os.path.join(TEST_DIR, "v1check", "V1Check.dll")],
                             input="\n".join(json.dumps(c) for c in cases), capture_output=True, text=True,
                             timeout=120)
        cls.results = {}
        for line in out.stdout.splitlines():
            d = json.loads(line)
            name, w = d["id"].split("|")
            cls.results[(name, float(w))] = d["v1"]

    def test_every_target_reproduces_its_v1_parameter(self):
        wrong = []
        for (name, w), v1 in self.results.items():
            if name in V1_UNSUPPORTED:
                self.assertEqual(v1[name], 0.0)  # VRCFT v6 always sends 0 for these
                continue
            if abs(v1[name] - w) > 0.005:
                wrong.append((name, w, round(v1[name], 4)))
        self.assertEqual(wrong, [])

    def test_combined_v1_parameters(self):
        r = self.results
        self.assertAlmostEqual(r[("JawRight", 0.55)]["JawX"], 0.55, places=3)
        self.assertAlmostEqual(r[("JawLeft", 0.55)]["JawX"], -0.55, places=3)
        self.assertAlmostEqual(r[("MouthSmileLeft", 0.55)]["SmileSadLeft"], 0.55, places=3)
        self.assertAlmostEqual(r[("MouthSadLeft", 0.55)]["SmileSadLeft"], -0.55, places=3)
        self.assertAlmostEqual(r[("MouthSadLeft", 0.55)]["MouthSadRight"], 0.0, places=3)  # no leak
        self.assertAlmostEqual(r[("TongueUp", 0.55)]["TongueY"], 0.55, places=3)
        self.assertAlmostEqual(r[("CheekPuffLeft", 0.55)]["PuffSuckLeft"], 0.55, places=3)
        self.assertAlmostEqual(r[("TongueLongStep1", 0.55)]["TongueLongStep2"], 0.0, places=3)


@unittest.skipUnless(TEST_DIR and os.path.isdir(TEST_DIR), "set PALBUDDY_VRCFT_TEST_DIR to run (needs .NET builds)")
class EndToEndTests(unittest.TestCase):
    def test_overrides_through_real_module_process(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        libs = os.path.join(tmp.name, "CustomLibs")
        sran = os.path.join(libs, "sranipal-guid")
        os.makedirs(sran)
        import shutil
        shutil.copy(os.path.join(TEST_DIR, "fake", "FakeSRanipal.dll"), sran)
        with open(os.path.join(sran, "module.json"), "w") as f:
            json.dump({"ModuleName": "SRanipal (fake)", "DllFileName": "FakeSRanipal.dll"}, f)

        srv = VRCFTServer("127.0.0.1", 0).start()
        self.addCleanup(srv.stop)
        cfg = Config(vrcft_custom_libs=libs, vrcft_port=srv.port)
        vrcft_install.install(cfg)  # installs the committed prebuilt DLL
        module_dll = os.path.join(libs, vrcft_install.MODULE_ID, "PalBuddyGuy.VRCFT.dll")

        proc = subprocess.Popen(
            ["dotnet", os.path.join(TEST_DIR, "harness", "Harness.dll"),
             os.path.join(TEST_DIR, "mp", "VRCFaceTracking.ModuleProcess.dll"), module_dll, "9"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(proc.kill)
        states = []
        threading.Thread(target=lambda: [states.append((time.time(), json.loads(line)))
                                         for line in proc.stdout], daemon=True).start()
        latest = lambda: states[-1][1] if states else {}  # noqa: E731

        self.assertTrue(wait_for(lambda: latest().get("init") and srv.protocol == 2, timeout=20), latest())
        self.assertEqual(latest()["name"], "PalBuddyGuy + SRanipal")
        self.assertTrue(latest()["eye"] and latest()["expr"])
        self.assertTrue(wait_for(lambda: abs(latest().get("JawOpen", 0) - 0.3) < 1e-3))  # plain SRanipal

        stop = threading.Event()

        def drive():
            while not stop.is_set():
                srv.send_targets([("JawOpen", 0.0), ("BrowLowererLeft", 0.6)])  # weights 0.5 and 0.8
                time.sleep(1 / 60)
        driver = threading.Thread(target=drive, daemon=True)
        driver.start()
        ok = wait_for(lambda: abs(latest().get("JawOpen", 0) - 0.5) < 1e-3
                      and abs(latest().get("BrowLowererLeft", 0) - 0.8) < 1e-3, timeout=5)
        reached = time.time()
        time.sleep(1.5)
        stop.set()
        driver.join()
        self.assertTrue(ok, latest())
        # no flicker back to SRanipal's value while overriding (the wrapped module writes its
        # values and then sleeps inside Update, so a naive overlay would lose most samples)
        window = [st for t, st in states if reached + 0.05 < t < reached + 1.4]
        self.assertGreater(len(window), 8)
        self.assertTrue(all(abs(st["JawOpen"] - 0.5) < 1e-3 for st in window),
                        [st["JawOpen"] for st in window])
        s = latest()
        self.assertAlmostEqual(s["EyeOpenLeft"], 0.8, places=3)          # eye data passes through
        self.assertAlmostEqual(s["MouthCornerPullLeft"], 0.25, places=3)  # untouched shape passes through
        self.assertAlmostEqual(s["EyeWideLeft"], 0.4, places=3)

        # no clear message: the module must drop stale overrides by itself
        self.assertTrue(wait_for(lambda: abs(latest().get("JawOpen", 0) - 0.3) < 1e-3
                                 and latest().get("BrowLowererLeft", 1) == 0.0, timeout=3), latest())
        proc.wait(timeout=20)
        stderr = proc.stderr.read()
        self.assertIn("Loaded inner module FakeSRanipal.FakeSRanipalModule", stderr)
        self.assertIn("Connected to Pal Buddy Guy", stderr)


if __name__ == "__main__":
    unittest.main()
