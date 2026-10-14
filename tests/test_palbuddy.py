"""Offline tests: fake SRanipal and VRCFT peers talk to the real servers over loopback."""

import os
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from palbuddy.config import Config, ExpressionClass, default_config  # noqa: E402
from palbuddy.datasets import Recorder, frame_count, open_recording  # noqa: E402
from palbuddy.engine import Engine, normalize  # noqa: E402
from palbuddy.frames import (CAMERA_BYTES, FrameHub, ProxyClient, ProxyServer,  # noqa: E402
                             SRanipalReceiver)
from palbuddy.model import BuddyNet  # noqa: E402
from palbuddy.net import recv_exact  # noqa: E402
from palbuddy.trainer import train  # noqa: E402
from palbuddy.vrcft import VRCFTServer, encode_params  # noqa: E402


def neural(value):
    return np.full(64 * 20 * 20, value, dtype=np.float32).tobytes()


def packet(payload):
    return struct.pack("<iii", 0, len(payload), 1) + payload


def wait_for(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def legacy_encode(key, f):
    """Byte-for-byte what the original write_param produced."""
    f = (f + 1) * 32767.0
    f = int(max(0, min(65535, f)))
    return bytes([key]) + bytes([f // 256, f % 256])


class ProtocolTests(unittest.TestCase):
    def test_encode_matches_original(self):
        pairs = [(0, -1.0), (19, 0.25), (2, 1.0), (11, 1.7), (3, -3)]
        expected = bytes([2, len(pairs)]) + b"".join(legacy_encode(k, v) for k, v in pairs)
        self.assertEqual(encode_params(pairs), expected)

    def test_normalize_matches_original(self):
        self.assertAlmostEqual(normalize(0.45, 0.9), 0.0)
        self.assertEqual(normalize(5, 0.9), 1.0)
        self.assertEqual(normalize(0, 0.9), -1.0)


class ModelTests(unittest.TestCase):
    def test_legacy_checkpoint_roundtrip(self):
        m = BuddyNet(4)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "buddyguy.pt")
            # exactly the dict layout the original `save` command wrote
            torch.save({k: getattr(m, k).state_dict() for k in ("conv1", "conv2", "linear1", "linear2")}, path)
            m2 = BuddyNet.load(path, expected_outputs=4).eval()
            x = torch.randn(3, 128, 20, 20)
            m.eval()
            self.assertTrue(torch.allclose(m(x), m2(x)))
            with self.assertRaises(ValueError):
                BuddyNet.load(path, expected_outputs=5)

    def test_any_batch_size(self):
        m = BuddyNet(3).eval()
        for b in (1, 7):
            self.assertEqual(tuple(m(torch.randn(b, 128, 20, 20)).shape), (b, 3))


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        self.hub = FrameHub()
        self.rx = SRanipalReceiver(self.hub, "127.0.0.1", 0, 0).start()

    def tearDown(self):
        self.rx.stop()

    def connect(self, index):
        return socket.create_connection(("127.0.0.1", self.rx.ports[index]))

    def test_pairs_eye_and_face(self):
        self.hub.want_cameras()  # as the GUI does while the preview is visible
        face, eye = self.connect(0), self.connect(1)
        eye.sendall(packet(neural(1.0)) + packet(np.zeros(CAMERA_BYTES // 4, np.float32).tobytes()))
        time.sleep(0.1)
        face.sendall(packet(neural(2.0)))
        seq, sample = self.hub.wait_next(0, timeout=2)
        self.assertIsNotNone(sample)
        self.assertEqual(np.frombuffer(sample[0], np.float32)[0], 1.0)
        self.assertEqual(np.frombuffer(sample[1], np.float32)[0], 2.0)
        self.assertIsNotNone(self.hub.cameras["eye"])
        snap = self.rx.snapshot()
        self.assertEqual(snap["face"]["state"], "ok")
        face.close()
        eye.close()

    def test_swap(self):
        self.rx.set_swapped(True)
        a, b = self.connect(0), self.connect(1)
        a.sendall(packet(neural(1.0)))  # port 0 is now the eye
        time.sleep(0.1)
        b.sendall(packet(neural(2.0)))
        _, sample = self.hub.wait_next(0, timeout=2)
        self.assertEqual(np.frombuffer(sample[0], np.float32)[0], 1.0)
        a.close()
        b.close()

    def test_disconnect_and_reconnect(self):
        face = self.connect(0)
        self.assertTrue(wait_for(lambda: self.rx.status["port0"].connected))
        face.close()
        self.assertTrue(wait_for(lambda: not self.rx.status["port0"].connected))
        face2, eye = self.connect(0), self.connect(1)
        eye.sendall(packet(neural(3.0)))
        time.sleep(0.1)
        face2.sendall(packet(neural(4.0)))
        _, sample = self.hub.wait_next(0, timeout=2)
        self.assertIsNotNone(sample)
        face2.close()
        eye.close()

    def test_new_connection_replaces_stale_one(self):
        old = self.connect(0)
        self.assertTrue(wait_for(lambda: self.rx.status["port0"].connected))
        new, eye = self.connect(0), self.connect(1)
        eye.sendall(packet(neural(5.0)))
        time.sleep(0.1)
        new.sendall(packet(neural(6.0)))
        _, sample = self.hub.wait_next(0, timeout=2)
        self.assertEqual(np.frombuffer(sample[1], np.float32)[0], 6.0)
        old.settimeout(2)
        self.assertEqual(old.recv(1), b"")  # server closed the stale socket
        for s in (old, new, eye):
            s.close()

    def test_bad_packet_drops_connection(self):
        face = self.connect(0)
        face.sendall(struct.pack("<iii", 0, 12345, 0))
        face.settimeout(2)
        self.assertEqual(face.recv(1), b"")
        face.close()


class ProxyTests(unittest.TestCase):
    def test_proxy_roundtrip(self):
        src = FrameHub()
        server = ProxyServer(src, "127.0.0.1", 0).start()
        dst = FrameHub()
        client = ProxyClient(dst, "127.0.0.1", server.port).start()
        self.assertTrue(wait_for(lambda: server.clients == 1))
        src.push(neural(7.0), neural(8.0))
        _, sample = dst.wait_next(0, timeout=2)
        self.assertEqual(np.frombuffer(sample[1], np.float32)[0], 8.0)
        client.stop()
        server.stop()


class VRCFTTests(unittest.TestCase):
    def test_send_and_reconnect(self):
        srv = VRCFTServer("127.0.0.1", 0).start()
        c1 = socket.create_connection(("127.0.0.1", srv.port))
        self.assertTrue(wait_for(lambda: srv.connected))
        self.assertTrue(srv.send_params([(3, 1.0)]))
        self.assertEqual(recv_exact(c1, 5), bytearray([2, 1, 3, 255, 254]))
        # VRCFT sends lip data; must be consumed without errors
        c1.sendall(bytes([2]) + b"\x7f\xff" * 60)
        self.assertTrue(wait_for(lambda: 2 in srv.last_received))
        c2 = socket.create_connection(("127.0.0.1", srv.port))
        self.assertTrue(wait_for(lambda: srv.peer and srv.peer.endswith(str(c2.getsockname()[1]))))
        self.assertTrue(srv.send_params([(1, -1.0)]))
        self.assertEqual(recv_exact(c2, 5), bytearray([2, 1, 1, 0, 0]))
        c2.close()
        self.assertTrue(wait_for(lambda: not srv.connected))
        self.assertFalse(srv.send_params([(1, 0.0)]))
        c1.close()
        srv.stop()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(dataset_folder=self.tmp.name, model_path=os.path.join(self.tmp.name, "m.pt"),
                          face_port=0, eye_port=0, vrcft_port=0, record_countdown=0, record_frames=40,
                          epochs=2, batch_size=16, mixed_precision=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_train_infer(self):
        hub = FrameHub()
        path = os.path.join(self.tmp.name, "a-em.mmap")
        done = threading.Event()
        result = {}
        Recorder(hub, path, 40, countdown=0, on_done=lambda ok, m: (result.update(ok=ok), done.set())).start()
        time.sleep(0.2)
        for i in range(60):
            hub.push(neural(i), neural(-i))
        self.assertTrue(done.wait(5))
        self.assertTrue(result["ok"])
        self.assertEqual(frame_count(path), 40)
        rec = open_recording(path)
        self.assertEqual(rec[3, 0, 0, 0], 3.0)
        self.assertEqual(rec[3, 64, 0, 0], -3.0)

        # tiny synthetic 2-class set
        rng = np.random.default_rng(0)
        for name, offset in (("n", 0.0), ("s", 1.0)):
            mm = np.memmap(os.path.join(self.tmp.name, name + ".mmap"), np.float32, "w+", shape=(32, 128, 20, 20))
            mm[:] = rng.normal(offset, 0.1, size=mm.shape)
            mm.flush()
            del mm
        self.cfg.classes = [ExpressionClass("neutral", ["n.mmap"]), ExpressionClass("smile", ["s.mmap"], "JawOpen")]
        model, history = train(self.cfg, log_fn=lambda *a: None, device=torch.device("cpu"))
        self.assertEqual(len(history), 2)

        engine = Engine(self.cfg)
        engine.device = torch.device("cpu")
        engine.start()
        try:
            engine.model = model
            vr = socket.create_connection(("127.0.0.1", engine.vrcft.port))
            self.assertTrue(wait_for(lambda: engine.vrcft.connected))
            engine.start_inference()
            eye = socket.create_connection(("127.0.0.1", engine.source.ports[1]))
            face = socket.create_connection(("127.0.0.1", engine.source.ports[0]))
            eye.sendall(packet(neural(1.0)))
            time.sleep(0.05)
            face.sendall(packet(neural(1.0)))
            vr.settimeout(3)
            msg = recv_exact(vr, 5)
            self.assertEqual(msg[:3], bytearray([2, 1, 3]))  # JawOpen
            self.assertIsNotNone(engine.last_raw)
            engine.stop_inference()
            self.assertFalse(engine.inferring)
            for s in (vr, eye, face):
                s.close()
        finally:
            engine.stop()


class SingleTrackerTests(unittest.TestCase):
    """Face-only and eye-only setups."""

    def make_rx(self, mode, swapped=False):
        hub = FrameHub()
        rx = SRanipalReceiver(hub, "127.0.0.1", 0, 0, swapped=swapped, mode=mode).start()
        self.addCleanup(rx.stop)
        return hub, rx

    def connect(self, rx, index):
        s = socket.create_connection(("127.0.0.1", rx.ports[index]))
        self.addCleanup(s.close)
        return s

    def test_face_only_on_either_port(self):
        # the single stream lands on the port that is "eye" by default: no swap needed
        hub, rx = self.make_rx("face")
        s = self.connect(rx, 1)
        s.sendall(packet(neural(2.0)))
        _, sample = hub.wait_next(0, timeout=2)
        self.assertIsNone(sample[0])
        self.assertEqual(np.frombuffer(sample[1], np.float32)[0], 2.0)
        self.assertEqual(rx.snapshot()["face"]["state"], "ok")

    def test_eye_only_on_either_port(self):
        hub, rx = self.make_rx("eye")
        s = self.connect(rx, 0)
        s.sendall(packet(neural(3.0)))
        _, sample = hub.wait_next(0, timeout=2)
        self.assertEqual(np.frombuffer(sample[0], np.float32)[0], 3.0)
        self.assertIsNone(sample[1])

    def test_face_mode_with_both_trackers_ignores_eye(self):
        hub, rx = self.make_rx("face")
        face, eye = self.connect(rx, 0), self.connect(rx, 1)
        face.sendall(packet(neural(1.0)))
        eye.sendall(packet(neural(9.0)))
        seq, sample = hub.wait_next(0, timeout=2)
        time.sleep(0.2)
        for _ in range(3):
            eye.sendall(packet(neural(9.0)))
        face.sendall(packet(neural(4.0)))
        deadline = time.time() + 2
        while time.time() < deadline:
            seq, sample = hub.wait_next(seq, timeout=0.5)
            if sample is None:
                continue
            self.assertIsNone(sample[0])
            self.assertNotEqual(np.frombuffer(sample[1], np.float32)[0], 9.0)
            if np.frombuffer(sample[1], np.float32)[0] == 4.0:
                break
        else:
            self.fail("face sample not received")

    def test_stalled_face_never_takes_eye_data(self):
        hub, rx = self.make_rx("face")
        face, eye = self.connect(rx, 0), self.connect(rx, 1)
        face.sendall(packet(neural(1.0)))
        seq, _ = hub.wait_next(0, timeout=2)
        rx.status["port0"].last_frame -= 10  # face tracker stalled for a long time
        eye.sendall(packet(neural(9.0)) * 5)
        seq2, sample = hub.wait_next(seq, timeout=0.5)
        self.assertIsNone(sample)

    def test_proxy_zero_fills_missing_tracker(self):
        src = FrameHub()
        server = ProxyServer(src, "127.0.0.1", 0).start()
        self.addCleanup(server.stop)
        c = socket.create_connection(("127.0.0.1", server.port))
        self.addCleanup(c.close)
        self.assertTrue(wait_for(lambda: server.clients == 1))
        src.push(None, neural(5.0))
        data = np.frombuffer(bytes(recv_exact(c, 2 * 102400)), np.float32)
        self.assertEqual(data[0], 0.0)
        self.assertEqual(data[25600], 5.0)

    def test_model_mode_in_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            m = BuddyNet(2, "face")
            self.assertEqual(m(torch.randn(1, 64, 20, 20)).shape[1], 2)
            m.save(path)
            self.assertEqual(BuddyNet.load(path, expected_mode="face").input_mode, "face")
            with self.assertRaises(ValueError):
                BuddyNet.load(path, expected_mode="both")
            legacy = os.path.join(d, "legacy.pt")
            b = BuddyNet(2)
            torch.save({k: getattr(b, k).state_dict() for k in ("conv1", "conv2", "linear1", "linear2")}, legacy)
            self.assertEqual(BuddyNet.load(legacy).input_mode, "both")

    def test_train_and_infer_face_only(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        rng = np.random.default_rng(1)
        for name, offset, eye in (("n", 0.0, True), ("s", 1.0, True), ("faceonly", 1.0, False)):
            mm = np.memmap(os.path.join(tmp.name, name + ".mmap"), np.float32, "w+", shape=(16, 128, 20, 20))
            mm[:] = rng.normal(offset, 0.1, size=mm.shape)
            if not eye:
                mm[:, :64] = 0  # recorded with the facial tracker only
            mm.flush()
            del mm
        cfg = Config(dataset_folder=tmp.name, model_path=os.path.join(tmp.name, "m.pt"),
                     face_port=0, eye_port=0, vrcft_port=0, epochs=1, batch_size=64, validation_split=0,
                     mixed_precision=False, input_mode="face",
                     classes=[ExpressionClass("neutral", ["n.mmap"]),
                              ExpressionClass("smile", ["s.mmap", "faceonly.mmap"], "JawOpen")])
        model, _ = train(cfg, log_fn=lambda *a: None, device=torch.device("cpu"))
        self.assertEqual(model.input_mode, "face")
        self.assertEqual(model.conv1.in_channels, 64)

        cfg.input_mode = "both"  # the face-only recording can't feed a both-model
        with self.assertRaises(ValueError):
            train(cfg, log_fn=lambda *a: None, device=torch.device("cpu"))
        cfg.input_mode = "face"

        engine = Engine(cfg)
        engine.device = torch.device("cpu")
        engine.start()
        self.addCleanup(engine.stop)
        engine.model = model
        vr = socket.create_connection(("127.0.0.1", engine.vrcft.port))
        self.addCleanup(vr.close)
        self.assertTrue(wait_for(lambda: engine.vrcft.connected))
        engine.start_inference()
        face = socket.create_connection(("127.0.0.1", engine.source.ports[1]))
        self.addCleanup(face.close)
        face.sendall(packet(neural(1.0)))
        vr.settimeout(3)
        self.assertEqual(recv_exact(vr, 5)[:3], bytearray([2, 1, 3]))
        self.assertIsNone(engine.status()["mode_hint"])
        engine.set_input_mode("both")  # a face model can't run in both mode
        self.assertIsNone(engine.model)
        self.assertFalse(engine.inferring)

    def test_mode_hint(self):
        e = Engine(Config(input_mode="both"))
        ok = {"state": "ok"}
        off = {"state": "disconnected"}
        self.assertEqual(e.mode_hint({"eye": off, "face": ok}), "single")
        self.assertEqual(e.mode_hint({"eye": ok, "face": off}), "single")
        self.assertIsNone(e.mode_hint({"eye": ok, "face": ok}))
        self.assertIsNone(e.mode_hint({"eye": off, "face": off}))
        e.cfg.input_mode = "face"
        self.assertEqual(e.mode_hint({"eye": ok, "face": ok}), "both")


class InferencePerfTests(unittest.TestCase):
    def test_runtime_matches_model(self):
        from palbuddy.inference import Runtime
        m = BuddyNet(5).eval()
        x = np.random.default_rng(0).random((1, 128, 20, 20), dtype=np.float32)
        with torch.no_grad():
            ref = m(torch.from_numpy(x))[0].numpy()
        sample = (x[0, :64].tobytes(), x[0, 64:].tobytes())
        fp32 = Runtime(m, "cpu", 1, int8=False)
        self.assertEqual(fp32.describe(), "CPU fp32 ×1")
        np.testing.assert_allclose(fp32.predict(sample), ref, atol=1e-5)
        q = Runtime(m, "cpu", 1, int8=True)
        if q.quantized:
            self.assertEqual(q.describe(), "CPU int8 ×1")
            self.assertLess(np.abs(q.predict(sample) - ref).max(), 0.05)
        # the original model is untouched by quantization
        self.assertIsInstance(m.linear1, torch.nn.Linear)

    def test_runtime_single_tracker(self):
        from palbuddy.inference import Runtime
        m = BuddyNet(2, "eye").eval()
        rt = Runtime(m, "cpu", 1, int8=False)
        self.assertEqual(rt.predict((neural(1.0), None)).shape, (2,))
        self.assertIsNone(rt.predict((None, neural(1.0))))

    def test_benchmark(self):
        from palbuddy.inference import benchmark
        res = benchmark(BuddyNet(3), threads=1, frames=5, include_gpu=False, log_fn=lambda *a: None)
        self.assertGreaterEqual(len(res), 1)
        self.assertTrue(all(r["ms"] > 0 for r in res))

    def test_rate_cap_and_thread_restore(self):
        cfg = Config(face_port=0, eye_port=0, vrcft_port=0, input_mode="face", max_infer_rate=10,
                     infer_engine="pytorch",
                     infer_device="cpu", infer_threads=1, classes=[ExpressionClass("n"), ExpressionClass("s")])
        engine = Engine(cfg).start()
        self.addCleanup(engine.stop)
        engine.model = BuddyNet(2, "face")
        threads_before = torch.get_num_threads()
        engine.start_inference()
        self.assertTrue(wait_for(lambda: engine.infer_backend is not None))
        stop = threading.Event()

        def feed():  # 100 Hz of frames
            while not stop.is_set():
                engine.hub.push(None, neural(0.5))
                time.sleep(0.01)
        threading.Thread(target=feed, daemon=True).start()
        time.sleep(2.2)
        rate = engine.infer_rate.rate()
        stop.set()
        self.assertLess(rate, 13)
        self.assertGreater(rate, 5)
        engine.stop_inference()
        self.assertEqual(torch.get_num_threads(), threads_before)
        self.assertIsNone(engine.infer_backend)

    def test_benchmark_via_engine_resumes_tracking(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cfg = Config(face_port=0, eye_port=0, vrcft_port=0, infer_device="cpu",
                     model_path=os.path.join(tmp.name, "m.pt"),
                     classes=[ExpressionClass("n"), ExpressionClass("s")])
        engine = Engine(cfg).start()
        self.addCleanup(engine.stop)
        engine.model = BuddyNet(2)
        engine.start_inference()
        done = threading.Event()
        out = {}
        engine.run_benchmark(on_done=lambda r, e: (out.update(r=r, e=e), done.set()))
        self.assertTrue(done.wait(120))
        self.assertIsNone(out["e"])
        self.assertTrue(wait_for(lambda: engine.inferring))
        self.assertIsNone(engine.busy)


def make_recordings(folder, spec, frames=32, seed=0):
    """spec: {name: offset}. Separable synthetic recordings."""
    rng = np.random.default_rng(seed)
    for name, offset in spec.items():
        mm = np.memmap(os.path.join(folder, name + ".mmap"), np.float32, "w+", shape=(frames, 128, 20, 20))
        mm[:] = rng.normal(offset, 0.1, size=mm.shape)
        mm.flush()
        del mm


class LiteAndValidationTests(unittest.TestCase):
    def test_lite_model(self):
        std, lite = BuddyNet(4), BuddyNet(4, arch="lite")
        n = lambda m: sum(p.numel() for p in m.parameters())
        self.assertLess(n(lite) * 6, n(std))
        self.assertEqual(tuple(lite.eval()(torch.randn(2, 128, 20, 20)).shape), (2, 4))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            lite.save(path)
            self.assertEqual(BuddyNet.load(path).arch, "lite")
            # checkpoint without an "arch" key: detected from the layer shapes
            torch.save({k: getattr(lite, k).state_dict() for k in ("conv1", "conv2", "linear1", "linear2")}, path)
            self.assertEqual(BuddyNet.load(path).arch, "lite")

    def test_split_holds_out_the_tail(self):
        from palbuddy.trainer import split_recordings
        rec = np.arange(100, dtype=np.float32)[:, None, None, None] * np.ones((1, 128, 20, 20), np.float32)
        train_recs, _, vx, vt = split_recordings([[rec], [rec]], 0.1, slice(0, 128))
        self.assertEqual(len(train_recs[0][0]), 90)
        self.assertTrue((vx[:, 0, 0, 0] >= 90).all())  # only frames the model never trains on
        self.assertEqual(sorted(set(vt.argmax(axis=1).tolist())), [0, 1])
        self.assertIsNone(split_recordings([[rec]], 0, slice(0, 128))[2])

    def test_train_lite_with_validation(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        make_recordings(tmp.name, {"n": 0.0, "s": 1.0}, frames=64)
        cfg = Config(dataset_folder=tmp.name, epochs=3, batch_size=32, mixed_precision=False, model_arch="lite",
                     validation_split=0.2, classes=[ExpressionClass("neutral", ["n.mmap"]),
                                                    ExpressionClass("smile", ["s.mmap"], "JawOpen")])
        seen = []
        model, _ = train(cfg, log_fn=lambda *a: None, device=torch.device("cpu"), seed=0,
                         on_progress=lambda **i: seen.append(i) if i.get("epoch_done") else None)
        self.assertEqual(model.arch, "lite")
        self.assertIn("val_acc", seen[-1])
        self.assertGreater(model.val_metrics["val_acc"], 0.9)  # trivially separable data
        self.assertEqual(set(model.val_metrics["per_class"]), {"neutral", "smile"})


@unittest.skipUnless(__import__("importlib").util.find_spec("onnxruntime"), "onnxruntime not installed")
class OnnxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_export_and_runtime_match_torch(self):
        from palbuddy.onnx_export import export_onnx
        from palbuddy.onnx_runtime import OnnxRuntime
        for arch, mode in (("standard", "both"), ("lite", "face")):
            m = BuddyNet(3, mode, arch).eval()
            base = os.path.join(self.tmp.name, arch)
            paths = export_onnx(m, base, ["a", "b", "c"])
            self.assertTrue(os.path.exists(paths["fp32"]))
            ch = 128 if mode == "both" else 64
            x = np.random.default_rng(1).random((1, ch, 20, 20), dtype=np.float32)
            with torch.no_grad():
                ref = m(torch.from_numpy(x))[0].numpy()
            fp32 = OnnxRuntime(base, "cpu", 1, int8=False, expected_mode=mode, expected_outputs=3)
            np.testing.assert_allclose(fp32.predict_array(x), ref, atol=1e-4)
            self.assertIn("fp32", fp32.describe())
            if os.path.exists(paths["int8"]):
                q = OnnxRuntime(base, "cpu", 1, int8=True)
                self.assertTrue(q.quantized)
                self.assertLess(np.abs(q.predict_array(x) - ref).max(), 0.05)
            with self.assertRaises(ValueError):
                OnnxRuntime(base, "cpu", expected_outputs=4)

    def test_engine_tracks_with_onnx_and_reexports_stale(self):
        make_recordings(self.tmp.name, {"n": 0.0, "s": 1.0})
        model_path = os.path.join(self.tmp.name, "m.pt")
        cfg = Config(dataset_folder=self.tmp.name, model_path=model_path, face_port=0, eye_port=0, vrcft_port=0,
                     infer_engine="onnx", infer_device="cpu",
                     classes=[ExpressionClass("neutral", ["n.mmap"]), ExpressionClass("smile", ["s.mmap"], "JawOpen")])
        engine = Engine(cfg)
        engine.device = torch.device("cpu")
        engine.start()
        self.addCleanup(engine.stop)

        engine.model, engine.model_dirty = BuddyNet(2), True
        engine.start_inference()  # unsaved model -> temporary export
        self.assertTrue(wait_for(lambda: engine.infer_backend is not None))
        self.assertTrue(engine.infer_backend.startswith("ONNX CPU"))
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "m.onnx")))
        engine.stop_inference()

        engine.save_model()  # saving also exports next to the .pt
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "m.onnx")))
        meta = os.path.join(self.tmp.name, "m.json")
        os.utime(meta, (time.time() - 100, time.time() - 100))  # .pt now newer than the export
        engine.model = None
        vr = socket.create_connection(("127.0.0.1", engine.vrcft.port))
        self.addCleanup(vr.close)
        self.assertTrue(wait_for(lambda: engine.vrcft.connected))
        engine.start_inference()
        self.assertGreater(os.path.getmtime(meta), time.time() - 50)  # re-exported
        engine.hub.push(neural(1.0), neural(1.0))
        vr.settimeout(3)
        self.assertEqual(recv_exact(vr, 5)[:3], bytearray([2, 1, 3]))

    def test_tracking_without_pytorch(self):
        """A process that can't import torch still tracks from the exported .onnx files."""
        from palbuddy.onnx_export import export_onnx
        base = os.path.join(self.tmp.name, "m")
        export_onnx(BuddyNet(2, "face"), base)
        code = r"""
import sys, time, json
sys.modules["torch"] = None  # any torch import now fails
sys.path.insert(0, %r)
import numpy as np
from palbuddy.config import Config, ExpressionClass
from palbuddy.engine import Engine
cfg = Config(model_path=%r, face_port=0, eye_port=0, vrcft_port=0, input_mode="face", infer_device="cpu",
             classes=[ExpressionClass("n"), ExpressionClass("s", target="JawOpen")])
e = Engine(cfg).start()
e.start_inference()
for _ in range(100):
    e.hub.push(None, np.full(25600, 0.5, np.float32).tobytes())
    time.sleep(0.01)
    if e.last_raw is not None:
        break
print(json.dumps({"backend": e.infer_backend, "raw": None if e.last_raw is None else len(e.last_raw)}))
e.stop()
""" % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), base + ".pt")
        import json
        import subprocess
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        result = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertTrue(result["backend"].startswith("ONNX"))
        self.assertEqual(result["raw"], 2)

    def test_benchmark_includes_onnx(self):
        cfg = Config(face_port=0, eye_port=0, vrcft_port=0, infer_device="cpu", model_arch="lite",
                     model_path=os.path.join(self.tmp.name, "m.pt"),
                     classes=[ExpressionClass("n"), ExpressionClass("s")])
        engine = Engine(cfg).start()
        self.addCleanup(engine.stop)
        done = threading.Event()
        out = {}
        engine.run_benchmark(on_done=lambda r, e: (out.update(r=r, e=e), done.set()))
        self.assertTrue(done.wait(120))
        self.assertIsNone(out["e"])
        self.assertTrue(any(r["backend"].startswith("ONNX") for r in out["r"]))


class SystemTests(unittest.TestCase):
    def test_parse_hybrid_core_info(self):
        """Synthetic GetLogicalProcessorInformationEx buffer for a 12700K: 8 P-cores with 2
        threads each (efficiency class 1) and 4 E-cores (class 0)."""
        import struct
        from palbuddy import system
        buf = b""
        for core in range(12):
            p_core = core < 8
            mask = (0b11 << (core * 2)) if p_core else (1 << (16 + core - 8))
            rel = struct.pack("<BB20xH", 1 if p_core else 0, 1 if p_core else 0, 1)
            rel += struct.pack("<QH6x", mask, 0)
            buf += struct.pack("<II", 0, 8 + len(rel)) + rel
        cores = system.parse_core_info(buf, len(buf))
        self.assertEqual(len(cores), 12)
        self.assertEqual(sum(1 for c, _ in cores if c == 0), 4)
        orig = system.core_efficiency_classes
        system.core_efficiency_classes = lambda: cores
        try:
            self.assertEqual(system.efficiency_core_mask(), 0xF0000)
            self.assertTrue(system.is_hybrid_cpu())
            system.core_efficiency_classes = lambda: [(0, 1), (0, 2)]  # not hybrid
            self.assertEqual(system.efficiency_core_mask(), 0)
        finally:
            system.core_efficiency_classes = orig

    def test_priority_validation(self):
        from palbuddy import system
        with self.assertRaises(ValueError):
            system.set_priority("turbo")
        self.assertEqual(Config().process_priority, "below_normal")


class RegressionTests(unittest.TestCase):
    """Bugs found in the final test pass."""

    def test_config_ignores_unknown_keys(self):
        cfg = Config.from_dict({"epochs": 3, "future_key": 1, "classes": [
            {"name": "n", "files": ["a.mmap"], "extra": True}, {"no_name": 1}, "garbage"]})
        self.assertEqual(cfg.epochs, 3)
        self.assertEqual([c.name for c in cfg.classes], ["n"])

    def test_corrupt_config_is_moved_aside(self):
        import subprocess
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "config.json")
        with open(path, "w") as f:
            f.write("{not json")
        r = subprocess.run([sys.executable, "-m", "palbuddy", "--cli", "--config", path], input="quit\n",
                           capture_output=True, text=True, timeout=120,
                           cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        self.assertTrue(any(f.startswith("config.json.broken-") for f in os.listdir(tmp.name)))
        Config.load(path)  # a fresh default config was written

    def _tiny_cfg(self, tmp):
        make = np.random.default_rng(0)
        for n, off in (("a", 0.0), ("b", 1.0)):
            mm = np.memmap(os.path.join(tmp, n + ".mmap"), np.float32, "w+", shape=(32, 128, 20, 20))
            mm[:] = make.normal(off, .3, mm.shape)
            mm.flush()
            del mm
        return Config(dataset_folder=tmp, epochs=1, batch_size=64, model_arch="lite", mixed_precision=False,
                      classes=[ExpressionClass("n", ["a.mmap"]), ExpressionClass("s", ["b.mmap"], "JawOpen")])

    def test_training_leaves_no_loader_threads(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        train(self._tiny_cfg(tmp.name), log_fn=lambda *a: None, device=torch.device("cpu"))
        self.assertFalse([t for t in threading.enumerate() if t.name.startswith("batch-loader")])

    def test_loader_error_surfaces_instead_of_hanging(self):
        from palbuddy import trainer as T
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        original = T.BatchSampler.sample
        T.BatchSampler.sample = lambda self: (_ for _ in ()).throw(OSError("disk gone"))
        try:
            result = {}
            th = threading.Thread(target=lambda: result.update(
                e=self.assertRaises(OSError, train, self._tiny_cfg(tmp.name), log_fn=lambda *a: None,
                                    device=torch.device("cpu"))))
            th.start()
            th.join(60)
            self.assertFalse(th.is_alive(), "training hung on a loader error")
        finally:
            T.BatchSampler.sample = original
        self.assertFalse([t for t in threading.enumerate() if t.name.startswith("batch-loader")])

    def test_install_while_vrcft_holds_the_dll(self):
        from palbuddy import vrcft_install
        import shutil
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cfg = Config(vrcft_custom_libs=tmp.name)
        original = shutil.copy2
        shutil.copy2 = lambda *a, **k: (_ for _ in ()).throw(PermissionError("in use"))
        try:
            with self.assertRaises(vrcft_install.VRCFTRunningError) as cm:
                vrcft_install.install(cfg)
            self.assertIn("Close VRCFaceTracking", str(cm.exception))
        finally:
            shutil.copy2 = original


def decode_osc(packet):
    """-> (address, float) for a single-float OSC message."""
    end = packet.index(b"\0")
    address = packet[:end].decode()
    i = (end + 4) & ~3
    assert packet[i:i + 2] == b",f", packet
    return address, struct.unpack(">f", packet[i + 4:i + 8])[0]


class MergedParamTests(unittest.TestCase):
    def test_osc_encoding(self):
        from palbuddy.osc import encode_float
        pkt = encode_float("/avatar/parameters/SmileSad", 0.25)
        self.assertEqual(len(pkt) % 4, 0)
        self.assertEqual(decode_osc(pkt), ("/avatar/parameters/SmileSad", 0.25))
        self.assertEqual(encode_float("/abc", 1.0), b"/abc\0\0\0\0,f\0\0" + struct.pack(">f", 1.0))

    def test_combine_ranges(self):
        from palbuddy.config import MergedParam
        m = MergedParam.pair("SmileSad", "smile", "sad", -1.0, 1.0)
        self.assertEqual(m.combine({"smile": 1.0, "sad": 0.0}), 1.0)
        self.assertEqual(m.combine({"smile": 0.0, "sad": 1.0}), -1.0)
        self.assertEqual(m.combine({"smile": 0.0, "sad": 0.0}), 0.0)
        self.assertAlmostEqual(m.combine({"smile": 0.3, "sad": 0.1}), 0.2)
        two = MergedParam.pair("SmileSad", "smile", "sad", 0.0, 2.0)
        self.assertEqual(two.combine({"smile": 0.0, "sad": 1.0}), 0.0)
        self.assertEqual(two.combine({}), 1.0)
        self.assertEqual(two.combine({"smile": 0.5}), 1.5)
        self.assertEqual(two.neutral, 1.0)
        self.assertTrue(two.beyond_sync_range)
        half = MergedParam.pair("X", "smile", None, 0.0, 1.0)
        self.assertEqual(half.combine({"smile": 1.0}), 1.0)
        self.assertEqual(half.combine({"smile": 0.0}), 0.5)
        flipped = MergedParam.pair("X", "smile", "sad", 1.0, -1.0)  # reversed range is allowed
        self.assertEqual(flipped.combine({"smile": 1.0}), -1.0)

    def test_config_roundtrip_and_validation(self):
        from palbuddy.config import MergedParam
        cfg = Config(classes=[ExpressionClass("neutral"), ExpressionClass("smile"), ExpressionClass("sad")],
                     merged_params=[MergedParam.pair("PBG_SmileSad", "smile", "sad", 0.0, 2.0)])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.json")
            cfg.save(path)
            again = Config.load(path)
        self.assertEqual(again.merged_params[0], cfg.merged_params[0])
        self.assertEqual(again.validate(), [])
        self.assertEqual(again.output_problems(), [])
        cfg.merged_params = [MergedParam.pair("bad name", "smile", "nope", 1.0, 1.0), MergedParam("PBG_Empty")]
        problems = cfg.output_problems()
        self.assertEqual(len(problems), 4, problems)
        self.assertEqual(cfg.validate(), [])  # output problems never block training

    def test_engine_sends_merged_over_osc(self):
        from palbuddy.config import MergedParam
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.bind(("127.0.0.1", 0))
        rx.settimeout(3)
        self.addCleanup(rx.close)
        raw = {"value": np.array([0.9, 0.0, 0.0], np.float32)}

        class FakeRuntime:
            def activate(self): pass
            def release(self): pass
            def warmup(self): pass
            def describe(self): return "fake"
            def predict(self, sample): return raw["value"]

        cfg = Config(face_port=0, eye_port=0, vrcft_port=0, osc_port=rx.getsockname()[1],
                     classes=[ExpressionClass("neutral"), ExpressionClass("smile", max_power=0.5),
                              ExpressionClass("sad", max_power=0.5)],
                     merged_params=[MergedParam.pair("PBG_SmileSad", "smile", "sad", -1.0, 1.0),
                                    MergedParam.pair("PBG_Mood02", "smile", "sad", 0.0, 2.0),
                                    MergedParam.pair("PBG_Off", "smile", None, enabled=False)])
        engine = Engine(cfg).start()
        self.addCleanup(engine.stop)
        engine._make_runtime = lambda: FakeRuntime()
        engine.start_inference()

        def latest(after_push):
            got = {}
            end = time.time() + 3
            while time.time() < end and len(got) < 2:
                after_push()
                try:
                    addr, v = decode_osc(rx.recv(256))
                    got[addr.rsplit("/", 1)[1]] = v
                except socket.timeout:
                    break
            return got

        push = lambda: engine.hub.push(neural(1.0), neural(1.0))  # noqa: E731
        raw["value"] = np.array([0.9, 0.5, 0.0], np.float32)  # smile full (max_power 0.5)
        self.assertTrue(wait_for(lambda: engine.last_merged.get("PBG_SmileSad") == 1.0 or push()))
        got = latest(push)
        self.assertAlmostEqual(got["PBG_SmileSad"], 1.0, places=4)
        self.assertAlmostEqual(got["PBG_Mood02"], 2.0, places=4)
        self.assertNotIn("PBG_Off", got)
        raw["value"] = np.array([0.9, 0.0, 0.25], np.float32)  # half sad
        self.assertTrue(wait_for(lambda: abs(engine.last_merged.get("PBG_SmileSad", 0) + 0.5) < 1e-6 or push()))
        engine.stop_inference()  # leaves the parameters at neutral
        tail = []
        try:
            while True:
                tail.append(decode_osc(rx.recv(256)))
        except socket.timeout:
            pass
        final = dict((a.rsplit("/", 1)[1], v) for a, v in tail)
        self.assertEqual(final["PBG_SmileSad"], 0.0)
        self.assertEqual(final["PBG_Mood02"], 1.0)


def decode_osc_any(packet):
    """-> (address, value) for a single float or bool OSC message."""
    end = packet.index(b"\0")
    address = packet[:end].decode()
    i = (end + 4) & ~3
    tag = packet[i:i + 2]
    if tag == b",f":
        return address, struct.unpack(">f", packet[i + 4:i + 8])[0]
    assert tag in (b",T", b",F"), packet
    return address, tag == b",T"


class MultiTermMergedTests(unittest.TestCase):
    def test_weighted_sum_and_neutral(self):
        from palbuddy.config import MergedParam, parse_terms
        # EyeLidExpandedSqueeze-like: 0.2*wide + 0.8*open - squeeze
        m = MergedParam("PBG_Lid", parse_terms("0.2*wide,0.8*open,-squeeze"), 0.0, 1.0)
        self.assertAlmostEqual(m.combined({"wide": 1, "open": 1}), 1.0)
        self.assertAlmostEqual(m.combined({"open": 1, "squeeze": 0.3}), 0.5)
        self.assertEqual(m.combined({"squeeze": 1}), -1.0)
        self.assertEqual(m.combined({"wide": 1, "open": 1, "squeeze": 0}), 1.0)
        self.assertEqual(m.combined({"wide": 5, "open": 5}), 1.0)  # clamped
        # custom neutral: -1 -> min, 0 -> neutral, +1 -> max (piecewise linear)
        n = MergedParam("PBG_N", parse_terms("smile,-sad"), 0.0, 1.0, 0.2)
        self.assertEqual(n.combine({}), 0.2)
        self.assertAlmostEqual(n.combine({"smile": 0.5}), 0.6)
        self.assertAlmostEqual(n.combine({"sad": 0.5}), 0.1)
        self.assertEqual(n.combine({"smile": 1}), 1.0)
        self.assertEqual(n.combine({"sad": 1}), 0.0)
        self.assertEqual(MergedParam.pair("PBG_M", "a", "b", 0, 2).neutral, 1.0)

    def test_binary_follows_the_output_value(self):
        from palbuddy.config import MergedParam, parse_terms
        m = MergedParam("PBG_S", parse_terms("smile,-sad"), -1.0, 1.0, 0.0)
        self.assertEqual(m.binary_input({"smile": 0.5}), (0.5, False))
        self.assertEqual(m.binary_input({"sad": 0.5}), (0.5, True))
        u = MergedParam("PBG_U", parse_terms("smile,-sad"), 0.0, 2.0, 0.5)
        self.assertEqual(u.binary_input({}), (0.25, False))  # neutral 0.5 of 0..2
        self.assertEqual(u.binary_input({"smile": 1}), (1.0, False))

    def test_parse_formula_and_validation(self):
        from palbuddy.config import MergedParam, merged_param_problems, parse_terms
        terms = parse_terms("smile, -sad, 0.2*wide, -0.5*squeeze")
        self.assertEqual([(t.cls, t.weight) for t in terms],
                         [("smile", 1.0), ("sad", -1.0), ("wide", 0.2), ("squeeze", -0.5)])
        self.assertEqual(MergedParam("X", terms).formula(), "smile - sad + 0.2*wide - 0.5*squeeze")
        names = {"smile", "sad", "wide", "squeeze"}
        self.assertEqual(merged_param_problems(MergedParam("PBG_X", terms), names), [])
        self.assertTrue(merged_param_problems(MergedParam("PBG_X", parse_terms("smile"), 0, 1, 2.0), names))  # neutral
        self.assertTrue(merged_param_problems(MergedParam("PBG_X", parse_terms("20*smile")), names))  # weight
        self.assertTrue(merged_param_problems(MergedParam("PBG_X", parse_terms("0*smile")), names))  # all zero

    def test_old_positive_negative_config_still_loads(self):
        cfg = Config.from_dict({"classes": [{"name": "smile"}, {"name": "sad"}], "merged_params": [
            {"name": "PBG_Old", "positive": "smile", "negative": "sad", "out_min": 0.0, "out_max": 2.0}]})
        m = cfg.merged_params[0]
        self.assertEqual([(t.cls, t.weight) for t in m.terms], [("smile", 1.0), ("sad", -1.0)])
        self.assertEqual(m.combine({"smile": 1}), 2.0)
        with tempfile.TemporaryDirectory() as d:  # and round-trips in the new form
            path = os.path.join(d, "c.json")
            cfg.save(path)
            self.assertEqual(Config.load(path).merged_params[0], m)


class OscOutputTests(unittest.TestCase):
    def test_every_vrcft_name_is_refused(self):
        from palbuddy.vrcft_names import VRCFT_BINARY_PARAMETERS, VRCFT_PARAMETERS, vrcft_conflict
        for n in VRCFT_PARAMETERS:
            self.assertIsNotNone(vrcft_conflict(n), n)
            self.assertIsNotNone(vrcft_conflict("Custom/" + n), n)  # VRCFT matches "/<name>" endings
        for n in VRCFT_BINARY_PARAMETERS:
            self.assertIsNotNone(vrcft_conflict(n + "4"), n)
        self.assertIsNotNone(vrcft_conflict("jawopen"))  # case-insensitive, to be safe
        for ok in ("PBG_SmileSad", "PBG/Happy", "MyFrown", "PBG_Mood2"):
            self.assertIsNone(vrcft_conflict(ok), ok)

    def test_output_names_and_problems(self):
        from palbuddy.config import MergedParam, class_output_problems, merged_param_problems, osc_output_names
        self.assertEqual(osc_output_names("X", "binary", 3, True), ["X1", "X2", "X4", "XNegative"])
        self.assertEqual(osc_output_names("X", "both", 2, False), ["X", "X1", "X2"])
        self.assertTrue(class_output_problems(ExpressionClass("a", osc_name="JawOpen")))
        self.assertTrue(class_output_problems(ExpressionClass("a", osc_name="PBG/SmileSad")))
        self.assertTrue(class_output_problems(ExpressionClass("a", osc_name="PBG_A", osc_format="binary", osc_bits=9)))
        self.assertEqual(class_output_problems(ExpressionClass("a", osc_name="PBG_A", osc_format="binary")), [])
        # a clash can come from a generated binary name only: "MouthX" + bit -> VRCFT binary MouthX<n>
        self.assertTrue(merged_param_problems(MergedParam.pair("Mouth", "a", None, 0, 1, osc_format="float"), {"a"}) == [])
        self.assertTrue(merged_param_problems(MergedParam.pair("SmileSad", "a"), {"a"}))
        cfg = Config(classes=[ExpressionClass("a", osc_name="PBG_Dup")],
                     merged_params=[MergedParam.pair("PBG_Dup", "a")])
        self.assertTrue(any("more than one output" in p for p in cfg.output_problems()))

    def test_binary_matches_vrcft(self):
        from palbuddy.config import binary_bits
        # VRCFaceTracking BinaryBaseParameter: bit i of (int)(v * 2^N); v > 0.99999 -> all set
        self.assertEqual(binary_bits(0.5, 4), [False, False, False, True])
        self.assertEqual(binary_bits(0.26, 4), [False, False, True, False])
        self.assertEqual(binary_bits(0.99, 4), [True] * 4)
        self.assertEqual(binary_bits(1.0, 3), [True] * 3)
        self.assertEqual(binary_bits(0.0, 5), [False] * 5)
        self.assertEqual(binary_bits(-2, 2), [False, False])

    def test_sensitivity_remap(self):
        c = ExpressionClass("a", in_min=0.2, in_max=0.8)
        self.assertEqual(c.remap(0.2), 0.0)
        self.assertAlmostEqual(c.remap(0.5), 0.5)
        self.assertEqual(c.remap(0.9), 1.0)
        self.assertEqual(ExpressionClass("b").remap(0.37), 0.37)

    def test_engine_sends_class_and_binary_outputs(self):
        from palbuddy.config import MergedParam
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.bind(("127.0.0.1", 0))
        rx.settimeout(0.5)
        self.addCleanup(rx.close)
        raw = {"value": np.array([0.9, 0.0, 0.0], np.float32)}

        class FakeRuntime:
            def activate(self): pass
            def release(self): pass
            def warmup(self): pass
            def describe(self): return "fake"
            def predict(self, sample): return raw["value"]

        cfg = Config(face_port=0, eye_port=0, vrcft_port=0, osc_port=rx.getsockname()[1],
                     classes=[ExpressionClass("neutral"),
                              ExpressionClass("smile", max_power=1.0, in_min=0.2, in_max=0.8,
                                              osc_name="PBG_Smile", osc_format="both", osc_bits=4),
                              ExpressionClass("sad", max_power=1.0, osc_name="JawOpen")],  # refused: VRCFT name
                     merged_params=[MergedParam.pair("PBG_SS", "smile", "sad", -1, 1, osc_format="binary", osc_bits=3),
                                    MergedParam.pair("PBG_Pos", "smile", "sad", 0, 2, osc_format="binary", osc_bits=2)])
        self.assertTrue(cfg.output_problems())  # the JawOpen output is reported...
        engine = Engine(cfg).start()
        self.addCleanup(engine.stop)
        engine._make_runtime = lambda: FakeRuntime()
        engine.start_inference()

        def collect(seconds):
            state, end = {}, time.time() + seconds
            while time.time() < end:
                engine.hub.push(neural(1.0), neural(1.0))
                try:
                    a, v = decode_osc_any(rx.recv(256))
                    state[a.rsplit("/", 1)[1]] = v
                except socket.timeout:
                    pass
            return state

        raw["value"] = np.array([0.9, 0.6, 0.0], np.float32)  # smile 0.6 -> sensitivity -> 0.667
        st = collect(1.5)
        self.assertNotIn("JawOpen", st)  # ...and never sent
        self.assertAlmostEqual(st["PBG_Smile"], 0.4 / 0.6, places=4)
        self.assertEqual([st["PBG_Smile%d" % b] for b in (1, 2, 4, 8)], [False, True, False, True])  # int(10.67)=10
        # merged -1..1 binary: sign + magnitude; 0..2 binary: position in range
        self.assertEqual([st["PBG_SS%d" % b] for b in (1, 2, 4)], [True, False, True])  # int(0.667*8)=5
        self.assertFalse(st["PBG_SSNegative"])
        self.assertEqual([st["PBG_Pos%d" % b] for b in (1, 2)], [True, True])  # int(0.833*4)=3
        raw["value"] = np.array([0.9, 0.0, 0.5], np.float32)  # sad 0.5 -> c = -0.5
        st = collect(1.5)
        self.assertTrue(st["PBG_SSNegative"])
        self.assertEqual([st["PBG_SS%d" % b] for b in (1, 2, 4)], [False, False, True])  # int(0.5*8)=4
        engine.stop_inference()
        st = collect(0.8)
        self.assertEqual(st["PBG_Smile"], 0.0)
        self.assertEqual([st["PBG_Smile%d" % b] for b in (1, 2, 4, 8)], [False] * 4)
        self.assertEqual([st["PBG_SS%d" % b] for b in (1, 2, 4)], [False] * 3)
        self.assertFalse(st["PBG_SSNegative"])
        self.assertEqual([st["PBG_Pos%d" % b] for b in (1, 2)], [False, True])  # neutral = middle (0.5 -> 2)


class LightweightTests(unittest.TestCase):
    """Optimisations must not change behaviour."""

    def test_idle_rate_until_something_receives_the_output(self):
        class FakeRuntime:
            def activate(self): pass
            def release(self): pass
            def warmup(self): pass
            def describe(self): return "fake"
            def predict(self, sample): return np.array([0.1, 0.5], np.float32)

        cfg = Config(face_port=0, eye_port=0, vrcft_port=0,
                     classes=[ExpressionClass("n"), ExpressionClass("s", target="JawOpen")])
        engine = Engine(cfg).start()
        self.addCleanup(engine.stop)
        engine._make_runtime = lambda: FakeRuntime()
        stop = threading.Event()

        def feed():
            while not stop.is_set():
                engine.hub.push(neural(1.0), neural(1.0))
                time.sleep(1 / 60)
        threading.Thread(target=feed, daemon=True).start()
        self.addCleanup(stop.set)
        engine.start_inference()
        time.sleep(2.5)
        self.assertTrue(engine.status()["idle"])
        self.assertLess(engine.infer_rate.rate(), 13)  # ~10 Hz while nothing listens
        self.assertGreater(engine.infer_rate.rate(), 6)
        self.assertIsNotNone(engine.last_out.get(1))  # the GUI still gets values
        engine.request_full_rate(2.5)  # e.g. the sensitivity measurement
        time.sleep(2.2)
        self.assertGreater(engine.infer_rate.rate(), 40)
        self.assertFalse(engine.status()["idle"])
        time.sleep(2.5)
        self.assertTrue(engine.status()["idle"])  # back to idle afterwards
        vr = socket.create_connection(("127.0.0.1", engine.vrcft.port))  # VRCFT module connects
        self.addCleanup(vr.close)
        vr.sendall(b"PBG2\x02")
        self.assertTrue(wait_for(lambda: not engine.status()["idle"], timeout=3))
        time.sleep(2.2)
        self.assertGreater(engine.infer_rate.rate(), 45)  # full rate

    def test_camera_frames_only_kept_while_shown(self):
        hub = FrameHub()
        rx = SRanipalReceiver(hub, "127.0.0.1", 0, 0).start()
        self.addCleanup(rx.stop)
        eye = socket.create_connection(("127.0.0.1", rx.ports[1]))
        self.addCleanup(eye.close)
        cam = packet(np.full(20000, 0.5, np.float32).tobytes())
        eye.sendall(cam + packet(neural(1.0)))
        self.assertTrue(wait_for(lambda: rx.status["port1"].last_frame > 0))
        self.assertIsNone(hub.cameras["eye"])  # nobody shows a preview: not copied
        hub.want_cameras()
        eye.sendall(cam)
        self.assertTrue(wait_for(lambda: hub.cameras["eye"] is not None))

    def test_normalize_without_numpy_matches(self):
        for raw, mp in ((0.45, 0.9), (5.0, 0.9), (-1.0, 0.9), (np.float32(0.3), 0.5)):
            self.assertAlmostEqual(normalize(raw, mp), float(np.clip(raw / mp * 2 - 1, -1, 1)), places=6)


class ConfigTests(unittest.TestCase):
    def test_roundtrip(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "config.json")
            cfg.save(p)
            cfg2 = Config.load(p)
        self.assertEqual(cfg2.to_dict(), cfg.to_dict())
        # same mapping as the original to_replace table
        self.assertEqual({i: t for i, t, _ in cfg.targets()},
                         {2: "JawRight", 1: "JawLeft", 4: "MouthUpperUpLeft", 5: "MouthUpperUpRight",
                          7: "JawForward", 3: "MouthPout"})
        cfg.classes[1].target = "BrowLowererLeft"  # Unified Expression names are valid too
        self.assertEqual(cfg.validate(), [])
        cfg.classes[1].target = "NotAShape"
        self.assertEqual(len(cfg.validate()), 1)
        cfg.classes[1].target = "JawLeft"
        self.assertEqual(cfg.validate(), [])


class CompactModelAndCompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _cfg(self, **kw):
        make_recordings(self.tmp.name, {"n": 0.0, "s": 1.0, "m": -1.0}, frames=64)
        opts = dict(dataset_folder=self.tmp.name, epochs=2, batch_size=32, mixed_precision=False,
                    validation_split=0.2, model_path=os.path.join(self.tmp.name, "m.pt"),
                    face_port=0, eye_port=0, vrcft_port=0, infer_device="cpu",
                    classes=[ExpressionClass("neutral", ["n.mmap"]), ExpressionClass("smile", ["s.mmap"], "JawOpen"),
                             ExpressionClass("mad", ["m.mmap"], "MouthPout")])
        opts.update(kw)
        return Config(**opts)

    def test_compact_sigmoid_roundtrip(self):
        from palbuddy.model import model_size
        m = BuddyNet(3, "face", "compact", "sigmoid")
        self.assertLess(sum(p.numel() for p in m.parameters()) * 25, sum(p.numel() for p in BuddyNet(3).parameters()))
        self.assertLess(model_size("compact")[1] * 8, model_size("standard")[1])
        m.set_normalization(np.arange(64) * 0.1, np.full(64, 2.0))
        m.eval()
        x = torch.randn(2, 64, 20, 20)
        y = m(x)
        self.assertTrue(((y >= 0) & (y <= 1)).all())
        torch.testing.assert_close(torch.sigmoid(m(x, logits=True)), y)
        path = os.path.join(self.tmp.name, "c.pt")
        m.save(path, ["a", "b", "c"])
        loaded = BuddyNet.load(path, expected_outputs=3, expected_mode="face").eval()
        self.assertEqual((loaded.arch, loaded.output, loaded.input_mode), ("compact", "sigmoid", "face"))
        torch.testing.assert_close(loaded(x), y)
        # the normalisation is part of the model: different inputs statistics -> different outputs
        loaded.set_normalization(np.zeros(64), np.ones(64))
        self.assertFalse(torch.allclose(loaded(x), y))

    def test_old_checkpoints_still_load_as_relu(self):
        path = os.path.join(self.tmp.name, "old.pt")
        std = BuddyNet(2)
        torch.save({k: getattr(std, k).state_dict() for k in ("conv1", "conv2", "linear1", "linear2")}, path)
        m = BuddyNet.load(path)
        self.assertEqual((m.arch, m.output), ("standard", "relu"))

    @unittest.skipUnless(__import__("importlib").util.find_spec("onnxruntime"), "onnxruntime not installed")
    def test_compact_onnx_matches_torch(self):
        from palbuddy.onnx_export import export_onnx
        from palbuddy.onnx_runtime import OnnxRuntime, read_meta
        m = BuddyNet(3, "both", "compact", "sigmoid")
        m.set_normalization(np.linspace(0, 1, 128), np.linspace(0.5, 2, 128))
        m.eval()
        base = os.path.join(self.tmp.name, "c")
        paths = export_onnx(m, base)
        self.assertEqual(read_meta(base)["output"], "sigmoid")
        x = np.random.default_rng(2).random((1, 128, 20, 20), dtype=np.float32)
        with torch.no_grad():
            ref = m(torch.from_numpy(x))[0].numpy()
        np.testing.assert_allclose(OnnxRuntime(base, "cpu", 1, int8=False).predict_array(x), ref, atol=1e-5)
        if os.path.exists(paths["int8"]):
            self.assertLess(np.abs(OnnxRuntime(base, "cpu", 1, int8=True).predict_array(x) - ref).max(), 0.05)

    def test_train_compact_bce(self):
        cfg = self._cfg(model_arch="compact", loss="bce", epochs=3)
        model, history = train(cfg, log_fn=lambda *a: None, device=torch.device("cpu"))
        self.assertEqual((model.arch, model.output), ("compact", "sigmoid"))
        self.assertTrue(model.in_scale.ne(1).any())  # input statistics were computed
        mt = model.val_metrics
        for key in ("val_acc", "val_hit", "val_false", "val_score", "epoch", "per_class"):
            self.assertIn(key, mt)
        self.assertGreater(mt["val_acc"], 0.9)
        self.assertLess(mt["val_false"], 0.2)
        with self.assertRaises(ValueError):  # a sigmoid model can't continue with MSE
            train(cfg, log_fn=lambda *a: None, device=torch.device("cpu"), init_model=model, loss="mse")

    def test_evaluate_metrics(self):
        from palbuddy.trainer import evaluate

        class Fixed(torch.nn.Module):
            def __init__(self, out):
                super().__init__()
                self.out = torch.tensor(out)

            def forward(self, x):
                return self.out[x[:, 0, 0, 0].long()]

        # frame 0: class 0 shown, clean; frame 1: class 1 shown, weak (0.4) and class 0 also at 0.35
        out = [[0.9, 0.1], [0.35, 0.4]]
        x = np.zeros((2, 1, 1, 1), np.float32)
        x[1] = 1
        m = evaluate(Fixed(out), x, np.array([0, 1]), 2, torch.device("cpu"))
        self.assertEqual((m["acc"], m["hit"], m["false"]), (1.0, 0.5, 0.5))
        self.assertAlmostEqual(m["err"], (0.1 + 0.6) / 2, places=6)
        self.assertAlmostEqual(m["score"], 100 * (1 + 0.5 + 2 * 0.5 + (1 - 0.35)) / 5, places=4)
        # soft targets (guided recording): frame 1 shows class 1 at 40% -> no "clear" frame for hit,
        # and class 0 (target 0.6) outputting 0.35 is not a false activation
        m = evaluate(Fixed(out), x, np.array([[1.0, 0.0], [0.6, 0.4]], np.float32), 2, torch.device("cpu"))
        self.assertEqual((m["hit"], m["false"]), (1.0, 0.0))

    def test_recommend_prefers_fast_among_equals(self):
        from palbuddy.trainer import recommend
        r = lambda score, ms: {"metrics": {"val_score": score}, "ms": ms, "params": 1}
        self.assertEqual(recommend([r(90.0, 5.0), r(89.8, 0.5), r(80.0, 0.1)]), 1)
        self.assertEqual(recommend([r(90.0, 5.0), r(88.0, 0.5)]), 0)
        self.assertIsNone(recommend([]))

    def test_config_rejects_unknown_loss_and_arch(self):
        cfg = self._cfg(loss="huber")
        self.assertTrue(any("loss" in p for p in cfg.validate()))
        cfg.loss, cfg.model_arch = "bce", "huge"
        self.assertTrue(any("model_arch" in p for p in cfg.validate()))
        cfg.model_arch = "compact"
        self.assertEqual(cfg.validate(), [])

    def test_engine_compare_and_apply(self):
        cfg = self._cfg(epochs=1)
        engine = Engine(cfg)
        engine.device = torch.device("cpu")
        engine.start()
        self.addCleanup(engine.stop)
        engine.model = BuddyNet(3)
        engine.save_model()  # an existing model: kept as .prev.pt when another one is applied
        engine.start_inference()
        self.assertTrue(wait_for(lambda: engine.infer_backend is not None))
        done = threading.Event()
        out = {}
        seen = []
        engine.compare_async([("lite", "mse"), ("compact", "bce")], on_progress=lambda **i: seen.append(i),
                             on_done=lambda ok, m, r, b: (out.update(ok=ok, m=m, r=r, b=b), done.set()))
        self.assertFalse(engine.inferring)  # paused while comparing
        self.assertTrue(done.wait(300), "comparison did not finish")
        self.assertTrue(out["ok"], out["m"])
        self.assertEqual([(r["arch"], r["loss"]) for r in out["r"]], [("lite", "mse"), ("compact", "bce")])
        self.assertEqual({i["candidate"] for i in seen}, {0, 1})
        for r in out["r"]:
            self.assertGreater(r["ms"], 0)
            self.assertGreater(r["size_mb"], 0)
        self.assertLess(out["r"][1]["params"], out["r"][0]["params"])
        self.assertIn(out["b"], (0, 1))

        engine.start_inference()
        engine.apply_candidate(out["r"][1])
        self.assertEqual((cfg.model_arch, cfg.loss), ("compact", "bce"))
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "m.prev.pt")))
        self.assertEqual(BuddyNet.load(cfg.model_path).arch, "compact")
        self.assertTrue(engine.inferring)  # tracking resumed with the new model
        self.assertTrue(wait_for(lambda: engine.infer_backend is not None))
        engine.stop_inference()

    def test_softener(self):
        from palbuddy.engine import softener
        self.assertIsNone(softener(1.0))
        f = softener(3.0)
        self.assertEqual((f(0.0), f(0.005), f(1.0)), (0.0, 0.0, 1.0))  # neutral stays 0, full stays 1
        self.assertAlmostEqual(f(0.5), 0.5)
        vals = [f(p) for p in (0.05, 0.2, 0.5, 0.8, 0.95)]
        self.assertEqual(vals, sorted(vals))
        self.assertGreater(f(0.05), 0.05)  # the steep ends of the sigmoid are spread out
        self.assertLess(f(0.95), 0.95)

    def test_stabilizer_ignores_speech_flicker_but_follows_held_expressions(self):
        from palbuddy.engine import Stabilizer
        st, t, out = Stabilizer(), 0.0, []
        dt = 1 / 60
        for i in range(60):  # 1 s neutral with a 150 ms flicker to 1.0 in the middle (a syllable)
            t += dt
            out.append(st.update(1.0 if 25 <= i < 34 else 0.0, t, 1.0))
        self.assertLess(max(out), 0.05)
        for _ in range(60):  # then the expression is held: reached within ~0.6 s
            t += dt
            v = st.update(1.0, t, 1.0)
        self.assertGreater(v, 0.9)
        # the same flicker passes unfiltered-ish with a weak setting
        weak, t = Stabilizer(), 0.0
        peak = 0.0
        for i in range(60):
            t += dt
            peak = max(peak, weak.update(1.0 if 25 <= i < 34 else 0.0, t, 0.1))
        self.assertGreater(peak, 0.5)

    def test_runtimes_report_output_kind(self):
        from palbuddy.inference import Runtime
        self.assertEqual(Runtime(BuddyNet(2, arch="compact", output="sigmoid"), "cpu").output, "sigmoid")
        self.assertEqual(Runtime(BuddyNet(2, arch="lite"), "cpu").output, "relu")

    def test_compare_needs_validation_frames(self):
        from palbuddy.trainer import compare
        cfg = self._cfg(validation_split=0.0)
        with self.assertRaises(ValueError):
            compare(cfg, [("lite", "mse")], log_fn=lambda *a: None, device=torch.device("cpu"))


class IntensityLearningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_guide_script(self):
        from palbuddy.datasets import GUIDE_DURATION, guide_target
        self.assertEqual(guide_target(0.5)[:2], (0.0, "neutral"))
        self.assertAlmostEqual(guide_target(2.5)[0], 0.5)  # halfway up the first ramp
        self.assertEqual(guide_target(4.0)[:2], (1.0, "hold"))
        self.assertEqual(guide_target(GUIDE_DURATION + 1)[1], "done")
        vals = [guide_target(t / 10)[0] for t in range(int(GUIDE_DURATION * 10))]
        self.assertEqual((min(vals), max(vals)), (0.0, 1.0))
        self.assertTrue(any(abs(v - 0.5) < 1e-6 for v in vals))  # half-strength holds

    def test_guided_recording_writes_labels(self):
        from palbuddy import datasets as D
        script = ((0.3, 0.0, 0.0, "neutral"), (0.4, 0.0, 1.0, "up"), (0.3, 1.0, 1.0, "hold"))
        old = D.GUIDE_SCRIPT, D.GUIDE_DURATION, D.GUIDE_LAG
        D.GUIDE_SCRIPT, D.GUIDE_DURATION, D.GUIDE_LAG = script, 1.0, 0.0
        self.addCleanup(lambda: setattr(D, "GUIDE_SCRIPT", old[0]) or setattr(D, "GUIDE_DURATION", old[1])
                        or setattr(D, "GUIDE_LAG", old[2]))
        hub = FrameHub()
        path = os.path.join(self.tmp.name, "smile-em.mmap")
        done, result, cues = threading.Event(), {}, []
        D.Recorder(hub, path, 0, countdown=0, guided=True, on_guide=lambda v, cue, left: cues.append(cue),
                   on_done=lambda ok, m: (result.update(ok=ok, m=m), done.set())).start()
        stop = time.time() + 1.3
        while time.time() < stop and not done.is_set():
            hub.push(neural(1.0), neural(1.0))
            time.sleep(1 / 200)
        self.assertTrue(done.wait(5))
        self.assertTrue(result["ok"], result["m"])
        n = frame_count(path)
        self.assertGreater(n, 64)
        labels = D.load_labels(path, n)
        self.assertEqual(len(labels), n)
        self.assertEqual(labels[0], 0.0)
        self.assertGreater(labels.max(), 0.9)
        self.assertIn("up", cues)
        self.assertIsNone(D.load_labels(path, n + 1))  # mismatched labels are ignored
        D.remove_recording(path)
        self.assertFalse(os.path.exists(D.labels_path(path)))

    def test_recording_names_never_overwrite(self):
        from palbuddy.engine import unique_recording_name
        cfg = Config(dataset_folder=self.tmp.name)
        self.assertEqual(unique_recording_name(cfg, "smile"), "smile-em.mmap")
        open(os.path.join(self.tmp.name, "smile-em.mmap"), "wb").close()
        self.assertEqual(unique_recording_name(cfg, "smile"), "smile-2-em.mmap")
        open(os.path.join(self.tmp.name, "smile-2-em.mmap.labels.json"), "w").close()  # labels count as taken
        self.assertEqual(unique_recording_name(cfg, "smile"), "smile-3-em.mmap")
        open(os.path.join(self.tmp.name, "raw.mmap"), "wb").close()
        self.assertEqual(unique_recording_name(cfg, "raw.mmap"), "raw-2.mmap")

    def test_soft_targets_and_mixup(self):
        from palbuddy.trainer import BatchSampler, targets_for
        np.testing.assert_allclose(targets_for(1, np.array([0.25], np.float32), 3), [[0.75, 0.25, 0.0]])
        np.testing.assert_allclose(targets_for(0, None, 3), [[1, 0, 0]])
        neutral = np.zeros((20, 128, 20, 20), np.float32)
        smile = np.ones((20, 128, 20, 20), np.float32)
        lab = np.linspace(0, 1, 20).astype(np.float32)
        plain = BatchSampler([[neutral], [smile]], 256, seed=0)
        x, t = plain.sample()
        self.assertTrue(set(np.unique(t)) <= {0.0, 1.0})
        guided = BatchSampler([[neutral], [smile]], 256, seed=0, labels=[[None], [lab]])
        x, t = guided.sample()
        np.testing.assert_allclose(t.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue(((t > 0.01) & (t < 0.99)).any())
        mixed = BatchSampler([[neutral], [smile]], 512, seed=0, mixup=0.5)
        x, t = mixed.sample()
        blended = (t[:, 1] > 0.01) & (t[:, 1] < 0.99)
        self.assertGreater(blended.sum(), 50)
        # inputs are blended with exactly the target's weight (neutral frames are 0, smile frames 1)
        np.testing.assert_allclose(x[blended, 0, 0, 0], t[blended, 1], atol=1e-6)

    def test_training_uses_guided_labels(self):
        import json
        from palbuddy.datasets import labels_path
        make_recordings(self.tmp.name, {"n": 0.0, "s": 1.0}, frames=64)
        with open(labels_path(os.path.join(self.tmp.name, "s.mmap")), "w") as f:
            json.dump({"intensity": np.linspace(0, 1, 64).round(3).tolist()}, f)
        cfg = Config(dataset_folder=self.tmp.name, epochs=2, batch_size=32, mixed_precision=False,
                     model_arch="compact", loss="bce", validation_split=0.2,
                     classes=[ExpressionClass("neutral", ["n.mmap"]), ExpressionClass("smile", ["s.mmap"], "JawOpen")])
        logs = []
        model, _ = train(cfg, log_fn=logs.append, device=torch.device("cpu"), seed=0)
        self.assertTrue(any("1 guided" in line for line in logs), logs[:3])
        self.assertIn("val_err", model.val_metrics)

    def test_one_euro_smoothing(self):
        from palbuddy.engine import OneEuro, smoothing_cutoff
        self.assertIsNone(smoothing_cutoff(0.0))
        self.assertGreater(smoothing_cutoff(0.2), smoothing_cutoff(0.8))
        rng = np.random.default_rng(0)
        f, t, out = OneEuro(), 0.0, []
        for _ in range(120):  # 2 s of jitter around 0.3
            t += 1 / 60
            out.append(f.update(0.3 + rng.normal(0, 0.05), t, smoothing_cutoff(0.6)))
        self.assertLess(np.std(out[30:]), 0.02)
        for _ in range(12):  # a real, fast movement to 1.0 comes through within 0.2 s
            t += 1 / 60
            v = f.update(1.0, t, smoothing_cutoff(0.6))
        self.assertGreater(v, 0.85)


class Float16RecordingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def record(self, name, float16, frames=20):
        hub = FrameHub()
        path = os.path.join(self.tmp.name, name)
        done, result = threading.Event(), {}
        Recorder(hub, path, frames, countdown=0, float16=float16,
                 on_done=lambda ok, m: (result.update(ok=ok, m=m), done.set())).start()
        time.sleep(0.2)
        for i in range(frames + 10):
            hub.push(neural(i * 0.25), neural(1e6 if i == 5 else -i * 0.25))
        self.assertTrue(done.wait(5))
        self.assertTrue(result["ok"], result["m"])
        return path

    def test_float16_is_half_size_and_reads_back(self):
        from palbuddy.datasets import F16_MAGIC, HEADER_SIZE, SAMPLE_BYTES, recording_layout
        p16 = self.record("h-em.mmap", True)
        p32 = self.record("f-em.mmap", False)
        self.assertEqual(os.path.getsize(p32), 20 * SAMPLE_BYTES)  # legacy layout, no header
        self.assertEqual(os.path.getsize(p16), HEADER_SIZE + 20 * SAMPLE_BYTES // 2)
        with open(p16, "rb") as f:
            self.assertEqual(f.read(len(F16_MAGIC)), F16_MAGIC)
        self.assertEqual(recording_layout(p16)[0], np.float16)
        self.assertEqual(recording_layout(p32)[0], np.float32)
        for p in (p16, p32):
            self.assertEqual(frame_count(p), 20)
            rec = open_recording(p)
            self.assertEqual(float(rec[3, 0, 0, 0]), 0.75)
            self.assertEqual(float(rec[3, 64, 0, 0]), -0.75)
        # values outside the float16 range are clipped, never stored as inf
        self.assertTrue(np.isfinite(open_recording(p16)[5]).all())
        self.assertEqual(open_recording(p16, in_ram=True).dtype, np.float16)  # RAM cache halves too

    def test_training_mixes_float16_and_legacy_recordings(self):
        from palbuddy.datasets import create_recording
        from palbuddy.trainer import BatchSampler, load_recordings, split_recordings
        rng = np.random.default_rng(0)
        for name, offset, f16 in (("n", 0.0, True), ("s", 1.0, False)):
            mm = create_recording(os.path.join(self.tmp.name, name + ".mmap"), 32, f16)
            mm[:] = rng.normal(offset, 0.1, size=mm.shape)
            mm.flush()
            del mm
        cfg = Config(dataset_folder=self.tmp.name, model_path=os.path.join(self.tmp.name, "m.pt"),
                     epochs=2, batch_size=16, mixed_precision=False, validation_split=0.2,
                     classes=[ExpressionClass("neutral", ["n.mmap"]), ExpressionClass("smile", ["s.mmap"], "JawOpen")])
        recs, labels, _ = load_recordings(cfg, log_fn=lambda *a: None)
        self.assertEqual(recs[0][0].dtype, np.float16)
        _, _, val_x, _ = split_recordings(recs, 0.2, slice(0, 128), labels)
        self.assertEqual(val_x.dtype, np.float32)  # the model never sees float16 input
        x, t = BatchSampler(recs, 64, seed=0).sample()
        self.assertEqual(x.dtype, np.float32)
        for c, mean in ((0, 0.0), (1, 1.0)):  # each class's frames keep their values
            self.assertAlmostEqual(float(x[t[:, c] == 1].mean()), mean, delta=0.05)
        model, history = train(cfg, log_fn=lambda *a: None, device=torch.device("cpu"), seed=0)
        self.assertEqual(len(history), 2)
        self.assertIn("val_acc", model.val_metrics)

    def test_legacy_pickles_convert_to_float16(self):
        import pickle
        from palbuddy.datasets import convert_legacy_pickles, recording_layout
        block = [(np.full((64, 20, 20), i, np.float32), np.full((64, 20, 20), -i, np.float32)) for i in range(4)]
        with open(os.path.join(self.tmp.name, "old-em.pkl"), "wb") as f:
            pickle.dump(block, f)
        out = convert_legacy_pickles(self.tmp.name)
        self.assertEqual(len(out), 1)
        self.assertEqual(recording_layout(out[0])[0], np.float16)
        rec = open_recording(out[0])
        self.assertEqual((len(rec), float(rec[2, 0, 0, 0]), float(rec[2, 100, 0, 0])), (4, 2.0, -2.0))


class SessionValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_split_holds_out_whole_recordings(self):
        from palbuddy.trainer import session_holdout, split_recordings
        # each frame's value is (recording id, frame index) so held-out frames can be traced
        def rec(rid, n=100):
            r = np.zeros((n, 128, 20, 20), np.float32)
            r[:, 0, 0, 0] = rid
            r[:, 1, 0, 0] = np.arange(n)
            return r
        recordings = [[rec(0), rec(1)], [rec(2)]]  # neutral: two sessions, smile: one
        holdout = session_holdout(recordings)
        self.assertEqual(holdout, {(0, 1)})
        train_recs, _, val_x, val_t = split_recordings(recordings, 0.1, slice(0, 128), holdout=holdout)
        # neutral keeps its first recording whole and loses the second entirely
        self.assertEqual([len(r) for r in train_recs[0]], [100])
        self.assertEqual(float(train_recs[0][0][0, 0, 0, 0]), 0.0)
        # smile has only one recording: its end is held out as before
        self.assertEqual([len(r) for r in train_recs[1]], [90])
        ids = val_x[:, 0, 0, 0]
        self.assertEqual(set(ids.tolist()), {1.0, 2.0})
        held = val_x[ids == 1.0, 1, 0, 0]
        self.assertEqual((held.min(), held.max(), len(held)), (0.0, 99.0, 100))  # spread over the whole file
        self.assertTrue((val_t[ids == 1.0, 0] == 1).all() and (val_t[ids == 2.0, 1] == 1).all())
        self.assertTrue((val_x[ids == 2.0, 1, 0, 0] >= 90).all())
        # without a hold-out the old tail split is unchanged
        train_recs, _, val_x, _ = split_recordings(recordings, 0.1, slice(0, 128))
        self.assertEqual([len(r) for r in train_recs[0]], [90, 90])
        self.assertEqual(set(val_x[:, 0, 0, 0].tolist()), {0.0, 1.0, 2.0})

    def test_training_with_session_validation(self):
        rng = np.random.default_rng(0)
        for name, offset in (("n1", 0.0), ("n2", 0.0), ("s1", 1.0), ("s2", 1.0)):
            mm = np.memmap(os.path.join(self.tmp.name, name + ".mmap"), np.float32, "w+", shape=(32, 128, 20, 20))
            mm[:] = rng.normal(offset, 0.1, size=mm.shape)
            mm.flush()
            del mm
        cfg = Config(dataset_folder=self.tmp.name, model_path=os.path.join(self.tmp.name, "m.pt"),
                     epochs=2, batch_size=16, mixed_precision=False, validation_sessions=True,
                     classes=[ExpressionClass("neutral", ["n1.mmap", "n2.mmap"]),
                              ExpressionClass("smile", ["s1.mmap", "s2.mmap"], "JawOpen")])
        logs = []
        model, history = train(cfg, log_fn=logs.append, device=torch.device("cpu"), seed=0)
        self.assertEqual(len(history), 2)
        self.assertIn("val_acc", model.val_metrics)
        self.assertTrue(any("n2.mmap (neutral), s2.mmap (smile)" in line for line in logs), logs[:4])
        self.assertTrue(any("Holding out 64 frames" in line for line in logs), logs[:4])
        # one recording per class: falls back to the recording ends and says so
        cfg.classes = [ExpressionClass("neutral", ["n1.mmap"]), ExpressionClass("smile", ["s1.mmap"], "JawOpen")]
        logs.clear()
        train(cfg, log_fn=logs.append, device=torch.device("cpu"), seed=0)
        self.assertTrue(any("No class has two recordings" in line for line in logs), logs[:4])


if __name__ == "__main__":
    unittest.main()
