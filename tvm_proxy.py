"""Optional standalone proxy (legacy two-process mode).

Not needed any more: `python -m palbuddy` receives the SRanipal streams itself
(source "direct" in config.json), which saves a TCP hop and a process.

Use this only if you want the receiver in a separate process, e.g. to keep it
running while restarting the trainer. Then set "source": "proxy" in
config.json. Commands: swap, image, status, exit.
"""

import logging
import sys
import threading
import time

from palbuddy.config import DEFAULT_CONFIG_PATH, Config
from palbuddy.frames import FrameHub, ProxyServer, SRanipalReceiver, decode_camera

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("tvm_proxy")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    cfg = Config.load(config_path)
    hub = FrameHub()
    receiver = SRanipalReceiver(hub, cfg.bind_host, cfg.face_port, cfg.eye_port, swapped=cfg.swapped,
                                mode=cfg.input_mode, stall_timeout=cfg.stall_timeout).start()
    server = ProxyServer(hub, cfg.bind_host, cfg.proxy_port).start()
    log.info("Serving samples on %s:%d", cfg.bind_host, cfg.proxy_port)

    state = {"image": True, "quit": False}

    def input_thread():
        while not state["quit"]:
            try:
                cmd = input("command (swap, image, status, exit): ").strip()
            except EOFError:
                cmd = "exit"
            if cmd == "swap":
                receiver.set_swapped(not receiver.swapped)
                cfg.swapped = receiver.swapped
                cfg.save(config_path)
                print("swapped = %s" % receiver.swapped)
            elif cmd == "image":
                state["image"] = not state["image"]
            elif cmd == "status":
                for role, st in receiver.snapshot(cfg.stall_timeout).items():
                    print("  %-5s %-12s %5.1f fps" % (role, st["state"], st["fps"]))
                print("  clients: %d" % server.clients)
            elif cmd == "exit":
                state["quit"] = True

    threading.Thread(target=input_thread, daemon=True).start()

    try:
        import cv2
        import numpy as np
    except ImportError:
        cv2 = None

    shown = False
    while not state["quit"]:
        if cv2 is None or not state["image"]:
            if shown:
                cv2.destroyAllWindows()
                shown = False
            time.sleep(0.1)
            continue
        hub.want_cameras()
        img = np.zeros((200, 200), dtype=np.uint8)
        if hub.cameras["eye"] is not None:
            img[:100] = decode_camera(hub.cameras["eye"])
        if hub.cameras["face"] is not None:
            img[100:] = decode_camera(hub.cameras["face"], flipped=True)
        try:
            cv2.imshow("PalBuddyGuy proxy", cv2.resize(img, (600, 600), interpolation=cv2.INTER_NEAREST))
            cv2.waitKey(33)
            shown = True
        except cv2.error as e:  # e.g. opencv-python-headless or no display
            log.warning("Camera window unavailable (%s); continuing without it", e)
            state["image"] = False

    receiver.stop()
    server.stop()


if __name__ == "__main__":
    main()
