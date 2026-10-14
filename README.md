# Pal Buddy Guy: The anipal's best friend
This project improves on the tracking of the Vive Pro Eye and the Vive Facial Tracker. You make an expression, record it, and train a small network that drives a VRCFaceTracking parameter from it.

It now comes with a GUI, and one process handles receiving data, recording, training, calibration and output.

> 한국어 전체 번역: [README.ko.md](README.ko.md) · 한국어 요약은 [아래](#한국어-빠른-시작)에 있습니다.

# System requirements
* An NVIDIA GPU with CUDA and at least 4 GB of VRAM is recommended. The code also runs on CPU (and Apple MPS), but training will be slow.
* Works with **both trackers, the facial tracker only, or the Pro Eye only**. See *Input mode* below.
* Python 3.8+ with PyTorch.

# Installation
1. Replace the tvm runtime inside SRanipal. Copy the two .DLL files from the `tvm runtime` folder into `C:\Program Files\VIVE\SRanipal`, replacing the existing files. Back up the old files first in case you want to revert.
2. **Double-click `PalBuddyGuy.bat`.** That's it.

On the first run the launcher:
* finds Python 3.9+, or offers to install it with winget,
* creates a private environment in `.venv`, so your system Python is left untouched,
* installs PyTorch (the CUDA build when an NVIDIA GPU is detected, otherwise the CPU build) and the other packages. This is a 2–3 GB download and takes a few minutes,
* offers to create a **Pal Buddy Guy** shortcut on the desktop.

After that, double-clicking the .bat or the shortcut opens the GUI directly, without a console window. Startup errors (such as a port already in use) appear in a dialog, and everything is logged to `palbuddy.log`. To reinstall from scratch, delete the `.venv` folder and run the .bat again.

# Running
Start Pal Buddy Guy (double-click) **before** SRanipalRuntime.

Manual / advanced use, with your own Python environment:

```
pip install -r requirements.txt   # install a GPU build of torch first: https://pytorch.org/get-started/locally/
python -m palbuddy          # GUI  (same as: python script.py)
python -m palbuddy --cli    # text commands, like the old script
python -m palbuddy --infer  # start tracking right away
```
Arguments given to the .bat are passed on as well, for example `PalBuddyGuy.bat --infer`.

You no longer need to start `tvm_proxy.py` separately, because the app receives the SRanipal streams itself. See *Legacy proxy mode* below if you still want a separate proxy.

All settings are saved in `config.json`, which is created on first start. You can edit them from the GUI; nothing has to be edited in the source code.

## 1. Check the streams (Live tab)
The status bar shows whether the eye tracker, face tracker and VRCFaceTracking are connected, and their frame rates.
The camera preview should show the **eyes on top and the face below**. If they're reversed, click **Swap eye/face**. The choice is saved.

### Input mode
The **Input** box on the Live tab selects which trackers feed the network:

| Input | Uses |
|---|---|
| Eye + face trackers | both feature maps (the original behaviour) |
| Face tracker only | only the facial tracker's features |
| Eye tracker only | only the Pro Eye's features |

* If the connected trackers don't match the selected mode, a suggestion appears. When only one tracker streams, check the preview and click *Face only* or *Eye only*. With a single tracker, it doesn't matter which SRanipal port its stream arrives on, so no swapping is needed.
* A model only works in the input mode it was trained in, and the mode is stored in the model file. After switching, train again.
* Recordings always store both halves, with a missing tracker saved as zeros. That means **recordings made with both trackers can be reused** for face-only or eye-only training. A recording made with one tracker can't be used for a mode that needs the other tracker, and training tells you which file is the problem.
* In every mode the outputs drive VRCFT *lip* shapes, because that's what the VRCFT module accepts. Normal eye tracking (gaze, blink) keeps working as before. In eye-only mode you can still train something like a frown seen by the eye cameras and map it to a lip shape.

## 2. Record (Record tab)
Enter a name and click **Record**. After the countdown, 2048 frames are captured (about 30 s) and written directly to `<dataset folder>/<name>-em.mmap`.
* You *must* also record a **neutral** set. It doesn't have to be a truly neutral face. It should cover any faces you **don't** want to track: talking, looking around, blinking. That way the network learns what *not* to fire on.
* While recording, keep the target expression dominant but move around a bit (adjust the headset, turn your head). The goal is diverse data where the target expression is the one constant.
* Each recording is about 200 MB (float16; recordings made before this change are ~400 MB float32 and still load), so point the dataset folder (Settings tab) at a drive with space. Set `"record_float16": false` in `config.json` to record in float32 again.
* Old `.pkl` recordings can be converted with **Convert old .pkl**.

## 3. Configure classes and train (Train tab)
Each class is one network output. The **first class must be neutral**. For each class, choose its recordings and, optionally, the SRanipal lip shape it should drive ("Drives shape"). You can also select recordings in the Record tab and click **Add to class…**.
Click **Train**. The loss curve should drop below the 0.001 line by the end. If it doesn't, something is wrong with the recordings or the class list. Then click **Save model**.

## 4. Track and calibrate (Live tab)
Click **Start tracking**. Each output is shown as its raw value and the value sent to VRCFT.
**FastCal** puppets each shape on your avatar one after another. Copy it with your face. The measured strengths become each class's *max power*, which you can also edit by hand in the Train tab. **Smoothing** reduces jitter at the cost of a little latency.

## 5. Inference performance (Settings tab → Inference performance)
Tracking runs the network on every new frame, about 60 times a second. These settings trade GPU, CPU, memory and latency against each other:

| Setting | What it does |
|---|---|
| **Engine** | *ONNX Runtime* is the fastest on the CPU and never loads PyTorch while tracking, which saves hundreds of MB of RAM. *auto* uses it when it's installed; the launcher installs it. When you save a model, an `.onnx` copy (plus an int8 version) is written next to the `.pt`, and it is refreshed automatically if the `.pt` is newer. |
| **Run on** | *CPU* (default) leaves the GPU entirely to VR; int8 inference costs about a third of one core. *auto* uses the GPU if there is one. With ONNX Runtime that means DirectML (`onnxruntime-directml`), which works on NVIDIA, **AMD** and Intel GPUs. *GPU* forces the GPU. |
| **CPU threads** | Default 1. More threads cut latency but **increase** total CPU use. |
| **int8 quantization** | On the CPU, uses int8 weights for the big linear layer: about 2–3× faster, with the same model and no retraining. The output difference is about 0.003. |
| **Max tracking rate** | For example 30 Hz. Skips frames in between and halves the work. Combine it with *Smoothing* on the Live tab. |

**Benchmark this PC** times every available backend with your model, so you can pick what works best on your machine. The Live tab shows the running backend, the time per frame and the whole process's CPU use.

### Model and training method (Train tab)
| Model | Weights | Compute | |
|---|---|---|---|
| *standard* | 27 M | 115 M multiply-adds | the original network |
| *lite* | 3.6 M | 33 M | same layout, fewer channels |
| *compact* | 0.9 M | 11 M | built for small datasets: input features standardised per channel (statistics from your recordings, stored in the model), 1×1 conv 128→64, small head |

97 % of the standard model's weights sit in one linear layer, which is a lot for a few minutes of recordings; smaller models overfit less, so they can be *more* accurate, not only faster. Which one is best depends on your recordings.

| Training method | |
|---|---|
| *MSE* (original) | ReLU outputs trained towards a one-hot target: the classes compete with each other. |
| *BCE* | Sigmoid outputs trained with binary cross-entropy: every class independently answers "is this expression showing". |

Every run keeps the weights of the epoch that did best on the held-out frames, so training too long doesn't hurt.

### Guided recording and intensity learning (Record / Train tab)
Blendshape trackers such as Project Babble are trained on *continuous* labels (how strong each shape is), not on "this is expression X". Pal Buddy Guy's original recordings only say "this whole file is expression X", which makes a model jump between 0 and 1 and lets it confuse the time of recording with the expression. Two additions close that gap:

* **Guided recording** (Record tab, recommended for every expression, ~36 s): a green bar and beeps (for use with the headset on; Windows) tell you how strong to make the expression: silence = neutral, higher pitch = stronger. The script ramps to full, holds, returns to neutral, goes to half strength and back, three times. Every frame gets its intensity as a label (shifted by 0.3 s for reaction time, stored next to the recording as `.labels.json`). A guided frame at intensity *v* is trained as *v* × expression + (1 − *v*) × neutral. Because each guided recording contains neutral stretches from the same session, the model can't tell expressions apart by when they were recorded.
* **Mixup** (Train tab, on by default): half of the expression samples are blended with a random neutral frame, `x = λ·expr + (1 − λ)·neutral`, targets blended the same way. This teaches in-between intensities even from plain recordings.

Guided and plain recordings can be mixed in a class. The validation metrics use the soft targets, so the comparison's *Int. error* shows how well each model follows intensity.

### Smoothing (Live tab)
*Smoothing* is a 1-Euro filter, the filter Project Babble and VRCFaceTracking use: small jitter is smoothed strongly, fast real movements pass with little lag. Around the middle of the slider it matches Babble's default (min cutoff ≈ 3 Hz, β 0.9).

### Compare models & pick (Train tab)
**Compare models & pick…** trains the selected combinations (by default standard/lite/compact with MSE and BCE) on the same recordings with the same sampling, scores each on the held-out frames and times it on this PC with your inference settings. The table shows:

| Column | Meaning |
|---|---|
| Accuracy | the shown expression has the highest output |
| Recognised | the shown expression's output is above 0.5 |
| False act. | another expression's output is above 0.3, i.e. a wrong shape would move (lower is better) |
| Int. error | mean distance between the output and the target intensity of the shown expression (guided recordings make this meaningful) |
| Score | (accuracy + recognised + 2 × (100 % − false act.) + (1 − int. error)) / 5 |
| ms/frame, MB | tracking cost on this PC |

The recommended row is marked ★: the highest score, and among models within 0.5 points of it the fastest. Select any row and press **Apply selected model** (or double-click it): it becomes the current model at once, is saved to the model file (the previous file is kept as `<name>.prev.pt`), the Train tab's choices are updated, and tracking restarts with it if it was running. CLI: `compare`, `arch`, `loss`.

**Softness** (Live tab, BCE models only): a BCE model's sigmoid outputs tend to snap between 0 and 1. Raising *Softness* (1 = off, up to 4) spreads the output out so it follows the expression gradually. It only changes how outputs are mapped, so there's no retraining; 1 % still maps to 0 and 99 % to 1. For shaking values use *Smoothing* instead.

After switching model or method, run FastCal once: the output scale of a new model can differ a little, and max_power is per model.

### Validation accuracy
Training holds out the last 10 % of every recording (`validation_split` in `config.json`) and reports accuracy, recognised, false activations and the score after each epoch, plus the worst classes at the end. The hold-out is taken from the end of each recording rather than as random frames, because neighbouring frames are nearly identical and a random split would overstate accuracy. It is still the same session as the training frames, so the numbers are optimistic; they are meant for comparing models, not as absolute accuracy.

**Validate on another session** (Train tab, off by default, `validation_sessions`): for every class with two or more recordings, the class's *last* recording is held out whole instead, and its other recordings are trained on in full. Make that recording on another day or after taking the headset off and on, and the numbers show how the model copes with a new headset fit - which is where an overfitted model falls apart, and what the same-session numbers can't show. The best-epoch choice and *Compare models* then also pick by this. Classes with a single recording still use their recording's end, and training lists them. It costs training frames (one recording per such class), so turn it off again for the final training once you have picked a model and settings.

**Against overfitting** (Train tab, both off by default; try them with *Validate on another session* on):
* **Augment (vary position / lighting)** (`augment`): every training batch is shifted by up to one feature cell (about 5 camera pixels, like the headset sitting slightly differently), scaled by ±10 % (lighting) and given a little noise. The eye and face features get their own shift and scale, because the two cameras move independently. Validation frames are never augmented. It runs on the training device: nearly free on an NVIDIA GPU, ~15-40 % slower training on the CPU.
* **Weight decay** (`weight_decay`, e.g. 0.05): AdamW instead of Adam (0 = the original Adam). The decay is multiplied by the learning rate, so it has a noticeable effect mainly with the compact model, whose step is 10x larger.
* In *Compare models*, tick **Also train each picked model with augmentation** to get every model with and without it side by side; applying a row also sets the Augment option it was trained with.

### Measurements
One 2.8 GHz Xeon core, frames arriving at 60 Hz, whole-process CPU as % of one core, memory measured while tracking:

| Model | Backend | ms/frame | CPU | RAM |
|---|---|---|---|---|
| standard | PyTorch fp32, all cores (old default) | 8.0 | 105 % | |
| standard | PyTorch int8, 1 thread | 7.6 | 50 % | 761 MB |
| standard | **ONNX int8, 1 thread** | 5.3 | 31 % | **102 MB** |
| lite | PyTorch fp32, 1 thread | 3.4 | 22 % | 576 MB |
| lite | **ONNX int8, 1 thread** | 1.4 | **11 %** | **72 MB** |

**Keeping it light**, all measured:
* While nothing receives the output (VRCFaceTracking not connected and no OSC outputs), tracking drops to **10 Hz**, and back to full rate within 0.5 s once something connects. The Live tab keeps updating.
* Camera frames are only copied while the preview is visible. The preview is scaled by Tk (1.6 ms per frame instead of 9 ms). While minimized, the window only refreshes once a second.
* The per-frame Python work avoids NumPy calls on single numbers and re-reads the output settings only every 0.5 s. VRCFT and OSC encodings are precompiled.
* Startup loads neither PyTorch nor ONNX Runtime until they're needed (import 0.1 s, 34 MB).
* Static int8 quantization of the conv layers was tried: 40 % faster, but a ~15 % output error, so it's not used.

Idle worker threads are also told not to busy-wait (`OMP_WAIT_POLICY=PASSIVE`, and ONNX Runtime spinning is off), so they don't burn CPU between frames.

### Tracking on a PC without PyTorch
Copy `config.json`, `buddyguy.onnx`, `buddyguy.int8.onnx` and `buddyguy.json` next to the app, then `pip install numpy onnxruntime`. Tracking works without PyTorch; only training needs it.

## 6. VRCFaceTracking (Settings tab → VRCFaceTracking module)
For **VRCFaceTracking v6**, install the module from the app with **Install / update**, then restart VRCFaceTracking. The status bar shows *VRCFaceTracking: connected (v6 module)*.

* VRCFaceTracking v6 lets only one module provide eye tracking and one provide expressions, so the Pal Buddy Guy module **runs the SRanipal module inside itself**. The installer disables the standalone SRanipal module by renaming its `module.json`, and **Remove** restores it. Install the SRanipal module in VRCFaceTracking first.
* A class's *Drives shape* can be an SRanipal lip shape (as before) or any **Unified Expression**, including eye-area shapes like `BrowLowererLeft` or `EyeSquintRight` that SRanipal can't express. For example, you can train a frown seen by the Pro Eye cameras.
* **v1 (legacy SRanipal) avatar parameters work.** VRCFaceTracking v6 computes the v1 parameters (`JawOpen`, `MouthSmileLeft`, `SmileSadLeft`, `JawX`, …) from Unified Expressions. SRanipal targets are translated so that this round trip gives the v1 parameter **exactly** the trained value, which is verified against VRCFaceTracking's own parameter functions for every shape. The v1 eye/brow parameters can also be chosen as targets directly: `LeftEyeWiden`, `LeftEyeSqueeze`, `BrowDownLeft`, `BrowsInnerUp`, and so on. Limitations that come from VRCFaceTracking itself:
  * The v1 `Tongue*Morph` parameters are always 0 in v6. Pal Buddy Guy drives the two tongue directions instead.
  * `MouthUpperOverturn` / `MouthLowerOverturn` also raise v1 `MouthUpperUp*` / `MouthLowerDown*`. This follows VRCFaceTracking's conversion formula, and every Unified-based module behaves the same way.
  * `MouthSadLeft/Right` are sent as `MouthStretch*`, because that reproduces v1 exactly without leaking into the other side. For a Unified (v2) avatar, pick `MouthFrownLeft/Right` instead.
* By default Pal Buddy Guy's value replaces SRanipal's for the shapes it drives. *Keep SRanipal's value when it is larger* switches to the maximum of the two.
* When tracking stops, the module falls back to plain SRanipal within 0.5 s.
* The original VRCFaceTracking module (≤ v4) still works; it only understands SRanipal lip shapes.

Details, the protocol and how to build or test the module: [`vrcft-module/README.md`](vrcft-module/README.md).

## Direct OSC outputs: merged parameters and class parameters
Besides driving VRCFaceTracking shapes, trained classes can go **straight to VRChat over OSC** (127.0.0.1:9000 by default) under custom avatar parameter names:

* **Class parameter** (Train tab → class → *OSC parameter*): the class weight, 0..1.
* **Merged parameter** (Merged tab): **any number of classes with weights**, `combined = Σ class × weight`, limited to -1..1 (0 when all classes are 0). For example `smile × 1 + sad × -1`, or in the spirit of VRCFT's EyeLidExpandedSqueeze, `wide × 0.2 + open × 0.8 + squeeze × -1`. The combined value is mapped through three points: **-1 → min, 0 → neutral, +1 → max**. Presets are -1..1, 0..1 and 0..2, and min, max and the **neutral value** can all be set by hand. With 0..2 and neutral 1, for example, sad gives 0, neutral gives 1 and smile gives 2.

**Format**: *float*, *binary* or *both*, with a selectable resolution of 1–8 bits. Binary uses the same scheme as VRCFaceTracking's binary parameters: bools `<name>1`, `<name>2`, `<name>4`, … hold bit *i* of `int(value × 2^bits)`, with all bits set near 1, so existing VRCFT binary animator setups decode them. Ranges crossing 0 (like -1..1) add `<name>Negative` and encode the magnitude. Other ranges encode the position within the range (min end 0, max end 1). Bools are only sent when they change, plus a full refresh every second.

**Names that could disturb face tracking are refused.** VRCFaceTracking drives any avatar parameter whose name equals one of its parameters or ends with `/<name>` (so `PBG/SmileSad` would be driven by VRCFT's `SmileSad`), plus binary bits `<name><number>` and `<name>Negative`. The app checks every name an output would write against all 744 v1/v2/eye/head parameters and 367 binary parameters of VRCFaceTracking v6 (generated from its source into `palbuddy/vrcft_names.py`) and refuses clashes. So if you don't use these outputs, or use them with names like `PBG_SmileSad`, normal face tracking is untouched. Outputs that write the same parameter twice are refused as well. Invalid outputs never block training; they just aren't sent.

When tracking stops, every output is set back to its neutral value. VRChat syncs float parameters in -1..1, so values outside that range (0..2) only arrive as-is for local, unsynced parameters. Binary parameters sync fine.

CLI: `merged add PBG_SmileSad smile,-sad -1 0 1 binary 4` (terms, min, neutral, max, format, bits), `merged add PBG_Lid 0.2*wide,0.8*open,-squeeze 0 0.5 1`, `oscout sad PBG_Sad both 3`, `merged`, `merged remove NAME`. Older configs with positive/negative classes load as two terms.

## Sensitivity (Live tab)
If a class only moves between, say, 0.2 and 0.8, stretch that part back to 0..1. Click the class in the output list, then drag **Low end** / **High end**; orange marks show them on the bar. **Auto (5 s)** measures it for you: make a neutral face, then the full expression, and the 5th–95th percentile of what it saw becomes the range. The top bar shows the value before the adjustment and the bottom bar what is sent. The adjustment applies to VRCFaceTracking shapes and OSC outputs alike. CLI: `sens smile 0.2 0.8`.

### Stability and talking
When emotion classes flicker while you talk, two things help:
* **A talking recording.** Record 30 s of reading text aloud with a neutral face (again in another session) and add it either to `neutral` or, better, as its own class `talk` with no target. The network then learns that mouth movement from speech is not an emotion. This is the real fix.
* **Stability** (Live tab, per class, below the sensitivity sliders): ignores changes shorter than the shown time (up to 400 ms; a syllable lasts ~100–250 ms), using a running median and a gentle low-pass. Set it on emotion classes and leave it off for mouth shapes that must follow speech. A held expression still comes through, just a fraction of a second later.

## 7. Process priority (Settings tab → Inference performance)
Tracking needs very little CPU, so by default the process runs at **below normal** priority and yields to VR and the game when the CPU is busy. On hybrid Intel CPUs (12th gen and later, e.g. the i7-12700K), **Efficiency cores only** keeps it off the P-cores entirely. On Windows 11, **Efficiency mode** (EcoQoS) lets the scheduler run it on slow, low-power cores. *low* priority is available too, but tracking may stutter when the CPU is fully loaded.

# What changed compared to the original scripts
**GUI**
* Tkinter GUI (no extra dependencies) with live connection status, camera preview, output meters, recording with progress, a class/dataset editor, training with a live loss chart and ETA, FastCal, and settings. Available in English and Korean.

**Performance**
* SRanipal is received in-process instead of through the proxy, which removes one TCP hop and one process.
* Frames are handed over through a latest-sample slot with a condition variable. The old code used a list queue with `pop(0)` and polled in 1 ms sleep loops.
* Inference runs once per *new* frame. The old loop recomputed the same frame and then slept 10 ms even when a new frame was waiting. It uses a pinned input buffer and `inference_mode`.
* VRCFT updates go out as one packet per frame. The old code made 1 + 3·N tiny `send` calls with Nagle enabled.
* Training: batches are built with one fancy-indexed read per recording and prepared by background threads while the GPU trains. Mixed precision (AMP) and cudnn autotuning are available. Recording writes straight to `.mmap`, so the `pkl → mmap` conversion step is gone.
* Fixed the `batch_size` global hack: inference used to set it to 1 and swap the dropout functions out.

**Connectivity**
* Disconnects are detected. Before, `recv()` returned `b""` forever after a peer left, and the thread spun at 100 % CPU without ever reconnecting.
* A restarted SRanipal or VRCFT replaces its stale connection automatically. There's no need to restart the script.
* The VRCFT connection uses TCP_NODELAY and keepalive. A send timeout keeps a frozen VRCFT from stalling tracking.
* All ports listen on `127.0.0.1` by default instead of on every network interface. This is configurable.
* Ports are configurable, and there's a clear error message when a port is already in use.
* The proxy client (legacy mode) reconnects with backoff.
* The status bar shows each stream as connected, no data (stalled) or disconnected.

The network architecture, the `buddyguy.pt` checkpoint format, the recording format and the VRCFT wire protocol are unchanged, so existing models, recordings and the VRCFT module keep working.

# Legacy proxy mode
If you want the receiver in its own process, run `python tvm_proxy.py` and set the input source to *proxy* in Settings (or `"source": "proxy"` in `config.json`).

# Tests
`python -m unittest discover tests` runs offline tests. They use fake SRanipal and VRCFT peers over loopback, and a tiny record → train → infer run on CPU.

---

# 한국어 빠른 시작
1. `tvm runtime` 폴더의 DLL 두 개를 `C:\Program Files\VIVE\SRanipal`에 덮어씁니다 (원본은 백업해 두세요).
2. **`PalBuddyGuy.bat`을 더블클릭합니다.** 처음 실행할 때는 Python(없으면 winget으로 설치할지 묻습니다), PyTorch(NVIDIA GPU가 있으면 CUDA 버전)와 나머지 패키지를 `.venv` 폴더에 자동으로 설치하고, 바탕화면 바로가기를 만들지 묻습니다. 다운로드가 약 2~3GB라 몇 분 걸립니다. 그다음부터는 더블클릭하면 콘솔 창 없이 바로 GUI가 뜹니다. 다시 설치하려면 `.venv` 폴더를 지우고 다시 실행하세요.
3. **SRanipalRuntime보다 먼저** 실행해야 합니다. 이제 `tvm_proxy.py`는 따로 실행할 필요가 없습니다.
4. **실시간** 탭: *입력 방식*을 고릅니다. **눈+입 / 입만(페이셜 트래커) / 눈만(Pro Eye)** 중 하나입니다. 트래커가 하나만 연결되어 있으면 미리보기를 보고 *입만* 또는 *눈만*을 누르면 됩니다. 두 트래커를 모두 쓸 때는 위에 눈, 아래에 얼굴이 보이는지 확인하고, 반대로 보이면 *눈/얼굴 뒤바꾸기*를 누르세요. 입력 방식을 바꾸면 다시 학습해야 합니다. 눈+입으로 녹화한 파일은 입만/눈만 학습에도 그대로 쓸 수 있습니다.
5. **녹화** 탭: 이름을 입력하고 녹화합니다 (약 30초, 파일 하나에 약 200MB). `neutral`(무표정) 녹화는 반드시 있어야 합니다.
6. **학습** 탭: 클래스별로 녹화 파일과 대상 파라미터를 지정합니다 (첫 번째 클래스는 neutral). *학습 시작*을 누르고, 손실이 0.001 아래로 내려가면 *모델 저장*을 누릅니다.
7. **실시간** 탭: *트래킹 시작*을 누른 뒤 *빠른 보정(FastCal)*으로 아바타를 따라 하며 보정합니다.

8. **설정 탭 → 추론 성능**: 기본값(자동: ONNX Runtime, CPU int8)으로도 가볍게 돌아갑니다. *이 PC에서 비교 측정*을 누르면 CPU/GPU/ONNX 중 어느 쪽이 이 PC에 맞는지 바로 확인할 수 있습니다.
   **학습 탭 → 모델 비교 후 선택…**: 표준/경량/컴팩트 모델 × MSE/BCE 학습 방식을 같은 녹화로 각각 학습해서 **정확도·인식률·오작동률·점수**와 이 PC에서의 **속도**를 표로 보여 줍니다. ★ 표시가 추천 모델(최고 점수, 0.5점 이내면 더 빠른 쪽)입니다. 원하는 줄을 고르고 *선택한 모델 적용*을 누르면 바로 그 모델로 트래킹합니다 (자동 저장, 이전 모델은 `.prev.pt`로 보관). 모델을 바꾼 뒤에는 FastCal을 한 번 해 주세요. BCE 모델 값이 0과 1 사이를 확 오가면 실시간 탭의 **부드러움**을 올리세요 (재학습 불필요). 값이 떨리면 **스무딩**을 올리면 됩니다. *컴팩트*는 표준보다 가중치가 약 30배 적어 과적합이 덜하고, *BCE*는 표정마다 독립적으로 학습합니다.
9. **설정 탭 → VRCFaceTracking 모듈 → 설치**: VRCFaceTracking v6용 모듈을 설치합니다. 먼저 VRCFaceTracking에서 SRanipal 모듈을 설치해 두세요. 설치하면 SRanipal 모듈은 Pal Buddy Guy 모듈 안에서 실행됩니다. v6는 눈과 표정을 각각 모듈 하나만 담당할 수 있기 때문입니다. 설치 후 VRCFaceTracking을 다시 시작하세요. *제거*를 누르면 원래대로 돌아갑니다. 대상 파라미터로 `BrowLowererLeft` 같은 **눈썹/눈 주변 표정**도 고를 수 있습니다.
10. **프로세스 우선순위**: 기본값은 "보통 이하"라서 VR/게임에 CPU를 양보합니다. 12세대 이후 Intel CPU에서는 *E코어만 사용*, Windows 11에서는 *효율 모드*도 쓸 수 있습니다.
11. **OSC 직접 출력**: 병합 파라미터 탭에서 **여러 클래스를 가중치로** 합치거나(예: 웃음 × 1 + 슬픔 × -1, 넓힘 × 0.2 + 뜸 × 0.8 + 찡그림 × -1; 범위 -1~1 / 0~1 / 0~2 / 직접 입력, **중간값 직접 설정**), 학습 탭의 클래스 설정에서 *OSC 파라미터* 이름을 지정해 클래스 값(0~1)을 VRChat에 직접 보낼 수 있습니다. 형식은 **float / 바이너리 / 둘 다** 중에서 고르고, 바이너리는 **1~8비트**로 VRCFT 바이너리와 같은 방식입니다. **VRCFT가 쓰는 이름(v1/v2, 바이너리, `/이름` 경로 포함)은 사용할 수 없게 막아서** 기존 페이셜에 영향이 가지 않습니다. `PBG_SmileSad`처럼 고유한 이름을 쓰세요.
12. **민감도 (실시간 탭)**: 출력 목록에서 클래스를 클릭하고 하한/상한 슬라이더를 조정하면, 예를 들어 0.2~0.8로만 움직이는 값을 0~1로 늘립니다. *자동 (5초)*을 누르고 무표정 → 최대 표정을 지으면 범위를 자동으로 잡아 줍니다.
13. **말할 때 감정이 출렁이면**: 무표정으로 글을 소리 내어 읽는 녹화(30초, 다른 날 한 번 더)를 만들어 **대상 없는 `talk` 클래스**로 추가하고 다시 학습하세요. 근본 해결책입니다. 추가로 실시간 탭에서 감정 클래스를 클릭하고 **안정화** 슬라이더를 올리면 짧은 출렁임(최대 400ms 미만)을 무시합니다. 입 모양 클래스에는 끄세요.
14. **가이드 녹화 (녹화 탭, 표정마다 권장)**: 초록 막대와 비프음(소리 없음 = 무표정, 음이 높을수록 강하게)을 따라 약 36초 동안 표정을 지으면, 프레임마다 **강도 정답**이 저장됩니다. Project Babble처럼 표정의 **세기**를 학습해서 0↔1로 튀지 않고, 녹화마다 같은 착용 상태의 무표정이 들어가 착용 차이에도 강해집니다. 학습 탭의 **중간 강도 학습 (mixup)**은 기본으로 켜져 있고, 일반 녹화로도 중간 강도를 배우게 합니다. 스무딩은 Babble·VRCFT와 같은 **1-Euro 필터**입니다.

언어는 설정 탭에서 바꿀 수 있습니다 (auto / en / ko).
