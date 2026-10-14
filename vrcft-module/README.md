# Pal Buddy Guy module for VRCFaceTracking v6

Most users don't need anything in this folder. Install the module from the app: **Settings → VRCFaceTracking module → Install**, or `vrcft install` in the CLI. The app copies `prebuilt/` into VRCFaceTracking's modules folder.

## Why it wraps the SRanipal module
VRCFaceTracking v6 runs every module in its own process. It gives **eye tracking to exactly one module and expressions to exactly one module**, whichever finishes initialising first, and the chosen module's values replace the global state. There is no ordering or merging. A separate Pal Buddy Guy module next to the SRanipal module would therefore race it, and at most one of them would reach VRChat.

So this module **runs the SRanipal module inside itself**:

1. The installer disables the standalone SRanipal module by renaming its `module.json` to `module.json.palbuddyguy-disabled`. VRCFaceTracking skips folders without a `module.json`, and nothing is deleted. The installer also writes `palbuddyguy.json`, containing the app's host and port and the path of the SRanipal DLL.
2. The module loads the SRanipal module into its own process with a **private copy of `VRCFaceTracking.Core`**. SRanipal therefore writes into a private `UnifiedTracking.Data`. Modules typically write their values and then sleep inside `Update()`, so if both wrote to the shared object, VRCFaceTracking would read SRanipal's value for most of each cycle and the overrides would only flash briefly.
3. After each SRanipal update, the eye and head data and the shapes are written to the shared data once. Shapes that Pal Buddy Guy drives get its value, either replacing SRanipal's or taking the larger of the two (the *max* option).

Because the module owns both slots, it can also drive **eye-area shapes**: brows, squint and widen. VRCFaceTracking only takes those from the eye-tracking module.

If no SRanipal module is found, the module still provides expressions, using only Pal Buddy Guy's shapes. This is useful for a Vive Pro Eye without a facial tracker next to another eye-tracking module.

When the app stops tracking, or disappears, the module falls back to plain SRanipal: immediately on a *clear* message, otherwise after 0.5 s without updates.

## Protocol (TCP, the app listens on port 26421)
```
module -> app : "PBG2" version:u8                       hello
app -> module : 0x05 mode:u8 count:u8 {len:u8 name}*     target table (Unified Expression names; mode 0 replace, 1 max)
app -> module : 0x03 count:u8 {slot:u8 weight:u16 BE}*   weights 0..65535 = 0..1; count 0 = stop overriding
```
The original module (VRCFaceTracking ≤ v4) is still supported. The app detects which one connected from what it sends first.

## Building
Requires the .NET 10 SDK.
```
dotnet build -c Release -p:VRCFTSource=C:\src\VRCFaceTracking -p:DebugType=none -o out
```
Alternatively, `-p:VRCFTInstallDir=<folder with VRCFaceTracking.Core.dll>` builds against an installed VRCFaceTracking. Copy `out/PalBuddyGuy.VRCFT.dll` and `module.json` to `prebuilt/`.

## Integration test (no Windows or headset needed)
`tests/Harness` stands in for the VRCFaceTracking app. It starts the **real** `VRCFaceTracking.ModuleProcess` with this module, speaks the sandbox IPC protocol, and prints the resulting tracking state. `tests/FakeSRanipal` stands in for the SRanipal module and provides fixed eye and lip values.
```
V=/path/to/VRCFaceTracking   # source checkout
dotnet build $V/VRCFaceTracking.ModuleProcess -c Release -o e2e/mp
dotnet build tests/Harness      -c Release -p:VRCFTSource=$V -o e2e/harness
dotnet build tests/FakeSRanipal -c Release -p:VRCFTSource=$V -o e2e/fake
dotnet build tests/V1Check      -c Release -p:VRCFTSource=$V -o e2e/v1check
PALBUDDY_VRCFT_TEST_DIR=$PWD/e2e python -m unittest tests.test_vrcft_module   # from the repo root
```
The test installs `prebuilt/` into a temporary CustomLibs folder and drives shapes from the Python app. It checks that the overrides arrive with no flicker back to SRanipal's value, that eye data and untouched shapes pass through, and that the module falls back to SRanipal when the app stops sending.

`tests/V1Check` evaluates VRCFaceTracking's own v1 (legacy SRanipal) parameter functions. The test uses it to check that every SRanipal and v1 target produces the trained value as a v1 avatar parameter, including combined ones like `JawX` and `SmileSadLeft`.

Not covered: the real SRanipal module and hardware. The wrapping relies on the SRanipal module behaving like a normal VRCFaceTracking module (only `ExtTrackingModule` and `UnifiedTracking.Data`). Its camera images aren't forwarded to VRCFaceTracking's UI.
