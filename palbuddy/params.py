"""Output targets.

Three naming schemes are accepted as a class's target:
  * SRanipal lip shapes (LIP_SHAPES) - what the original script and the old
    VRCFaceTracking module used; also VRCFaceTracking's "v1" avatar parameters,
  * v1-only eye/brow parameters (V1_ONLY), e.g. BrowDownLeft, LeftEyeWiden,
  * Unified Expressions (UNIFIED_EXPRESSIONS) - VRCFaceTracking v5/v6's shapes.

The v6 module (protocol 2) receives Unified Expression weights. VRCFaceTracking v6
computes v1 avatar parameters back from those (UnifiedSRanMapper / EyeTrackingParams),
so SRanipal and v1 names are translated such that the round trip gives the v1
parameter exactly the trained value. This is checked against VRCFaceTracking's own
parameter functions in tests/test_vrcft_module.py (V1RoundTripTests).
The old module (protocol 1) only understands SRanipal names.
"""

LIP_SHAPES = {
    "JawRight": 0,  # +JawX
    "JawLeft": 1,  # -JawX
    "JawForward": 2,
    "JawOpen": 3,
    "MouthApeShape": 4,
    "MouthUpperRight": 5,  # +MouthUpper
    "MouthUpperLeft": 6,  # -MouthUpper
    "MouthLowerRight": 7,  # +MouthLower
    "MouthLowerLeft": 8,  # -MouthLower
    "MouthUpperOverturn": 9,
    "MouthLowerOverturn": 10,
    "MouthPout": 11,
    "MouthSmileRight": 12,  # +SmileSadRight
    "MouthSmileLeft": 13,  # +SmileSadLeft
    "MouthSadRight": 14,  # -SmileSadRight
    "MouthSadLeft": 15,  # -SmileSadLeft
    "CheekPuffRight": 16,
    "CheekPuffLeft": 17,
    "CheekSuck": 18,
    "MouthUpperUpRight": 19,
    "MouthUpperUpLeft": 20,
    "MouthLowerDownRight": 21,
    "MouthLowerDownLeft": 22,
    "MouthUpperInside": 23,
    "MouthLowerInside": 24,
    "MouthLowerOverlay": 25,
    "TongueLongStep1": 26,
    "TongueLeft": 27,  # -TongueX
    "TongueRight": 28,  # +TongueX
    "TongueUp": 29,  # +TongueY
    "TongueDown": 30,  # -TongueY
    "TongueRoll": 31,
    "TongueLongStep2": 32,
    "TongueUpRightMorph": 33,
    "TongueUpLeftMorph": 34,
    "TongueDownRightMorph": 35,
    "TongueDownLeftMorph": 36,
}

SHAPE_NAMES = {v: k for k, v in LIP_SHAPES.items()}

# VRCFaceTracking.Core.Params.Expressions.UnifiedExpressions, in enum order (VRCFT v6)
UNIFIED_EXPRESSIONS = (
    "EyeSquintRight", "EyeSquintLeft", "EyeWideRight", "EyeWideLeft", "BrowPinchRight", "BrowPinchLeft",
    "BrowLowererRight", "BrowLowererLeft", "BrowInnerUpRight", "BrowInnerUpLeft", "BrowOuterUpRight",
    "BrowOuterUpLeft", "NasalDilationRight", "NasalDilationLeft", "NasalConstrictRight", "NasalConstrictLeft",
    "CheekSquintRight", "CheekSquintLeft", "CheekPuffRight", "CheekPuffLeft", "CheekSuckRight",
    "CheekSuckLeft", "JawOpen", "JawRight", "JawLeft", "JawForward", "JawBackward", "JawClench",
    "JawMandibleRaise", "MouthClosed", "LipSuckUpperRight", "LipSuckUpperLeft", "LipSuckLowerRight",
    "LipSuckLowerLeft", "LipSuckCornerRight", "LipSuckCornerLeft", "LipFunnelUpperRight",
    "LipFunnelUpperLeft", "LipFunnelLowerRight", "LipFunnelLowerLeft", "LipPuckerUpperRight",
    "LipPuckerUpperLeft", "LipPuckerLowerRight", "LipPuckerLowerLeft", "MouthUpperUpRight",
    "MouthUpperUpLeft", "MouthUpperDeepenRight", "MouthUpperDeepenLeft", "NoseSneerRight", "NoseSneerLeft",
    "MouthLowerDownRight", "MouthLowerDownLeft", "MouthUpperRight", "MouthUpperLeft", "MouthLowerRight",
    "MouthLowerLeft", "MouthCornerPullRight", "MouthCornerPullLeft", "MouthCornerSlantRight",
    "MouthCornerSlantLeft", "MouthFrownRight", "MouthFrownLeft", "MouthStretchRight", "MouthStretchLeft",
    "MouthDimpleRight", "MouthDimpleLeft", "MouthRaiserUpper", "MouthRaiserLower", "MouthPressRight",
    "MouthPressLeft", "MouthTightenerRight", "MouthTightenerLeft", "TongueOut", "TongueUp", "TongueDown",
    "TongueRight", "TongueLeft", "TongueRoll", "TongueBendDown", "TongueCurlUp", "TongueSquish", "TongueFlat",
    "TongueTwistRight", "TongueTwistLeft", "SoftPalateClose", "ThroatSwallow", "NeckFlexRight",
    "NeckFlexLeft",
)

# SRanipal lip shape -> Unified Expression shape(s) it corresponds to
LEGACY_TO_UNIFIED = {
    "JawRight": ("JawRight",), "JawLeft": ("JawLeft",), "JawForward": ("JawForward",), "JawOpen": ("JawOpen",),
    "MouthApeShape": ("MouthClosed",),
    "MouthUpperRight": ("MouthUpperRight",), "MouthUpperLeft": ("MouthUpperLeft",),
    "MouthLowerRight": ("MouthLowerRight",), "MouthLowerLeft": ("MouthLowerLeft",),
    "MouthUpperOverturn": ("LipFunnelUpperRight", "LipFunnelUpperLeft"),
    "MouthLowerOverturn": ("LipFunnelLowerRight", "LipFunnelLowerLeft"),
    "MouthPout": ("LipPuckerUpperRight", "LipPuckerUpperLeft", "LipPuckerLowerRight", "LipPuckerLowerLeft"),
    "MouthSmileRight": ("MouthCornerPullRight", "MouthCornerSlantRight"),
    "MouthSmileLeft": ("MouthCornerPullLeft", "MouthCornerSlantLeft"),
    # v1 MouthSad* = max((FrownL+FrownR)/2, Stretch*) - Smile*: Stretch reproduces it exactly
    # without leaking into the other side (a v2 avatar sees a stretch; pick MouthFrown* there)
    "MouthSadRight": ("MouthStretchRight",), "MouthSadLeft": ("MouthStretchLeft",),
    "CheekPuffRight": ("CheekPuffRight",), "CheekPuffLeft": ("CheekPuffLeft",),
    "CheekSuck": ("CheekSuckRight", "CheekSuckLeft"),
    "MouthUpperUpRight": ("MouthUpperUpRight",), "MouthUpperUpLeft": ("MouthUpperUpLeft",),
    "MouthLowerDownRight": ("MouthLowerDownRight",), "MouthLowerDownLeft": ("MouthLowerDownLeft",),
    "MouthUpperInside": ("LipSuckUpperRight", "LipSuckUpperLeft"),
    "MouthLowerInside": ("LipSuckLowerRight", "LipSuckLowerLeft"),
    "MouthLowerOverlay": ("MouthRaiserLower",),
    # v1 TongueLongStep1 = min(1, 2*TongueOut), Step2 = clamp(2*TongueOut - 1): see TRANSFORMS
    "TongueLongStep1": ("TongueOut",), "TongueLongStep2": ("TongueOut",),
    "TongueLeft": ("TongueLeft",), "TongueRight": ("TongueRight",),
    "TongueUp": ("TongueUp",), "TongueDown": ("TongueDown",), "TongueRoll": ("TongueRoll",),
    # VRCFaceTracking v6 always sends 0 for the v1 Tongue*Morph parameters; the closest we
    # can do is drive the two directions (visible through v1 TongueUp/TongueRight etc.)
    "TongueUpRightMorph": ("TongueUp", "TongueRight"), "TongueUpLeftMorph": ("TongueUp", "TongueLeft"),
    "TongueDownRightMorph": ("TongueDown", "TongueRight"), "TongueDownLeftMorph": ("TongueDown", "TongueLeft"),
}

# v1 avatar parameters that aren't SRanipal lip shapes (from VRCFT's EyeTrackingParams)
V1_ONLY = {
    "LeftEyeWiden": ("EyeWideLeft",), "RightEyeWiden": ("EyeWideRight",), "EyeWiden": ("EyeWideLeft", "EyeWideRight"),
    "LeftEyeSqueeze": ("EyeSquintLeft",), "RightEyeSqueeze": ("EyeSquintRight",),
    "EyesSqueeze": ("EyeSquintLeft", "EyeSquintRight"),
    # v1 BrowDown* = (BrowPinch + BrowLowerer) / 2
    "BrowDownLeft": ("BrowPinchLeft", "BrowLowererLeft"), "BrowDownRight": ("BrowPinchRight", "BrowLowererRight"),
    "BrowsDown": ("BrowPinchLeft", "BrowLowererLeft", "BrowPinchRight", "BrowLowererRight"),
    "BrowsInnerUp": ("BrowInnerUpLeft", "BrowInnerUpRight"), "BrowsOuterUp": ("BrowOuterUpLeft", "BrowOuterUpRight"),
}

# target -> (scale, offset): Unified weight = offset + scale * w for w > 0 (0 when w == 0)
TRANSFORMS = {
    "TongueLongStep1": (0.5, 0.0),   # TongueOut 0..0.5 -> v1 Step1 0..1, Step2 0
    "TongueLongStep2": (0.5, 0.5),   # TongueOut 0.5..1 -> v1 Step2 0..1 (Step1 full, as in SRanipal)
}

# SRanipal targets VRCFaceTracking v6 can't reproduce as v1 parameters
V1_UNSUPPORTED = frozenset(("TongueUpRightMorph", "TongueUpLeftMorph", "TongueDownRightMorph", "TongueDownLeftMorph"))

_UNIFIED_SET = frozenset(UNIFIED_EXPRESSIONS)


def is_valid_target(name):
    return name in LIP_SHAPES or name in _UNIFIED_SET or name in V1_ONLY


def unified_targets(name):
    """Unified Expression names driven by a target (SRanipal and v1 names are translated)."""
    if name in LEGACY_TO_UNIFIED:
        return LEGACY_TO_UNIFIED[name]
    if name in V1_ONLY:
        return V1_ONLY[name]
    if name in _UNIFIED_SET:
        return (name,)
    raise KeyError(name)


def unified_weights(name, w):
    """[(unified_name, weight)] for a target at weight w (0..1)."""
    w = min(1.0, max(0.0, float(w)))
    scale, offset = TRANSFORMS.get(name, (1.0, 0.0))
    uw = min(1.0, offset + scale * w) if w > 0 else 0.0
    return [(u, uw) for u in unified_targets(name)]


def legacy_index(name):
    """SRanipal index for the old VRCFT module, or None if the target has no SRanipal equivalent."""
    return LIP_SHAPES.get(name)


def all_target_names():
    """SRanipal names (in SRanipal order), v1 eye/brow names, then Unified-only names, for pickers."""
    legacy = sorted(LIP_SHAPES, key=LIP_SHAPES.get)
    return legacy + list(V1_ONLY) + [n for n in UNIFIED_EXPRESSIONS if n not in LIP_SHAPES and n not in V1_ONLY]


def shape_id(name_or_id):
    """SRanipal index of a lip shape name ("JawOpen") or a raw index (legacy helper)."""
    if isinstance(name_or_id, int):
        return name_or_id
    if isinstance(name_or_id, str) and name_or_id.isdigit():
        return int(name_or_id)
    return LIP_SHAPES[name_or_id]
