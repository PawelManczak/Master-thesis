"""
Shared configuration constants for all dataset processing scripts.

Sampling frequencies are hardware-specific:
  - Empatica E4: EDA=4Hz, BVP=64Hz, TEMP=4Hz, ACC=32Hz
  - CASE: ECG/physio=1000Hz, Annotations=20Hz

Window size:
  - 30s = minimum for stable time-domain HRV metrics (Task Force ESC, 1996).
    K-emoCon annotations are every 5s, so one 30s window = 6 annotations.

IBI validity:
  - 300–2000 ms = 30–200 BPM physiological range filter.
"""

# =============================================================================
# WINDOWING
# =============================================================================
WINDOW_SEC = 10          # seconds per window (HRV minimum, Task Force ESC 1996)
ANNOT_STEP_SEC = 5       # K-emoCon annotation interval (seconds)
ANNOTS_PER_WINDOW = WINDOW_SEC // ANNOT_STEP_SEC  # = 6

# =============================================================================
# EMPATICA E4 SAMPLING FREQUENCIES
# =============================================================================
FS_EDA = 4.0    # Hz
FS_BVP = 64.0   # Hz
FS_TEMP = 4.0   # Hz
FS_ACC = 32.0   # Hz
FS_HR = 1.0     # Hz  (derived from IBI interpolation)

# =============================================================================
# CASE DATASET SAMPLING FREQUENCIES
# =============================================================================
CASE_FS_PHYSIO = 1000.0  # Hz  (ECG, EDA, etc.)
CASE_FS_ANNOT = 20.0     # Hz  (joystick annotations)

# =============================================================================
# CEAP DATASET
# =============================================================================
CEAP_FS_EDA = 4.0    # Hz
CEAP_FS_BVP = 62.5   # Hz  (slightly different from E4 standard)
CEAP_FS_ACC = 32.26  # Hz
CEAP_WINDOW_SEC = 5.0  # CEAP uses 5-second annotation windows (not 30s)

# =============================================================================
# IBI / HRV VALIDITY RANGE
# =============================================================================
IBI_MIN_MS = 300   # ms = 200 BPM max
IBI_MAX_MS = 2000  # ms = 30 BPM min
