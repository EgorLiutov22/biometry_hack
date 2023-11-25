import numpy as np

PRETRAINED_MODELS_3D = [{'type':'i3d',
                         'path':"/kaggle/input/deepfake-detection-jph/j3d_e1_l0.1374.model"},
                        {'type':'res34',
                         'path':"/kaggle/input/deepfake-detection-jph/res34_1cy_minaug_nonorm_e4_l0.1794.model"},
                        {'type':'mc3_112',
                         'path':"/kaggle/input/deepfake-detection-jph/mc3_18_112_1cy_lilaug_nonorm_e9_l0.1905.model"},
                        {'type':'mc3_224',
                         'path':"/kaggle/input/deepfake-detection-jph/mc3_18_112t224_1cy_lilaug_nonorm_e7_l0.1901.model"},
                        {'type':'r2p1_112',
                         'path':"/kaggle/input/deepfake-detection-jph/r2p1_18_8_112tr_112te_e12_l0.1741.model"},
                        {'type':'i3d',
                         'path':"/kaggle/input/deepfake-detection-jph/i3dcutmix_e11_l0.1612.model"},
                        {'type':'r2p1_112',
                         'path':"/kaggle/input/deepfake-detection-jph/r2plus1dcutmix_112_e10_l0.1608.model"}]

# Face detection
MAX_FRAMES_TO_LOAD = 100
MIN_FRAMES_FOR_FACE = 30
MAX_FRAMES_FOR_FACE = 100
FACE_FRAMES = 10
MAX_FACES_HIGHTHRESH = 5
MAX_FACES_LOWTHRESH = 1
FACEDETECTION_DOWNSAMPLE = 0.25
MTCNN_THRESHOLDS = (0.8, 0.8, 0.9)  # Default [0.6, 0.7, 0.7]
MTCNN_THRESHOLDS_RETRY = (0.5, 0.5, 0.5)
MMTNN_FACTOR = 0.71  # Default 0.709 p
TWO_FRAME_OVERLAP = False

# Inference
PROB_MIN, PROB_MAX = 0.001, 0.999
REVERSE_PROBS = True
DEFAULT_MISSING_PRED = 0.5
USE_FACE_FUNCTION = np.mean

# 3D inference
RATIO_3D = 1
OUTPUT_FACE_SIZE = (256, 256)
PRE_INFERENCE_CROP = (224, 224)

# 2D
RATIO_2D = 1