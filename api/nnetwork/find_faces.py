import os

import numpy as np

from settings import *


def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def get_coords(faces_roi):
    coords = np.argwhere(faces_roi == 1)
    # print(coords)
    if coords.shape[0] == 0:
        return None
    y1, x1 = coords[0]
    y2, x2 = coords[-1]
    return x1, y1, x2, y2


def interpolate_center(c1, c2, length):
    x1, y1 = c1
    x2, y2 = c2
    xi, yi = np.linspace(x1, x2, length), np.linspace(y1, y2, length)
    return np.vstack([xi, yi]).transpose(1, 0)


def get_faces(faces_roi, upsample):
    all_faces = []
    rows = faces_roi[0].shape[1]
    cols = faces_roi[0].shape[2]
    for i in range(len(faces_roi)):
        faces = np.asarray([get_coords(faces_roi[i][j]) for j in range(len(faces_roi[i]))])
        if faces[0] is None:  faces[0] = faces[1]
        if faces[-1] is None: faces[-1] = faces[-2]
        if None in faces:
            # print(faces)
            raise Exception('This should not have happened ...')
        all_faces.append(faces)

    extracted_faces = []
    for face in all_faces:
        # Get max dim size
        max_dim = np.concatenate([face[:, 2] - face[:, 0], face[:, 3] - face[:, 1]])
        max_dim = np.percentile(max_dim, 90)
        # Enlarge by 1.2
        max_dim = int(max_dim * 1.2)
        # Get center coords
        centers = np.asarray([get_center(_) for _ in face])
        # Interpolate
        centers = np.vstack(
            [interpolate_center(centers[i], centers[i + 1], length=10) for i in range(len(centers) - 1)]).astype('int')
        x1y1 = centers - max_dim // 2
        x2y2 = centers + max_dim // 2
        x1, y1 = x1y1[:, 0], x1y1[:, 1]
        x2, y2 = x2y2[:, 0], x2y2[:, 1]
        # If x1 or y1 is negative, turn it to 0
        # Then add to x2 y2 or y2
        x2[x1 < 0] -= x1[x1 < 0]
        y2[y1 < 0] -= y1[y1 < 0]
        x1[x1 < 0] = 0
        y1[y1 < 0] = 0
        # If x2 or y2 is too big, turn it to max image shape
        # Then subtract from y1
        y1[y2 > rows] += rows - y2[y2 > rows]
        x1[x2 > cols] += cols - x2[x2 > cols]
        y2[y2 > rows] = rows
        x2[x2 > cols] = cols
        vidface = np.asarray([[x1[_], y1[_], x2[_], y2[_]] for _, c in enumerate(centers)])
        vidface = (vidface * upsample).astype('int')
        extracted_faces.append(vidface)

    return extracted_faces


def detect_face_with_mtcnn(mtcnn_model, pil_frames, facedetection_upsample, video_shape, face_frames):
    boxes, _probs = mtcnn_model.detect(pil_frames, landmarks=False)
    faces, faces_roi = get_roi_for_each_face(faces_by_frame=boxes, probs=_probs, video_shape=video_shape,
                                             temporal_upsample=face_frames, upsample=facedetection_upsample)
    coords = [] if len(faces_roi) == 0 else get_faces(faces_roi, upsample=facedetection_upsample)
    return faces, coords


def face_detection_wrapper(mtcnn_model, videopath, every_n_frames, facedetection_downsample, max_frames_to_load):
    video, pil_frames, rescale = load_video(videopath, every_n_frames=every_n_frames, to_rgb=True,
                                            rescale=facedetection_downsample, inc_pil=True,
                                            max_frames=max_frames_to_load)
    if len(pil_frames):
        try:
            faces, coords = detect_face_with_mtcnn(mtcnn_model=mtcnn,
                                                   pil_frames=pil_frames,
                                                   facedetection_upsample=1 / rescale,
                                                   video_shape=video.shape,
                                                   face_frames=every_n_frames)
        except RuntimeError:  # Out of CUDA RAM
            print(f"Failed to process {videopath} ! Downsampling x2 ...")
            video, pil_frames, rescale = load_video(videopath, every_n_frames=every_n_frames, to_rgb=True,
                                                    rescale=facedetection_downsample / 2, inc_pil=True,
                                                    max_frames=max_frames_to_load)

            try:
                faces, coords = detect_face_with_mtcnn(mtcnn_model=mtcnn,
                                                       pil_frames=pil_frames,
                                                       facedetection_upsample=1 / rescale,
                                                       video_shape=video.shape,
                                                       face_frames=every_n_frames)
            except RuntimeError:
                print(f"Failed on downsample ! Skipping...")
                return [], []

    else:
        print('Failed to fetch frames ! Skipping ...')
        return [], []

    if len(faces) == 0:
        print('Failed to find faces ! Upsampling x2 ...')
        try:
            video, pil_frames, rescale = load_video(videopath, every_n_frames=every_n_frames, to_rgb=True,
                                                    rescale=facedetection_downsample * 2, inc_pil=True,
                                                    max_frames=max_frames_to_load)
            faces, coords = detect_face_with_mtcnn(mtcnn_model=mtcnn,
                                                   pil_frames=pil_frames,
                                                   facedetection_upsample=1 / rescale,
                                                   video_shape=video.shape,
                                                   face_frames=every_n_frames)
        except Exception as e:
            print(e)
            return [], []

    return faces, coords


videopaths = sorted(glob(os.path.join(INPUT_DIR, "*.mp4")))
print(f'Found {len(videopaths)} videos !')

mtcnn = MTCNN(margin=0, keep_all=True, post_process=False, select_largest=False, device='cuda:0',
              thresholds=MTCNN_THRESHOLDS, factor=MMTNN_FACTOR)

faces_by_videopath = {}
coords_by_videopath = {}

for i_video, videopath in enumerate(tqdm(videopaths)):
    faces, coords = face_detection_wrapper(mtcnn, videopath, every_n_frames=FACE_FRAMES,
                                           facedetection_downsample=FACEDETECTION_DOWNSAMPLE,
                                           max_frames_to_load=MAX_FRAMES_TO_LOAD)

    if len(faces):
        faces_by_videopath[videopath] = faces[:MAX_FACES_HIGHTHRESH]
        coords_by_videopath[videopath] = coords[:MAX_FACES_HIGHTHRESH]
    else:
        print(f"Found no faces for {videopath} !")

del mtcnn
import gc

gc.collect()

videopaths_missing_faces = {p for p in videopaths if p not in faces_by_videopath}
print(f"Found faces for {len(faces_by_videopath)} videos; {len(videopaths_missing_faces)} missing")