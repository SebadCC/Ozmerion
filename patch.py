# patch.py
import cv2
import numpy as np
import mediapipe.python.solutions.face_mesh as mp_face_mesh

# ==============================================================================
# CONFIGURACIÓN Y PARÁMETROS DE CALIBRACIÓN CANÓNICOS
# ==============================================================================
TARGET_PATCH_SIZE = (64, 64)
YAW_RATIO_RANGE = (0.60, 1.75)

# Margen de holgura porcentual ampliado sobre la caja anatómica real
PADDING_PERIOCULAR = 0.35  # 35% de holgura para asegurar ceja y cuenca ocular
PADDING_BOCA = 0.40        # 40% de holgura para capturar comisuras y barbilla

# Configuración de CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# Confianza MediaPipe
DETECTION_CONFIDENCE = 0.4


class GeometricPatchExtractor:
    CANTO_INTERNO_IZQ = 133
    CANTO_INTERNO_DER = 362
    SELLION = 168

    PTS_PERIOCULAR_IZQ = [
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
    ]

    PTS_PERIOCULAR_DER = [
        336, 296, 334, 293, 300, 276, 283, 282, 295, 285,
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
    ]

    PTS_BOCA = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402,
        317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 0
    ]

    def __init__(
        self,
        target_size=TARGET_PATCH_SIZE,
        yaw_threshold_range=YAW_RATIO_RANGE,
        detection_confidence=DETECTION_CONFIDENCE
    ):
        self.target_size = target_size
        self.yaw_min, self.yaw_max = yaw_threshold_range
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence
        )
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_GRID_SIZE
        )

    def _normalize_patch(self, patch):
        if patch is None or patch.size == 0:
            return None
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch
        resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
        return self.clahe.apply(resized)

    def _get_square_box_from_points(self, points, padding_ratio, img_w, img_h):
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        box_w = x_max - x_min
        box_h = y_max - y_min

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        side = max(box_w, box_h) * (1.0 + padding_ratio)

        x1 = int(cx - side / 2.0)
        y1 = int(cy - side / 2.0)
        x2 = int(cx + side / 2.0)
        y2 = int(cy + side / 2.0)

        # Truncamiento en bordes para no desbordar las dimensiones del frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return [y1, y2, x1, x2]

    def extract_patches(self, frame_bgr):
        if frame_bgr is None or frame_bgr.size == 0:
            return None, "NO_FACE"

        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None, "NO_FACE"

        mesh = results.multi_face_landmarks[0].landmark
        coords = np.array([[lm.x * w, lm.y * h] for lm in mesh], dtype=np.float32)

        # ----------------------------------------------------------------------
        # Control de pose (Yaw)
        # ----------------------------------------------------------------------
        p_canto_izq = coords[self.CANTO_INTERNO_IZQ]
        p_canto_der = coords[self.CANTO_INTERNO_DER]
        p_sellion = coords[self.SELLION]

        d_izq_nasal = np.linalg.norm(p_canto_izq - p_sellion)
        d_der_nasal = np.linalg.norm(p_canto_der - p_sellion)

        if d_der_nasal < 1e-6:
            return None, "DISCARDED_YAW_PITCH"

        yaw_ratio = d_izq_nasal / d_der_nasal
        if not (self.yaw_min <= yaw_ratio <= self.yaw_max):
            return None, "DISCARDED_YAW_PITCH"

        # ----------------------------------------------------------------------
        # Corrección de rotación (Roll)
        # ----------------------------------------------------------------------
        d_eyes = p_canto_der - p_canto_izq
        angle = np.degrees(np.arctan2(d_eyes[1], d_eyes[0]))

        center_eyes = (
            float((p_canto_izq[0] + p_canto_der[0]) / 2.0),
            float((p_canto_izq[1] + p_canto_der[1]) / 2.0)
        )

        rot_mat = cv2.getRotationMatrix2D(center_eyes, angle, 1.0)
        rotated_frame = cv2.warpAffine(frame_bgr, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

        ones = np.ones(shape=(len(coords), 1))
        coords_homo = np.hstack([coords, ones])
        aligned_coords = rot_mat.dot(coords_homo.T).T

        # ----------------------------------------------------------------------
        # Extracción de parches cuadrados por envolventes anatómicas
        # ----------------------------------------------------------------------
        pts_izq = aligned_coords[self.PTS_PERIOCULAR_IZQ]
        pts_der = aligned_coords[self.PTS_PERIOCULAR_DER]
        pts_boca = aligned_coords[self.PTS_BOCA]

        box_izq = self._get_square_box_from_points(pts_izq, PADDING_PERIOCULAR, w, h)
        box_der = self._get_square_box_from_points(pts_der, PADDING_PERIOCULAR, w, h)
        box_boca = self._get_square_box_from_points(pts_boca, PADDING_BOCA, w, h)

        if box_izq is None or box_der is None or box_boca is None:
            return None, "OUT_OF_BOUNDS"

        crop_l = rotated_frame[box_izq[0]:box_izq[1], box_izq[2]:box_izq[3]]
        crop_r = rotated_frame[box_der[0]:box_der[1], box_der[2]:box_der[3]]
        crop_m = rotated_frame[box_boca[0]:box_boca[1], box_boca[2]:box_boca[3]]

        # ----------------------------------------------------------------------
        # Normalización visual (Grayscale + Resizing 64x64 + CLAHE)
        # ----------------------------------------------------------------------
        patches = {
            "periocular_izq": self._normalize_patch(crop_l),
            "periocular_der": self._normalize_patch(crop_r),
            "boca": self._normalize_patch(crop_m)
        }

        if (
            patches["periocular_izq"] is None
            or patches["periocular_der"] is None
            or patches["boca"] is None
        ):
            return None, "OUT_OF_BOUNDS"

        return patches, "OK"