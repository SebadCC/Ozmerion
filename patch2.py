# patch2.py
import os
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Binario canónico inmutable de MediaPipe Tasks para patch2
MODEL_TASK_NAME = "face_landmarker_v2_with_blendshapes.task"


class TasksGeometricPatchExtractor:
    # Cantos internos oculares para IOD y cálculo de Roll
    CANTO_EXT_IZQ = 33
    CANTO_INT_IZQ = 133
    CANTO_INT_DER = 362
    CANTO_EXT_DER = 263

    # Límites Ceja Izquierda
    CEJA_IN_IZQ = 55
    CEJA_TOP_IZQ = 105
    CEJA_OUT_IZQ = 46

    # Límites Ceja Derecha
    CEJA_IN_DER = 285
    CEJA_TOP_DER = 334
    CEJA_OUT_DER = 276

    # Límites Ojo Izquierdo
    PARPADO_SUP_IZQ = 159
    PARPADO_INF_IZQ = 145

    # Límites Ojo Derecho
    PARPADO_SUP_DER = 386
    PARPADO_INF_DER = 374

    # Límites Boca
    COMISURA_IZQ = 61
    COMISURA_DER = 291
    LABIO_EXT_SUP = 0
    LABIO_EXT_INF = 17

    def __init__(
        self,
        caller: str,
        max_pitch: float,
        max_yaw: float,
        max_roll: float,
        min_detection_confidence: float,
        min_presence_confidence: float,
        min_tracking_confidence: float,
        padding_periocular: float,
        padding_boca: float,
        target_size: tuple = (64, 64)
    ):
        self.caller = caller.lower().strip()
        self.max_pitch = float(max_pitch)
        self.max_yaw = float(max_yaw)
        self.max_roll = float(max_roll)
        self.padding_periocular = float(padding_periocular)
        self.padding_boca = float(padding_boca)
        self.target_size = target_size

        if not os.path.exists(MODEL_TASK_NAME):
            raise FileNotFoundError(
                f"[ERROR] No se encontró el binario de MediaPipe en: '{MODEL_TASK_NAME}'"
            )

        # Solo computamos blendshapes si el llamador es builder2
        need_blendshapes = (self.caller == "builder2")

        base_options = python.BaseOptions(model_asset_path=MODEL_TASK_NAME)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_facial_transformation_matrixes=True,
            output_face_blendshapes=need_blendshapes
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _extract_euler_angles(self, matrix_4x4: np.ndarray) -> tuple:
        """Calcula Pitch, Yaw y Roll en grados a partir de la matriz 3x3 de rotación."""
        R = matrix_4x4[:3, :3]
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(R[2, 1], R[2, 2])
            yaw = math.atan2(-R[2, 0], sy)
            roll = math.atan2(R[1, 0], R[0, 0])
        else:
            pitch = math.atan2(-R[1, 2], R[1, 1])
            yaw = math.atan2(-R[2, 0], sy)
            roll = 0.0

        return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)

    def _crop_and_normalize(self, image_gray: np.ndarray, box_coords: tuple) -> np.ndarray:
        """Recorta la caja delimitadora, redimensiona a target_size y aplica CLAHE."""
        x_min, y_min, x_max, y_max = box_coords
        h, w = image_gray.shape

        x_min = max(0, int(round(x_min)))
        y_min = max(0, int(round(y_min)))
        x_max = min(w, int(round(x_max)))
        y_max = min(h, int(round(y_max)))

        if (x_max - x_min) <= 0 or (y_max - y_min) <= 0:
            return np.zeros(self.target_size, dtype=np.uint8)

        crop = image_gray[y_min:y_max, x_min:x_max]
        resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_AREA)
        return self.clahe.apply(resized)

    def extract_patches(self, frame: np.ndarray):
        """
        Procesa el frame y extrae los 3 parches anatómicos.
        
        Retorno dinámico:
            - Si caller == "builder2":
                retorna (patches, status, telemetry)
            - Si caller != "builder2" (ej. "realtime", "retrainer"):
                retorna (patches, status)
        """
        if frame is None:
            if self.caller == "builder2":
                return None, "NO_FRAME", None
            return None, "NO_FRAME"

        h, w, _ = frame.shape

        # Inferencia nativa sRGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)

        # 1. Validación de rostro
        if not result.face_landmarks or len(result.face_landmarks) == 0:
            if self.caller == "builder2":
                return None, "NO_FACE", None
            return None, "NO_FACE"

        # 2. Control de pose angular
        matrix = result.facial_transformation_matrixes[0]
        pitch_deg, yaw_deg, roll_deg = self._extract_euler_angles(matrix)

        if (abs(pitch_deg) > self.max_pitch or
            abs(yaw_deg) > self.max_yaw or
            abs(roll_deg) > self.max_roll):
            if self.caller == "builder2":
                telemetry_rejected = {
                    "angles": (pitch_deg, yaw_deg, roll_deg),
                    "landmarks": None
                }
                return None, "EXCEEDED_MAX_ANGLE", telemetry_rejected
            return None, "EXCEEDED_MAX_ANGLE"

        # 3. Conversión de coordenadas de la malla facial a píxeles
        mesh = result.face_landmarks[0]
        coords = np.array([[lm.x * w, lm.y * h] for lm in mesh], dtype=np.float32)

        # 4. Escala de grises para el recorte
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # A. Periocular Izquierdo
        pts_p_izq = coords[[
            self.CANTO_EXT_IZQ, self.CANTO_INT_IZQ,
            self.CEJA_IN_IZQ, self.CEJA_TOP_IZQ, self.CEJA_OUT_IZQ,
            self.PARPADO_SUP_IZQ, self.PARPADO_INF_IZQ
        ]]
        x_min, y_min = np.min(pts_p_izq, axis=0)
        x_max, y_max = np.max(pts_p_izq, axis=0)
        bw = x_max - x_min
        bh = y_max - y_min
        side = max(bw, bh) * (1.0 + self.padding_periocular)
        cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
        box_p_izq = (cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0)

        # B. Periocular Derecho
        pts_p_der = coords[[
            self.CANTO_EXT_DER, self.CANTO_INT_DER,
            self.CEJA_IN_DER, self.CEJA_TOP_DER, self.CEJA_OUT_DER,
            self.PARPADO_SUP_DER, self.PARPADO_INF_DER
        ]]
        x_min, y_min = np.min(pts_p_der, axis=0)
        x_max, y_max = np.max(pts_p_der, axis=0)
        bw = x_max - x_min
        bh = y_max - y_min
        side = max(bw, bh) * (1.0 + self.padding_periocular)
        cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
        box_p_der = (cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0)

        # C. Boca
        pts_boca = coords[[
            self.COMISURA_IZQ, self.COMISURA_DER,
            self.LABIO_EXT_SUP, self.LABIO_EXT_INF
        ]]
        x_min, y_min = np.min(pts_boca, axis=0)
        x_max, y_max = np.max(pts_boca, axis=0)
        bw = x_max - x_min
        bh = y_max - y_min
        side = max(bw, bh) * (1.0 + self.padding_boca)
        cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
        box_boca = (cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0)

        # 5. Generación de parches
        patches = {
            "periocular_izq": self._crop_and_normalize(frame_gray, box_p_izq),
            "periocular_der": self._crop_and_normalize(frame_gray, box_p_der),
            "boca": self._crop_and_normalize(frame_gray, box_boca)
        }

        # 6. Adaptación de salida según el llamador
        if self.caller == "builder2":
            telemetry = {
                "angles": (pitch_deg, yaw_deg, roll_deg),
                "landmarks": coords,
                "raw_mesh": mesh,
                "blendshapes": result.face_blendshapes[0] if result.face_blendshapes else None,
                "boxes": {
                    "periocular_izq": box_p_izq,
                    "periocular_der": box_p_der,
                    "boca": box_boca
                }
            }
            return patches, "OK", telemetry

        return patches, "OK"