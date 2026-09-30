import os
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

# ==============================================================================
# CONFIGURACIÓN GENERAL DEL SAMPLER
# ==============================================================================
INPUT_DIR = "Data/raw/archive/UvA/"
OUTPUT_DIR = "Data/raw/archive/UvA Frames/"
MODEL_PATH = "face_landmarker_v2_with_blendshapes.task"

MAX_ANGLE_LIMIT = 80          # Límite angular en grados (Yaw/Pitch) con respecto a la cámara
TARGET_FRAMES_PER_VIDEO = 20    # Cuota exacta de fotogramas a extraer por video
VIDEO_EXTENSIONS = (".mp4", ".mpeg", ".avi", ".mov", ".mkv", ".mpg")


# ==============================================================================
# FUNCIONES AUXILIARES DE TRANSFORMACIÓN Y CINEMÁTICA
# ==============================================================================
def extract_euler_angles_from_matrix(matrix_4x4: np.ndarray):
    """
    Extrae los ángulos de Euler (Yaw, Pitch, Roll) en grados a partir de la 
    matriz de transformación facial rígida 4x4 provista por MediaPipe Tasks.
    """
    R = matrix_4x4[:3, :3]
    
    # Descomposición de matriz de rotación ortogonal (secuencia Y-X-Z)
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

    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    roll_deg = math.degrees(roll)

    return abs(pitch_deg), abs(yaw_deg), abs(roll_deg)


def select_hierarchical_frames(valid_frames: list, scores: list, target_count: int) -> list:
    """
    Selecciona índices de frames siguiendo la jerarquía estricta:
    1. Inicio y Fin (2 frames obligatorios).
    2. Picos, Valles y Mesetas en la curva de actividad muscular.
    3. Puntos de mayor variación (velocidad cinemática de cambio).
    4. Relleno distribuido uniformemente en caso de falta de dinamismo.
    
    Retorna la lista de índices reales del video, ordenados cronológicamente.
    """
    total_valid = len(valid_frames)
    if total_valid <= target_count:
        return [valid_frames[i] for i in range(total_valid)]

    selected_indices = set()

    # 1. Jerarquía 1: Inicio y Fin exactos
    selected_indices.add(0)
    selected_indices.add(total_valid - 1)

    if len(selected_indices) >= target_count:
        final_list = [valid_frames[i] for i in sorted(selected_indices)]
        return final_list[:target_count]

    # Derivadas de primer orden (velocidad de cambio del gesto)
    scores_arr = np.array(scores, dtype=np.float32)
    velocity = np.abs(np.diff(scores_arr, prepend=scores_arr[0]))

    # 2. Jerarquía 2: Picos, Valles y Mesetas
    extrema_candidates = []
    for i in range(1, total_valid - 1):
        prev_val = scores_arr[i - 1]
        curr_val = scores_arr[i]
        next_val = scores_arr[i + 1]

        is_peak = curr_val >= prev_val and curr_val > next_val
        is_valley = curr_val <= prev_val and curr_val < next_val
        is_plateau = abs(curr_val - prev_val) < 1e-4 and abs(curr_val - next_val) < 1e-4

        if is_peak or is_valley or is_plateau:
            # Prominencia relativa respecto a los vecinos inmediatos
            prominence = abs(curr_val - (prev_val + next_val) / 2.0)
            extrema_candidates.append((prominence, i))

    # Ordenar extremos por prominencia descendente
    extrema_candidates.sort(key=lambda x: x[0], reverse=True)

    for _, idx in extrema_candidates:
        if len(selected_indices) >= target_count:
            break
        selected_indices.add(idx)

    # 3. Jerarquía 3: Puntos de gran variación cinemática (máxima velocidad)
    if len(selected_indices) < target_count:
        velocity_candidates = [
            (velocity[i], i) for i in range(total_valid) if i not in selected_indices
        ]
        velocity_candidates.sort(key=lambda x: x[0], reverse=True)

        for _, idx in velocity_candidates:
            if len(selected_indices) >= target_count:
                break
            selected_indices.add(idx)

    # 4. Jerarquía 4: Distribución uniforme para cupos sobrantes
    if len(selected_indices) < target_count:
        uniform_indices = np.linspace(0, total_valid - 1, num=target_count, dtype=int)
        for idx in uniform_indices:
            if idx not in selected_indices:
                selected_indices.add(idx)
            if len(selected_indices) >= target_count:
                break

    # Si aún faltara por colisiones de redondeo, llenar secuencialmente
    if len(selected_indices) < target_count:
        for i in range(total_valid):
            if i not in selected_indices:
                selected_indices.add(i)
            if len(selected_indices) >= target_count:
                break

    # Orden cronológico según el video
    sorted_relative_indices = sorted(list(selected_indices))
    return [valid_frames[i] for i in sorted_relative_indices]


# ==============================================================================
# MOTOR PRINCIPAL DE PROCESAMIENTO
# ==============================================================================
def create_landmarker(model_path: str):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"No se encontró el modelo MediaPipe Tasks en: {model_path}"
        )

    base_options = BaseOptions(model_asset_path=model_path)
    options = FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    return FaceLandmarker.create_from_options(options)


def process_video(video_path: str, output_dir: str, landmarker):
    video_filename = os.path.basename(video_path)
    video_stem, _ = os.path.splitext(video_filename)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo decodificar el video: {video_path}")
        return

    valid_frame_indices = []
    biomechanical_scores = []
    frame_count = 0

    # FASE 1: Análisis ligero de poses y cinemática en streaming (RAM mínima)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)

        # Validar presencia de rostro, blendshapes y matriz de pose
        if (detection_result.face_blendshapes and 
            detection_result.facial_transformation_matrixes):
            
            matrix = np.array(detection_result.facial_transformation_matrixes[0])
            pitch, yaw, _ = extract_euler_angles_from_matrix(matrix)

            # Filtro por límite angular respecto a cámara
            if pitch <= MAX_ANGLE_LIMIT and yaw <= MAX_ANGLE_LIMIT:
                # La magnitud de movimiento se mide sumando los pesos de los 52 blendshapes
                blendshapes = detection_result.face_blendshapes[0]
                total_activation = sum(b.score for b in blendshapes)

                valid_frame_indices.append(frame_count)
                biomechanical_scores.append(total_activation)

        frame_count += 1

    cap.release()

    if not valid_frame_indices:
        print(f"[SKIP] Video sin frames válidos o rostro fuera de ángulo: {video_filename}")
        return

    # Selección jerárquica de índices objetivos
    target_indices = select_hierarchical_frames(
        valid_frames=valid_frame_indices,
        scores=biomechanical_scores,
        target_count=TARGET_FRAMES_PER_VIDEO
    )

    # FASE 2: Extracción y guardado ordenado cronológicamente (1 a N)
    target_set = set(target_indices)
    cap = cv2.VideoCapture(video_path)
    
    current_frame_idx = 0
    saved_counter = 1

    while True:
        ret, frame = cap.read()
        if not ret or saved_counter > len(target_indices):
            break

        if current_frame_idx in target_set:
            output_name = f"{video_stem}_{saved_counter}.jpg"
            output_filepath = os.path.join(output_dir, output_name)
            cv2.imwrite(output_filepath, frame)
            saved_counter += 1

        current_frame_idx += 1

    cap.release()
    print(f"[OK] {video_filename} -> {saved_counter - 1} frames extraídos en orden temporal.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    landmarker = create_landmarker(MODEL_PATH)

    print("=" * 80)
    print("OZMERION - SAMPLER CINEMÁTICO DE VIDEO")
    print(f"Directorio Origen: {INPUT_DIR}")
    print(f"Directorio Destino: {OUTPUT_DIR}")
    print(f"Límite Angular: <= {MAX_ANGLE_LIMIT}° | Frames por video: {TARGET_FRAMES_PER_VIDEO}")
    print("=" * 80)

    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(VIDEO_EXTENSIONS):
                full_video_path = os.path.join(root, file)
                process_video(full_video_path, OUTPUT_DIR, landmarker)

    print("\n[PROCESO COMPLETADO] Extracción finalizada en la carpeta destino.")


if __name__ == "__main__":
    main()