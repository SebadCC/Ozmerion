# builder2.py
import os
import re
import stat
import time
import json
import shutil
from datetime import datetime
import cv2
import numpy as np

from patch2 import TasksGeometricPatchExtractor

# ==============================================================================
# 1. PARÁMETROS CONFIGURABLES DEL EXTRACTOR (PATCH2)
# ==============================================================================
MAX_PITCH_DEG = 20.0
MAX_YAW_DEG = 20.0
MAX_ROLL_DEG = 15.0

MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

PADDING_PERIOCULAR = 0.45
PADDING_BOCA = 0.45

# ==============================================================================
# 2. UMBRALES BIOMECÁNICOS CALIBRADOS
# ==============================================================================
# Ojos: Relación de Aspecto Ocular (EAR)
EAR_OPEN_THRESHOLD = 0.19

# Boca: Relación de Aspecto Bucal (MAR)
MAR_OPEN_THRESHOLD = 0.12

# Comisuras: Desviación vertical respecto a IOD
CORNER_UP_THRESHOLD = 0.08
CORNER_DOWN_THRESHOLD = -0.06

# Cejas: Cotas estrictas de aproximación medial y colapso óseo (normalizados por IOD)
BROW_CORRUGATOR_EUCLIDEAN_TH = 0.25  # Distancia 2D ceja_in -> sellion
BROW_DIST_IN_Y_TH = 0.39             # Caída vertical estricta ceja_in -> canto interno óseo
BROW_DIST_IN_X_TH = 0.18             # Desplazamiento horizontal ceja_in -> sellion (eje medio)
BROW_ARCH_HIGH_THRESHOLD = 0.32      # Elevación de curvatura del arco ciliar

# ==============================================================================
# 3. CONTROL DE INCERTIDUMBRE (HISTÉRESIS POR FRONTERAS)
# ==============================================================================
UNCERTAINTY_MARGIN = 0.08  # Margen del 8% para evitar absorción de neutras basales

# ==============================================================================
# 4. RUTAS DEL SISTEMA DE ARCHIVOS
# ==============================================================================
ROOT_DATA = "Data"
DIR_RAW_READY = os.path.join(ROOT_DATA, "raw", "ready")
DIR_DATASET_READY = os.path.join(ROOT_DATA, "dataset", "ready", "builder2")

DIR_PERIOCULAR_IZQ = os.path.join(DIR_DATASET_READY, "periocular_izq")
DIR_PERIOCULAR_DER = os.path.join(DIR_DATASET_READY, "periocular_der")
DIR_BOCA = os.path.join(DIR_DATASET_READY, "boca")
DIR_INVALIDAS = os.path.join(DIR_DATASET_READY, "invalidas")

DIR_INCERTIDUMBRE = os.path.join(DIR_DATASET_READY, "incertidumbre")
DIR_INC_PERIOCULAR_IZQ = os.path.join(DIR_INCERTIDUMBRE, "periocular_izq")
DIR_INC_PERIOCULAR_DER = os.path.join(DIR_INCERTIDUMBRE, "periocular_der")
DIR_INC_BOCA = os.path.join(DIR_INCERTIDUMBRE, "boca")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


class DatasetBuilderTasks:
    # Landmarks anatómicos canónicos
    CANTO_INT_IZQ = 133
    CANTO_EXT_IZQ = 33
    CANTO_INT_DER = 362
    CANTO_EXT_DER = 263
    SELLION = 168

    # Ceja Izquierda
    CEJA_IN_IZQ = 55
    CEJA_MID_IZQ = 105
    CEJA_OUT_IZQ = 46

    # Ceja Derecha
    CEJA_IN_DER = 285
    CEJA_MID_DER = 334
    CEJA_OUT_DER = 276

    # Párpados (EAR 6 puntos)
    OJO_IZQ_PTS = [33, 160, 158, 133, 153, 144]
    OJO_DER_PTS = [263, 385, 387, 362, 373, 380]

    # Complejo Labial
    COMISURA_IZQ = 61
    COMISURA_DER = 291
    LABIO_EXTERNO_SUP = 0
    LABIO_INTERNO_SUP = 13
    LABIO_INTERNO_INF = 14
    LABIO_EXTERNO_INF = 17

    def __init__(self):
        self.extractor = TasksGeometricPatchExtractor(
            caller="builder2",
            max_pitch=MAX_PITCH_DEG,
            max_yaw=MAX_YAW_DEG,
            max_roll=MAX_ROLL_DEG,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_presence_confidence=MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            padding_periocular=PADDING_PERIOCULAR,
            padding_boca=PADDING_BOCA,
            target_size=(64, 64)
        )

    def _safe_clear_directory(self, target_dir):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            return

        for item in os.listdir(target_dir):
            item_path = os.path.join(target_dir, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, onerror=remove_readonly)
                else:
                    try:
                        os.chmod(item_path, stat.S_IWRITE)
                        os.remove(item_path)
                    except PermissionError:
                        time.sleep(0.05)
                        os.remove(item_path)
            except Exception as e:
                print(f"[AVISO] No se pudo eliminar {item_path}: {e}")

    def _reset_dataset_workspace(self):
        self._safe_clear_directory(DIR_DATASET_READY)
        os.makedirs(DIR_PERIOCULAR_IZQ, exist_ok=True)
        os.makedirs(DIR_PERIOCULAR_DER, exist_ok=True)
        os.makedirs(DIR_BOCA, exist_ok=True)
        os.makedirs(DIR_INVALIDAS, exist_ok=True)
        os.makedirs(DIR_INC_PERIOCULAR_IZQ, exist_ok=True)
        os.makedirs(DIR_INC_PERIOCULAR_DER, exist_ok=True)
        os.makedirs(DIR_INC_BOCA, exist_ok=True)

    def _is_uncertain(self, value: float, threshold: float, margin: float) -> bool:
        delta = abs(threshold * margin)
        return (threshold - delta) <= value <= (threshold + delta)

    def _calculate_ear(self, coords, eye_pts):
        p1, p2, p3, p4, p5, p6 = coords[eye_pts]
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        h = np.linalg.norm(p1 - p4)
        if h < 1e-4:
            return 0.0
        return float((v1 + v2) / (2.0 * h))

    def _classify_eyebrow(self, p_in, p_mid, p_out, p_canto_int, p_sellion, iod) -> str:
        brow_width = max(1e-4, float(np.linalg.norm(p_out - p_in)))

        # 1. Componentes de tracción del corrugador normalizados por IOD
        dist_in_y = float((p_canto_int[1] - p_in[1]) / iod)
        dist_in_x = float(abs(p_in[0] - p_sellion[0]) / iod)
        dist_euclidiana_sellion = float(np.linalg.norm(p_in - p_sellion) / iod)

        # Deltas de incertidumbre para la frontera baja
        delta_euclid = BROW_CORRUGATOR_EUCLIDEAN_TH * UNCERTAINTY_MARGIN
        delta_in_y = BROW_DIST_IN_Y_TH * UNCERTAINTY_MARGIN
        delta_in_x = BROW_DIST_IN_X_TH * UNCERTAINTY_MARGIN

        # Criterio compuesto de ceño fruncido / ceja descendida
        fruncimiento_severo = dist_euclidiana_sellion < (BROW_CORRUGATOR_EUCLIDEAN_TH - delta_euclid)
        colapso_coordinado = (dist_in_y < (BROW_DIST_IN_Y_TH - delta_in_y)) and \
                             (dist_in_x < (BROW_DIST_IN_X_TH - delta_in_x))

        if fruncimiento_severo or colapso_coordinado:
            return "descendida"

        # Verificación de banda de incertidumbre en frontera descendida vs neutra
        duda_euclid = self._is_uncertain(dist_euclidiana_sellion, BROW_CORRUGATOR_EUCLIDEAN_TH, UNCERTAINTY_MARGIN)
        duda_coordinada = self._is_uncertain(dist_in_y, BROW_DIST_IN_Y_TH, UNCERTAINTY_MARGIN) and \
                          (dist_in_x < (BROW_DIST_IN_X_TH + delta_in_x))

        if duda_euclid or duda_coordinada:
            return "incertidumbre"

        # 2. Frontera alta: elevación del arco ciliar
        y_base_interp = (p_in[1] + p_out[1]) / 2.0
        arch_height = float((y_base_interp - p_mid[1]) / brow_width)

        delta_arch = BROW_ARCH_HIGH_THRESHOLD * UNCERTAINTY_MARGIN
        if arch_height > (BROW_ARCH_HIGH_THRESHOLD + delta_arch):
            return "elevada"

        if self._is_uncertain(arch_height, BROW_ARCH_HIGH_THRESHOLD, UNCERTAINTY_MARGIN):
            return "incertidumbre"

        # 3. Estado basal de reposo
        return "neutra"

    def _classify_face_dynamics(self, coords):
        iod = np.linalg.norm(coords[self.CANTO_INT_DER] - coords[self.CANTO_INT_IZQ])
        if iod < 1e-6:
            return None, None, None

        sellion = coords[self.SELLION]

        # 1. Periocular Izquierdo
        ceja_izq = self._classify_eyebrow(
            coords[self.CEJA_IN_IZQ], coords[self.CEJA_MID_IZQ], coords[self.CEJA_OUT_IZQ],
            coords[self.CANTO_INT_IZQ], sellion, iod
        )
        ear_izq = self._calculate_ear(coords, self.OJO_IZQ_PTS)
        if self._is_uncertain(ear_izq, EAR_OPEN_THRESHOLD, UNCERTAINTY_MARGIN) or ceja_izq == "incertidumbre":
            label_p_izq = "incertidumbre"
        else:
            ojo_izq = "cerrado" if ear_izq < EAR_OPEN_THRESHOLD else "abierto"
            label_p_izq = f"ceja_{ceja_izq}_ojo_{ojo_izq}"

        # 2. Periocular Derecho
        ceja_der = self._classify_eyebrow(
            coords[self.CEJA_IN_DER], coords[self.CEJA_MID_DER], coords[self.CEJA_OUT_DER],
            coords[self.CANTO_INT_DER], sellion, iod
        )
        ear_der = self._calculate_ear(coords, self.OJO_DER_PTS)
        if self._is_uncertain(ear_der, EAR_OPEN_THRESHOLD, UNCERTAINTY_MARGIN) or ceja_der == "incertidumbre":
            label_p_der = "incertidumbre"
        else:
            ojo_der = "cerrado" if ear_der < EAR_OPEN_THRESHOLD else "abierto"
            label_p_der = f"ceja_{ceja_der}_ojo_{ojo_der}"

        # 3. Complejo Bucal
        ancho_boca = np.linalg.norm(coords[self.COMISURA_DER] - coords[self.COMISURA_IZQ])
        if ancho_boca < 1e-4:
            return None, None, None

        altura_interna = np.linalg.norm(coords[self.LABIO_INTERNO_INF] - coords[self.LABIO_INTERNO_SUP])
        mar = float(altura_interna / ancho_boca)

        y_centro_labios = (coords[self.LABIO_INTERNO_SUP][1] + coords[self.LABIO_INTERNO_INF][1]) / 2.0
        y_comisuras = (coords[self.COMISURA_IZQ][1] + coords[self.COMISURA_DER][1]) / 2.0
        diff_comisuras = (y_centro_labios - y_comisuras) / iod

        if self._is_uncertain(mar, MAR_OPEN_THRESHOLD, UNCERTAINTY_MARGIN):
            return label_p_izq, label_p_der, "incertidumbre"

        boca_estado = "abierta" if (mar >= MAR_OPEN_THRESHOLD) else "cerrada"

        delta_down = abs(CORNER_DOWN_THRESHOLD * UNCERTAINTY_MARGIN)
        delta_up = abs(CORNER_UP_THRESHOLD * UNCERTAINTY_MARGIN)

        if diff_comisuras < (CORNER_DOWN_THRESHOLD - delta_down):
            comisuras_estado = "abajo"
        elif self._is_uncertain(diff_comisuras, CORNER_DOWN_THRESHOLD, UNCERTAINTY_MARGIN):
            comisuras_estado = "incertidumbre"
        elif diff_comisuras > (CORNER_UP_THRESHOLD + delta_up):
            comisuras_estado = "arriba"
        elif self._is_uncertain(diff_comisuras, CORNER_UP_THRESHOLD, UNCERTAINTY_MARGIN):
            comisuras_estado = "incertidumbre"
        else:
            comisuras_estado = "neutra"

        if comisuras_estado == "incertidumbre":
            label_boca = "incertidumbre"
        else:
            label_boca = f"comisuras_{comisuras_estado}_{boca_estado}"

        return label_p_izq, label_p_der, label_boca

    def _generate_class_distribution(self):
        distribution = {
            "periocular_izq": {},
            "periocular_der": {},
            "boca": {},
            "incertidumbre": {}
        }

        for region, target_dir in [("periocular_izq", DIR_PERIOCULAR_IZQ),
                                   ("periocular_der", DIR_PERIOCULAR_DER),
                                   ("boca", DIR_BOCA)]:
            if os.path.exists(target_dir):
                for folder in sorted(os.listdir(target_dir)):
                    folder_path = os.path.join(target_dir, folder)
                    if os.path.isdir(folder_path):
                        count = len([f for f in os.listdir(folder_path) if f.lower().endswith(".png")])
                        distribution[region][folder] = count

        for region, target_dir in [("periocular_izq", DIR_INC_PERIOCULAR_IZQ),
                                   ("periocular_der", DIR_INC_PERIOCULAR_DER),
                                   ("boca", DIR_INC_BOCA)]:
            if os.path.exists(target_dir):
                distribution["incertidumbre"][region] = len([f for f in os.listdir(target_dir) if f.lower().endswith(".png")])

        return distribution

    def _write_metadata(self, lotes_procesados, total_analizadas, total_guardadas, total_descartes):
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M")
        distribucion = self._generate_class_distribution()

        inc_izq = distribucion["incertidumbre"].get("periocular_izq", 0)
        inc_der = distribucion["incertidumbre"].get("periocular_der", 0)
        inc_boca = distribucion["incertidumbre"].get("boca", 0)
        total_incertidumbre = inc_izq + inc_der + inc_boca

        metadata = {
            "origen": "builder2",
            "motor_parches": "patch2",
            "timestamp": timestamp,
            "configuracion": {
                "max_pitch_deg": MAX_PITCH_DEG,
                "max_yaw_deg": MAX_YAW_DEG,
                "max_roll_deg": MAX_ROLL_DEG,
                "padding_periocular": PADDING_PERIOCULAR,
                "padding_boca": PADDING_BOCA,
                "mar_open_threshold": MAR_OPEN_THRESHOLD,
                "ear_open_threshold": EAR_OPEN_THRESHOLD,
                "brow_corrugator_euclidean_th": BROW_CORRUGATOR_EUCLIDEAN_TH,
                "brow_dist_in_y_th": BROW_DIST_IN_Y_TH,
                "brow_dist_in_x_th": BROW_DIST_IN_X_TH,
                "brow_arch_high_threshold": BROW_ARCH_HIGH_THRESHOLD,
                "uncertainty_margin": UNCERTAINTY_MARGIN
            },
            "lotes_procesados": lotes_procesados,
            "resumen_global": {
                "total_imagenes_analizadas": total_analizadas,
                "total_parches_guardados": total_guardadas,
                "total_parches_incertidumbre": total_incertidumbre,
                "desglose_incertidumbre": {
                    "periocular_izq": inc_izq,
                    "periocular_der": inc_der,
                    "boca": inc_boca
                },
                "imagenes_descartadas_pose": total_descartes
            },
            "distribucion_parches": distribucion
        }

        metadata_path = os.path.join(DIR_DATASET_READY, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Manifiesto guardado en: {metadata_path}")
        return distribucion

    def process(self):
        if not os.path.exists(DIR_RAW_READY):
            print(f"[ERROR] No se encontró el directorio de entrada en: '{DIR_RAW_READY}'")
            return

        self._reset_dataset_workspace()

        voluntarios = sorted([
            v for v in os.listdir(DIR_RAW_READY)
            if os.path.isdir(os.path.join(DIR_RAW_READY, v))
        ], key=natural_sort_key)

        if not voluntarios:
            print(f"[AVISO] No se encontraron carpetas de lotes en: '{DIR_RAW_READY}'")
            return

        total_imagenes = 0
        total_guardadas = 0
        total_descartes = 0

        print(f"\n[INFO] Iniciando procesamiento estricto con vector biomecánico corrugador...")

        for vol in voluntarios:
            vol_dir = os.path.join(DIR_RAW_READY, vol)
            archivos = sorted([
                f for f in os.listdir(vol_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            ], key=natural_sort_key)

            if not archivos:
                continue

            print(f"  -> Procesando {vol}: {len(archivos)} imágenes detectadas")

            for idx, fname in enumerate(archivos, start=1):
                total_imagenes += 1
                img_path = os.path.join(vol_dir, fname)
                frame = cv2.imread(img_path)

                if frame is None:
                    continue

                patches, status, telemetry = self.extractor.extract_patches(frame)
                if patches is None or status != "OK" or telemetry is None:
                    total_descartes += 1
                    cv2.imwrite(os.path.join(DIR_INVALIDAS, f"{vol}_{idx}.jpg"), frame)
                    continue

                coords = telemetry["landmarks"]

                label_p_izq, label_p_der, label_boca = self._classify_face_dynamics(coords)
                if label_p_izq is None or label_p_der is None or label_boca is None:
                    total_descartes += 1
                    cv2.imwrite(os.path.join(DIR_INVALIDAS, f"{vol}_{idx}.jpg"), frame)
                    continue

                # Periocular Izquierdo
                if label_p_izq == "incertidumbre":
                    out_p_izq = DIR_INC_PERIOCULAR_IZQ
                else:
                    out_p_izq = os.path.join(DIR_PERIOCULAR_IZQ, label_p_izq)
                    os.makedirs(out_p_izq, exist_ok=True)
                cv2.imwrite(os.path.join(out_p_izq, f"{vol}_{idx}_periocular_izq.png"), patches["periocular_izq"])
                total_guardadas += 1

                # Periocular Derecho
                if label_p_der == "incertidumbre":
                    out_p_der = DIR_INC_PERIOCULAR_DER
                else:
                    out_p_der = os.path.join(DIR_PERIOCULAR_DER, label_p_der)
                    os.makedirs(out_p_der, exist_ok=True)
                cv2.imwrite(os.path.join(out_p_der, f"{vol}_{idx}_periocular_der.png"), patches["periocular_der"])
                total_guardadas += 1

                # Boca
                if label_boca == "incertidumbre":
                    out_boca = DIR_INC_BOCA
                else:
                    out_boca = os.path.join(DIR_BOCA, label_boca)
                    os.makedirs(out_boca, exist_ok=True)
                cv2.imwrite(os.path.join(out_boca, f"{vol}_{idx}_boca.png"), patches["boca"])
                total_guardadas += 1

        distribucion = self._write_metadata(voluntarios, total_imagenes, total_guardadas, total_descartes)

        inc_izq = distribucion["incertidumbre"].get("periocular_izq", 0)
        inc_der = distribucion["incertidumbre"].get("periocular_der", 0)
        inc_boca = distribucion["incertidumbre"].get("boca", 0)
        total_incertidumbre = inc_izq + inc_der + inc_boca

        print("\n==================================================")
        print("RESUMEN DE PROCESAMIENTO (BUILDER2 CON BANDA DE INCERTIDUMBRE)")
        print(f"  - Total imágenes analizadas       : {total_imagenes}")
        print(f"  - Total parches guardados         : {total_guardadas}")
        print(f"  - Parches en INCERTIDUMBRE        : {total_incertidumbre}")
        print(f"      * Periocular Izquierdo        : {inc_izq}")
        print(f"      * Periocular Derecho          : {inc_der}")
        print(f"      * Complejo Bucal              : {inc_boca}")
        print(f"  - Imágenes descartadas por pose   : {total_descartes} (en {DIR_INVALIDAS})")
        print(f"  - Directorio de salida            : {DIR_DATASET_READY}")
        print("==================================================\n")


if __name__ == "__main__":
    builder = DatasetBuilderTasks()
    builder.process()