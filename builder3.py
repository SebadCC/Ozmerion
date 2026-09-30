# builder3.py
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
# 2. UMBRALES DE ACTIVACIÓN NEUROMUSCULAR (BLENDSHAPES CO-ACTIVADOS)
# ==============================================================================
# Ojos: Párpado y tensión orbitaria
TH_EYE_BLINK = 0.38
TH_EYE_SQUINT_BIAS = 0.20
TH_EYE_WIDE = 0.25

# Cejas: Corrugador, elevadores frontales y soporte nasal
TH_BROW_DOWN = 0.28
TH_BROW_UP_MEDIAL = 0.28
TH_BROW_UP_LATERAL = 0.28
TH_NOSE_SNEER_SUPPORT = 0.22

# Mandíbula y Apertura Labial
TH_JAW_OPEN = 0.18
TH_LIP_SEPARATION = 0.20

# Comisuras Bucales: Cigomático vs Depresores
TH_MOUTH_SMILE = 0.25
TH_MOUTH_FROWN = 0.22
TH_CHEEK_SQUINT_SUPPORT = 0.20

# Margen porcentual de histeresis para fronteras de duda
UNCERTAINTY_MARGIN = 0.10

# ==============================================================================
# 3. RUTAS DEL SISTEMA DE ARCHIVOS
# ==============================================================================
ROOT_DATA = "Data"
DIR_RAW_READY = os.path.join(ROOT_DATA, "raw", "ready")
DIR_DATASET_READY = os.path.join(ROOT_DATA, "dataset", "ready", "builder3")

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


class DatasetBuilderBlendshapes:
    def __init__(self):
        self.extractor = TasksGeometricPatchExtractor(
            caller="builder2",  # Fingimos ser builder2 para que patch2 entregue los 3 elementos
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

    def _is_uncertain(self, value: float, threshold: float, margin: float = UNCERTAINTY_MARGIN) -> bool:
        delta = abs(threshold * margin)
        return (threshold - delta) <= value <= (threshold + delta)

    def _classify_eye_state(self, bs: dict, side: str) -> str:
        s = "Left" if side == "izq" else "Right"
        blink = bs.get(f"eyeBlink{s}", 0.0)
        squint = bs.get(f"eyeSquint{s}", 0.0)
        wide = bs.get(f"eyeWide{s}", 0.0)
        look_down = bs.get(f"eyeLookDown{s}", 0.0)

        # Ajuste de cierre corregido por mirada hacia abajo
        # Si la persona mira hacia abajo, el blink tiende a inflarse pasivamente
        blink_efectivo = blink - (0.25 * look_down) + (0.15 * squint)

        if self._is_uncertain(blink_efectivo, TH_EYE_BLINK):
            return "incertidumbre"

        if blink_efectivo >= TH_EYE_BLINK and wide < 0.15:
            return "cerrado"

        if blink_efectivo < TH_EYE_BLINK:
            return "abierto"

        return "incertidumbre"

    def _classify_eyebrow_state(self, bs: dict, side: str) -> str:
        s = "Left" if side == "izq" else "Right"

        brow_down = bs.get(f"browDown{s}", 0.0)
        brow_outer_up = bs.get(f"browOuterUp{s}", 0.0)
        brow_inner_up = bs.get("browInnerUp", 0.0)
        nose_sneer = bs.get(f"noseSneer{s}", 0.0)
        cheek_squint = bs.get(f"cheekSquint{s}", 0.0)

        # 1. Puntuación compuesta de ceño fruncido (AU4 + sinergia nasal)
        score_descenso = brow_down + (0.20 * nose_sneer) + (0.10 * cheek_squint)

        # 2. Puntuación compuesta de ceja elevada (AU1 medial + AU2 lateral)
        score_elevacion = max(brow_outer_up, brow_inner_up)

        delta_down = TH_BROW_DOWN * UNCERTAINTY_MARGIN
        delta_up = TH_BROW_UP_LATERAL * UNCERTAINTY_MARGIN

        es_descendida = score_descenso > (TH_BROW_DOWN + delta_down)
        es_elevada = score_elevacion > (TH_BROW_UP_LATERAL + delta_up)

        # Conflicto muscular simultáneo
        if es_descendida and es_elevada:
            return "incertidumbre"

        # Zonas de frontera
        if self._is_uncertain(score_descenso, TH_BROW_DOWN) or self._is_uncertain(score_elevacion, TH_BROW_UP_LATERAL):
            return "incertidumbre"

        # Activación nítida
        if es_descendida and score_elevacion < (TH_BROW_UP_LATERAL - delta_up):
            return "descendida"

        if es_elevada and score_descenso < (TH_BROW_DOWN - delta_down):
            return "elevada"

        # Reposo / Neutra
        if score_descenso <= (TH_BROW_DOWN - delta_down) and score_elevacion <= (TH_BROW_UP_LATERAL - delta_up):
            return "neutra"

        return "incertidumbre"

    def _classify_mouth_state(self, bs: dict) -> str:
        jaw_open = bs.get("jawOpen", 0.0)
        mouth_stretch = (bs.get("mouthStretchLeft", 0.0) + bs.get("mouthStretchRight", 0.0)) / 2.0
        lip_lower = (bs.get("mouthLowerDownLeft", 0.0) + bs.get("mouthLowerDownRight", 0.0)) / 2.0
        lip_upper = (bs.get("mouthUpperUpLeft", 0.0) + bs.get("mouthUpperUpRight", 0.0)) / 2.0

        # Score compuesto de apertura oral
        score_apertura = jaw_open + (0.25 * lip_lower) + (0.15 * lip_upper) + (0.10 * mouth_stretch)

        if self._is_uncertain(score_apertura, TH_JAW_OPEN):
            boca_apertura = "incertidumbre"
        else:
            boca_apertura = "abierta" if score_apertura >= TH_JAW_OPEN else "cerrada"

        # Evaluación de comisuras
        smile = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2.0
        dimple = (bs.get("mouthDimpleLeft", 0.0) + bs.get("mouthDimpleRight", 0.0)) / 2.0
        cheek_sq = (bs.get("cheekSquintLeft", 0.0) + bs.get("cheekSquintRight", 0.0)) / 2.0

        frown = (bs.get("mouthFrownLeft", 0.0) + bs.get("mouthFrownRight", 0.0)) / 2.0
        shrug_lower = bs.get("mouthShrugLower", 0.0)

        # Sinergias compuestas
        score_smile = smile + (0.20 * dimple) + (0.15 * cheek_sq)
        score_frown = frown + (0.20 * shrug_lower)

        delta_smile = TH_MOUTH_SMILE * UNCERTAINTY_MARGIN
        delta_frown = TH_MOUTH_FROWN * UNCERTAINTY_MARGIN

        activo_smile = score_smile > (TH_MOUTH_SMILE + delta_smile)
        activo_frown = score_frown > (TH_MOUTH_FROWN + delta_frown)

        if activo_smile and activo_frown:
            comisuras = "incertidumbre"
        elif self._is_uncertain(score_smile, TH_MOUTH_SMILE) or self._is_uncertain(score_frown, TH_MOUTH_FROWN):
            comisuras = "incertidumbre"
        elif activo_smile and score_frown < (TH_MOUTH_FROWN - delta_frown):
            comisuras = "arriba"
        elif activo_frown and score_smile < (TH_MOUTH_SMILE - delta_smile):
            comisuras = "abajo"
        elif score_smile <= (TH_MOUTH_SMILE - delta_smile) and score_frown <= (TH_MOUTH_FROWN - delta_frown):
            comisuras = "neutra"
        else:
            comisuras = "incertidumbre"

        if boca_apertura == "incertidumbre" or comisuras == "incertidumbre":
            return "incertidumbre"

        return f"comisuras_{comisuras}_{boca_apertura}"

    def _classify_face_dynamics_bs(self, bs: dict):
        if not bs:
            return None, None, None

        # Periocular Izquierdo
        ojo_izq = self._classify_eye_state(bs, "izq")
        ceja_izq = self._classify_eyebrow_state(bs, "izq")
        if ojo_izq == "incertidumbre" or ceja_izq == "incertidumbre":
            label_p_izq = "incertidumbre"
        else:
            label_p_izq = f"ceja_{ceja_izq}_ojo_{ojo_izq}"

        # Periocular Derecho
        ojo_der = self._classify_eye_state(bs, "der")
        ceja_der = self._classify_eyebrow_state(bs, "der")
        if ojo_der == "incertidumbre" or ceja_der == "incertidumbre":
            label_p_der = "incertidumbre"
        else:
            label_p_der = f"ceja_{ceja_der}_ojo_{ojo_der}"

        # Complejo Bucal
        label_boca = self._classify_mouth_state(bs)

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
            "origen": "builder3",
            "motor_parches": "patch2",
            "metodo": "FACS_FaceBlendshapes_Synergies",
            "timestamp": timestamp,
            "configuracion": {
                "max_pitch_deg": MAX_PITCH_DEG,
                "max_yaw_deg": MAX_YAW_DEG,
                "max_roll_deg": MAX_ROLL_DEG,
                "padding_periocular": PADDING_PERIOCULAR,
                "padding_boca": PADDING_BOCA,
                "th_eye_blink": TH_EYE_BLINK,
                "th_brow_down": TH_BROW_DOWN,
                "th_brow_up_lateral": TH_BROW_UP_LATERAL,
                "th_jaw_open": TH_JAW_OPEN,
                "th_mouth_smile": TH_MOUTH_SMILE,
                "th_mouth_frown": TH_MOUTH_FROWN,
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
                "imagenes_descartadas_pose_o_sin_blendshapes": total_descartes
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

        print(f"\n[INFO] Iniciando procesamiento con Blendshapes y sinergias musculares (Builder3)...")

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

                # 1. Extracción geométrica de parches y telemetría (fingiendo caller='builder2')
                patches, status, telemetry = self.extractor.extract_patches(frame)
                if patches is None or status != "OK" or telemetry is None:
                    total_descartes += 1
                    cv2.imwrite(os.path.join(DIR_INVALIDAS, f"{vol}_{idx}.jpg"), frame)
                    continue

                # 2. Deserialización robusta de blendshapes desde patch2
                raw_bs = telemetry.get("blendshapes")
                if not raw_bs:
                    total_descartes += 1
                    cv2.imwrite(os.path.join(DIR_INVALIDAS, f"{vol}_{idx}.jpg"), frame)
                    continue

                if isinstance(raw_bs, dict):
                    blendshapes = raw_bs
                else:
                    # Convierte la lista/objeto CategoryList de MediaPipe a un dict {nombre: score}
                    blendshapes = {b.category_name: float(b.score) for b in raw_bs}

                # 3. Clasificación neuromuscular robusta
                label_p_izq, label_p_der, label_boca = self._classify_face_dynamics_bs(blendshapes)
                if label_p_izq is None or label_p_der is None or label_boca is None:
                    total_descartes += 1
                    cv2.imwrite(os.path.join(DIR_INVALIDAS, f"{vol}_{idx}.jpg"), frame)
                    continue

                # 4. Guardado enrutando incertidumbre a sus carpetas correspondientes
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
        print("RESUMEN DE PROCESAMIENTO (BUILDER3 CON BLENDSHAPES Y FACS)")
        print(f"  - Total imágenes analizadas       : {total_imagenes}")
        print(f"  - Total parches guardados         : {total_guardadas}")
        print(f"  - Parches en INCERTIDUMBRE        : {total_incertidumbre}")
        print(f"      * Periocular Izquierdo        : {inc_izq}")
        print(f"      * Periocular Derecho          : {inc_der}")
        print(f"      * Complejo Bucal              : {inc_boca}")
        print(f"  - Descartadas (pose o sin BS)     : {total_descartes} (en {DIR_INVALIDAS})")
        print(f"  - Directorio de salida            : {DIR_DATASET_READY}")
        print("==================================================\n")


if __name__ == "__main__":
    builder = DatasetBuilderBlendshapes()
    builder.process()