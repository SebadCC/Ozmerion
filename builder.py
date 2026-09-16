# builder.py
import os
import stat
import time
import shutil
import cv2
import numpy as np
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from patch import GeometricPatchExtractor

# ==============================================================================
# RUTAS DEL SISTEMA DE ARCHIVOS
# ==============================================================================
ROOT_DATA = "Data"
DIR_MUESTRAS = os.path.join(ROOT_DATA, "muestras")
DIR_DATASET = os.path.join(ROOT_DATA, "dataset")

DIR_PERIOCULAR_IZQ = os.path.join(DIR_DATASET, "periocular_izq")
DIR_PERIOCULAR_DER = os.path.join(DIR_DATASET, "periocular_der")
DIR_BOCA = os.path.join(DIR_DATASET, "boca")
DIR_INVALIDAS = os.path.join(DIR_DATASET, "invalidas")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ==============================================================================
# UMBRALES BIOMECÁNICOS CALIBRADOS
# ==============================================================================
DETECTION_CONFIDENCE = 0.4

# Cejas: relación respecto al ancho intrínseco de cada ceja
BROW_ARCH_HIGH_THRESHOLD = 0.28   # Altura de curvatura (ceja elevada)
BROW_SLOPE_LOW_THRESHOLD = 0.18   # Caída del extremo interno respecto al externo (ceja descendida)

# Ojos: apertura párpados respecto a la distancia interocular (IOD)
EYE_OPEN_THRESHOLD = 0.08

# Boca: apertura respecto al grosor del labio superior
MOUTH_OPEN_LIP_RATIO = 4.0

# Comisuras: elevación/depresión vertical respecto a la línea media labial normalizada por IOD
CORNER_UP_THRESHOLD = 0.10
CORNER_DOWN_THRESHOLD = -0.08


def remove_readonly(func, path, exc_info):
    """Manejador de errores para Windows: limpia el flag de solo lectura y reintenta."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


class DatasetBuilder:
    # Cantos internos oculares para IOD y escala
    CANTO_INT_IZQ = 133
    CANTO_INT_DER = 362

    # Tríada anatómica Ceja Izquierda (Interno, Centro/Arco, Externo)
    CEJA_IN_IZQ = 55
    CEJA_MID_IZQ = 105
    CEJA_OUT_IZQ = 46

    # Tríada anatómica Ceja Derecha (Interno, Centro/Arco, Externo)
    CEJA_IN_DER = 285
    CEJA_MID_DER = 334
    CEJA_OUT_DER = 276

    # Párpados
    PARPADO_SUP_IZQ = 159
    PARPADO_INF_IZQ = 145
    PARPADO_SUP_DER = 386
    PARPADO_INF_DER = 374

    # Estructura bucal
    COMISURA_IZQ = 61
    COMISURA_DER = 291
    LABIO_EXTERNO_SUP = 0
    LABIO_INTERNO_SUP = 13
    LABIO_INTERNO_INF = 14

    def __init__(self):
        self.extractor = GeometricPatchExtractor(target_size=(64, 64))
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=DETECTION_CONFIDENCE
        )

    def _safe_clear_directory(self, target_dir):
        """Limpia el contenido interno de un directorio sin destruir la raíz."""
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
                print(f"[AVISO] No se pudo eliminar temporalmente {item_path}: {e}")

    def _reset_dataset_workspace(self):
        self._safe_clear_directory(DIR_DATASET)
        os.makedirs(DIR_PERIOCULAR_IZQ, exist_ok=True)
        os.makedirs(DIR_PERIOCULAR_DER, exist_ok=True)
        os.makedirs(DIR_BOCA, exist_ok=True)
        os.makedirs(DIR_INVALIDAS, exist_ok=True)

    def _classify_eyebrow(self, p_in, p_mid, p_out) -> str:
        brow_width = max(1e-4, float(np.linalg.norm(p_out - p_in)))

        # 1. Descenso/Fruncido: caída del extremo interno (coordenada Y mayor hacia abajo en imagen)
        slope_in = (p_in[1] - p_out[1]) / brow_width
        if slope_in > BROW_SLOPE_LOW_THRESHOLD:
            return "descendida"

        # 2. Elevación: altura de la cúspide central respecto a la cuerda base
        y_base_interp = (p_in[1] + p_out[1]) / 2.0
        arch_height = (y_base_interp - p_mid[1]) / brow_width
        if arch_height > BROW_ARCH_HIGH_THRESHOLD:
            return "elevada"

        return "neutra"

    def _classify_face_dynamics(self, coords):
        iod = np.linalg.norm(coords[self.CANTO_INT_DER] - coords[self.CANTO_INT_IZQ])
        if iod < 1e-6:
            return None, None

        # 1. Periocular Izquierdo
        ceja_izq = self._classify_eyebrow(
            coords[self.CEJA_IN_IZQ],
            coords[self.CEJA_MID_IZQ],
            coords[self.CEJA_OUT_IZQ]
        )
        dist_ojo_izq = np.linalg.norm(coords[self.PARPADO_SUP_IZQ] - coords[self.PARPADO_INF_IZQ]) / iod
        ojo_izq = "cerrado" if dist_ojo_izq < EYE_OPEN_THRESHOLD else "abierto"
        label_periocular_izq = f"ceja_{ceja_izq}_ojo_{ojo_izq}"

        # 2. Periocular Derecho
        ceja_der = self._classify_eyebrow(
            coords[self.CEJA_IN_DER],
            coords[self.CEJA_MID_DER],
            coords[self.CEJA_OUT_DER]
        )
        dist_ojo_der = np.linalg.norm(coords[self.PARPADO_SUP_DER] - coords[self.PARPADO_INF_DER]) / iod
        ojo_der = "cerrado" if dist_ojo_der < EYE_OPEN_THRESHOLD else "abierto"
        label_periocular_der = f"ceja_{ceja_der}_ojo_{ojo_der}"

        # 3. Boca y Comisuras
        grosor_labio_sup = max(1e-4, float(coords[self.LABIO_INTERNO_SUP][1] - coords[self.LABIO_EXTERNO_SUP][1]))
        cavidad_interna = max(0.0, float(coords[self.LABIO_INTERNO_INF][1] - coords[self.LABIO_INTERNO_SUP][1]))
        boca_estado = "abierta" if (cavidad_interna >= MOUTH_OPEN_LIP_RATIO * grosor_labio_sup) else "cerrada"

        y_centro_labios = (coords[self.LABIO_INTERNO_SUP][1] + coords[self.LABIO_INTERNO_INF][1]) / 2.0
        y_comisuras = (coords[self.COMISURA_IZQ][1] + coords[self.COMISURA_DER][1]) / 2.0
        diff_comisuras = (y_centro_labios - y_comisuras) / iod

        if diff_comisuras > CORNER_UP_THRESHOLD:
            comisuras_estado = "arriba"
        elif diff_comisuras < CORNER_DOWN_THRESHOLD:
            comisuras_estado = "abajo"
        else:
            comisuras_estado = "neutra"

        label_boca = f"comisuras_{comisuras_estado}_{boca_estado}"

        return (label_periocular_izq, label_periocular_der), label_boca

    def process(self):
        if not os.path.exists(DIR_MUESTRAS):
            print(f"[ERROR] No se encontró el directorio de muestras en: '{DIR_MUESTRAS}'")
            return

        self._reset_dataset_workspace()

        voluntarios = sorted([
            v for v in os.listdir(DIR_MUESTRAS)
            if os.path.isdir(os.path.join(DIR_MUESTRAS, v))
        ])

        if not voluntarios:
            print(f"[AVISO] No se encontraron carpetas de voluntarios en: '{DIR_MUESTRAS}'")
            return

        total_imagenes = 0
        total_validas = 0
        total_descartes = 0

        print(f"\n[INFO] Iniciando procesamiento de {len(voluntarios)} carpetas de voluntarios...")

        for vol in voluntarios:
            vol_dir = os.path.join(DIR_MUESTRAS, vol)
            archivos = sorted([
                f for f in os.listdir(vol_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            ])

            if not archivos:
                continue

            print(f"  -> Procesando {vol}: {len(archivos)} imágenes detectadas")

            for idx, fname in enumerate(archivos, start=1):
                total_imagenes += 1
                img_path = os.path.join(vol_dir, fname)
                frame = cv2.imread(img_path)

                if frame is None:
                    continue

                # 1. Extracción y control de pose/roll
                patches, status = self.extractor.extract_patches(frame)

                if patches is None or status != "OK":
                    total_descartes += 1
                    descarte_name = f"{vol}_{idx}.jpg"
                    descarte_path = os.path.join(DIR_INVALIDAS, descarte_name)
                    cv2.imwrite(descarte_path, frame)
                    continue

                # 2. Análisis biomecánico de puntos anatómicos
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.face_mesh.process(rgb)

                if not res.multi_face_landmarks:
                    total_descartes += 1
                    descarte_name = f"{vol}_{idx}.jpg"
                    descarte_path = os.path.join(DIR_INVALIDAS, descarte_name)
                    cv2.imwrite(descarte_path, frame)
                    continue

                mesh = res.multi_face_landmarks[0].landmark
                coords = np.array([[lm.x * w, lm.y * h] for lm in mesh], dtype=np.float32)

                labels_periocular, label_boca = self._classify_face_dynamics(coords)
                if labels_periocular is None or label_boca is None:
                    total_descartes += 1
                    descarte_name = f"{vol}_{idx}.jpg"
                    descarte_path = os.path.join(DIR_INVALIDAS, descarte_name)
                    cv2.imwrite(descarte_path, frame)
                    continue

                label_p_izq, label_p_der = labels_periocular

                # 3. Guardado con nomenclatura canónica {vol}_{idx}_{parche}.png
                # Periocular Izquierdo
                out_dir_izq = os.path.join(DIR_PERIOCULAR_IZQ, label_p_izq)
                os.makedirs(out_dir_izq, exist_ok=True)
                path_p_izq = os.path.join(out_dir_izq, f"{vol}_{idx}_periocular_izq.png")
                cv2.imwrite(path_p_izq, patches["periocular_izq"])

                # Periocular Derecho
                out_dir_der = os.path.join(DIR_PERIOCULAR_DER, label_p_der)
                os.makedirs(out_dir_der, exist_ok=True)
                path_p_der = os.path.join(out_dir_der, f"{vol}_{idx}_periocular_der.png")
                cv2.imwrite(path_p_der, patches["periocular_der"])

                # Boca
                out_dir_boca = os.path.join(DIR_BOCA, label_boca)
                os.makedirs(out_dir_boca, exist_ok=True)
                path_boca = os.path.join(out_dir_boca, f"{vol}_{idx}_boca.png")
                cv2.imwrite(path_boca, patches["boca"])

                total_validas += 1

        print("\n==================================================")
        print("RESUMEN DE PROCESAMIENTO (BUILDER)")
        print(f"  - Total imágenes analizadas : {total_imagenes}")
        print(f"  - Muestras válidas (tríadas): {total_validas}")
        print(f"  - Total parches guardados   : {total_validas * 3}")
        print(f"  - Imágenes descartadas      : {total_descartes} (en Data/dataset/invalidas/)")
        print(f"  - Directorio de salida      : {DIR_DATASET}")
        print("==================================================\n")


if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.process()