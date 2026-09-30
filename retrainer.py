# retrainer.py
import os
import json
import time
from datetime import datetime
import cv2
import numpy as np

# ==============================================================================
# CONFIGURACIÓN Y PARÁMETROS PRINCIPALES
# ==============================================================================
VERSION = 10                  # Versión del modelo ONNX a utilizar como oráculo
CONFIDENCE_THRESHOLD = 0.90   # Umbral mínimo de certeza matemática (90%)
PATCHER_VERSION = 2           # 1: patch.py (FaceMesh) | 2: patch2.py (Tasks FaceLandmarker)
ENABLE_PREVIEW = True         # True: HUD visual fluido. False: consola pura

ROOT_DATA = "Data"
DIR_VERSION_ONNX = os.path.join(ROOT_DATA, "models", f"v{VERSION}", "onnx")
PERIOCULAR_MODEL_PATH = os.path.join(DIR_VERSION_ONNX, "model_periocular.onnx")
BOCA_MODEL_PATH = os.path.join(DIR_VERSION_ONNX, "model_boca.onnx")

DIR_RAW_READY = os.path.join(ROOT_DATA, "raw", "ready")
DIR_DATASET_READY = os.path.join(ROOT_DATA, "dataset", "ready", "retrainer")

DIR_PERIOCULAR_IZQ = os.path.join(DIR_DATASET_READY, "periocular_izq")
DIR_PERIOCULAR_DER = os.path.join(DIR_DATASET_READY, "periocular_der")
DIR_BOCA = os.path.join(DIR_DATASET_READY, "boca")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Mapeos canónicos de clases
EYEBROW_LABELS = {0: "descendida", 1: "neutra", 2: "elevada"}
EYE_LABELS = {0: "cerrado", 1: "abierto"}
COMMISSURE_LABELS = {0: "abajo", 1: "neutra", 2: "arriba"}
MOUTH_OPEN_LABELS = {0: "cerrada", 1: "abierta"}

# Selección dinámica de extractor
if PATCHER_VERSION == 2:
    from patch2 import TasksGeometricPatchExtractor as PatchExtractor
else:
    from patch import GeometricPatchExtractor as PatchExtractor


def softmax_raw(logits):
    """Softmax a temperatura natural (T=1.0) para evaluar la certeza real."""
    exp_z = np.exp(logits - np.max(logits))
    return exp_z / np.sum(exp_z)


class RetrainerPipeline:
    def __init__(self):
        if not os.path.exists(PERIOCULAR_MODEL_PATH) or not os.path.exists(BOCA_MODEL_PATH):
            raise FileNotFoundError(
                f"[ERROR] No se encontraron los modelos de la versión v{VERSION} en:\n"
                f"  - {PERIOCULAR_MODEL_PATH}\n"
                f"  - {BOCA_MODEL_PATH}\n"
                "Verifica que la carpeta exista o ajusta la variable VERSION."
            )

        print(f"[INFO] Cargando modelos ONNX oráculo: v{VERSION}...")
        self.net_periocular = cv2.dnn.readNetFromONNX(PERIOCULAR_MODEL_PATH)
        self.net_mouth = cv2.dnn.readNetFromONNX(BOCA_MODEL_PATH)

        self.net_periocular.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net_periocular.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.net_mouth.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net_mouth.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.out_names_p = self.net_periocular.getUnconnectedOutLayersNames()
        self.out_names_m = self.net_mouth.getUnconnectedOutLayersNames()

        self.extractor = PatchExtractor(target_size=(64, 64))

        os.makedirs(DIR_PERIOCULAR_IZQ, exist_ok=True)
        os.makedirs(DIR_PERIOCULAR_DER, exist_ok=True)
        os.makedirs(DIR_BOCA, exist_ok=True)

    def _infer_periocular(self, patch):
        blob = (patch.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]
        self.net_periocular.setInput(blob)
        outputs = self.net_periocular.forward(self.out_names_p)

        out_ceja = outputs[0] if outputs[0].shape[-1] == 3 else outputs[1]
        out_ojo = outputs[1] if outputs[0].shape[-1] == 3 else outputs[0]

        prob_ceja = softmax_raw(out_ceja[0])
        prob_ojo = softmax_raw(out_ojo[0])

        idx_ceja = int(np.argmax(prob_ceja))
        idx_ojo = int(np.argmax(prob_ojo))

        return (
            EYEBROW_LABELS.get(idx_ceja, "neutra"),
            float(prob_ceja[idx_ceja]),
            EYE_LABELS.get(idx_ojo, "abierto"),
            float(prob_ojo[idx_ojo])
        )

    def _infer_mouth(self, patch):
        blob = (patch.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]
        self.net_mouth.setInput(blob)
        outputs = self.net_mouth.forward(self.out_names_m)

        out_com = outputs[0] if outputs[0].shape[-1] == 3 else outputs[1]
        out_ap = outputs[1] if outputs[0].shape[-1] == 3 else outputs[0]

        prob_com = softmax_raw(out_com[0])
        prob_ap = softmax_raw(out_ap[0])

        idx_com = int(np.argmax(prob_com))
        idx_ap = int(np.argmax(prob_ap))

        return (
            COMMISSURE_LABELS.get(idx_com, "neutra"),
            float(prob_com[idx_com]),
            MOUTH_OPEN_LABELS.get(idx_ap, "cerrada"),
            float(prob_ap[idx_ap])
        )

    def _draw_hud_box(self, img, pt1, pt2, alpha=0.55):
        x1, y1 = max(0, pt1[0]), max(0, pt1[1])
        x2, y2 = min(img.shape[1], pt2[0]), min(img.shape[0], pt2[1])
        if x2 <= x1 or y2 <= y1:
            return
        sub_img = img[y1:y2, x1:x2]
        black_rect = np.zeros_like(sub_img)
        res = cv2.addWeighted(sub_img, 1.0 - alpha, black_rect, alpha, 1.0)
        img[y1:y2, x1:x2] = res

    def _apply_letterboxing(self, frame, target_w, target_h):
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        dx = (target_w - new_w) // 2
        dy = (target_h - new_h) // 2
        canvas[dy:dy + new_h, dx:dx + new_w] = resized
        return canvas

    def _generate_class_distribution(self):
        distribution = {
            "periocular_izq": {},
            "periocular_der": {},
            "boca": {}
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

        return distribution

    def _write_metadata(self, lotes_procesados, total_analizadas, total_descartes, stats_parches):
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M")
        metadata = {
            "origen": "retrainer",
            "version_oraculo": f"v{VERSION}",
            "patcher_utilizado": f"patch{PATCHER_VERSION}",
            "umbral_confianza": CONFIDENCE_THRESHOLD,
            "timestamp": timestamp,
            "lotes_procesados": lotes_procesados,
            "resumen_global": {
                "total_imagenes_analizadas": total_analizadas,
                "imagenes_descartadas_pose": total_descartes,
                "parches_guardados_izq": stats_parches["izq"],
                "parches_guardados_der": stats_parches["der"],
                "parches_guardados_boca": stats_parches["boca"],
                "total_parches_rescatados": sum(stats_parches.values())
            },
            "distribucion_parches": self._generate_class_distribution()
        }

        metadata_path = os.path.join(DIR_DATASET_READY, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Manifiesto de retrainer guardado en: {metadata_path}")

    def process_all(self):
        if not os.path.exists(DIR_RAW_READY):
            print(f"[ERROR] No se encontró el directorio de entrada en: '{DIR_RAW_READY}'")
            return

        voluntarios = sorted([
            v for v in os.listdir(DIR_RAW_READY)
            if os.path.isdir(os.path.join(DIR_RAW_READY, v))
        ])

        if not voluntarios:
            print(f"[AVISO] No se encontraron carpetas de lotes en: '{DIR_RAW_READY}'")
            return

        window_name = f"Retrainer Oráculo - v{VERSION} (Patcher v{PATCHER_VERSION})"
        if ENABLE_PREVIEW:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        total_analizadas = 0
        stats_parches = {"izq": 0, "der": 0, "boca": 0}
        total_descartes_pose = 0

        fps = 0.0
        prev_time = time.time()

        print(f"\n[INFO] Iniciando pseudo-etiquetado con umbral >= {int(CONFIDENCE_THRESHOLD * 100)}%...")

        for vol in voluntarios:
            vol_dir = os.path.join(DIR_RAW_READY, vol)
            archivos = sorted([
                f for f in os.listdir(vol_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            ])

            if not archivos:
                continue

            print(f"  -> Procesando lote: {vol} ({len(archivos)} imágenes)")

            for idx, fname in enumerate(archivos, start=1):
                total_analizadas += 1
                curr_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
                prev_time = curr_time

                img_path = os.path.join(vol_dir, fname)
                frame = cv2.imread(img_path)

                if frame is None:
                    continue

                h_frame, w_frame, _ = frame.shape
                patches, status = self.extractor.extract_patches(frame)

                preds_ready = False
                ceja_izq = ceja_der = ojo_izq = ojo_der = com_boca = ap_boca = "N/A"
                conf_c_izq = conf_c_der = conf_o_izq = conf_o_der = conf_com = conf_ap = 0.0

                saved_izq = saved_der = saved_boca = False

                if patches is None or status != "OK":
                    total_descartes_pose += 1
                else:
                    preds_ready = True

                    # 1. Periocular Izquierdo
                    ceja_izq, conf_c_izq, ojo_izq, conf_o_izq = self._infer_periocular(patches["periocular_izq"])
                    if conf_c_izq >= CONFIDENCE_THRESHOLD and conf_o_izq >= CONFIDENCE_THRESHOLD:
                        subfolder_izq = f"ceja_{ceja_izq}_ojo_{ojo_izq}"
                        dir_out_izq = os.path.join(DIR_PERIOCULAR_IZQ, subfolder_izq)
                        os.makedirs(dir_out_izq, exist_ok=True)
                        cv2.imwrite(
                            os.path.join(dir_out_izq, f"{vol}_{idx}_periocular_izq.png"),
                            patches["periocular_izq"]
                        )
                        stats_parches["izq"] += 1
                        saved_izq = True

                    # 2. Periocular Derecho
                    ceja_der, conf_c_der, ojo_der, conf_o_der = self._infer_periocular(patches["periocular_der"])
                    if conf_c_der >= CONFIDENCE_THRESHOLD and conf_o_der >= CONFIDENCE_THRESHOLD:
                        subfolder_der = f"ceja_{ceja_der}_ojo_{ojo_der}"
                        dir_out_der = os.path.join(DIR_PERIOCULAR_DER, subfolder_der)
                        os.makedirs(dir_out_der, exist_ok=True)
                        cv2.imwrite(
                            os.path.join(dir_out_der, f"{vol}_{idx}_periocular_der.png"),
                            patches["periocular_der"]
                        )
                        stats_parches["der"] += 1
                        saved_der = True

                    # 3. Boca
                    com_boca, conf_com, ap_boca, conf_ap = self._infer_mouth(patches["boca"])
                    if conf_com >= CONFIDENCE_THRESHOLD and conf_ap >= CONFIDENCE_THRESHOLD:
                        subfolder_boca = f"comisuras_{com_boca}_{ap_boca}"
                        dir_out_boca = os.path.join(DIR_BOCA, subfolder_boca)
                        os.makedirs(dir_out_boca, exist_ok=True)
                        cv2.imwrite(
                            os.path.join(dir_out_boca, f"{vol}_{idx}_boca.png"),
                            patches["boca"]
                        )
                        stats_parches["boca"] += 1
                        saved_boca = True

                # --------------------------------------------------------------
                # RENDERIZADO VISUAL
                # --------------------------------------------------------------
                if ENABLE_PREVIEW:
                    display_frame = frame.copy()

                    if status == "OK":
                        display_status = "Procesando"
                        status_color = (0, 255, 0)
                    elif "YAW" in status or "PITCH" in status:
                        display_status = "Discarded_Yaw_Pitch"
                        status_color = (0, 165, 255)
                    elif "NO_FACE" in status:
                        display_status = "No Face"
                        status_color = (0, 0, 255)
                    else:
                        display_status = status
                        status_color = (0, 200, 255)

                    # Formateo visual estricto para evitar overflow del 100%
                    def fmt_conf(c):
                        val = int(round(c * 100))
                        return "100%" if val >= 100 else f"{val}%"

                    # 1. Superior Izquierda
                    self._draw_hud_box(display_frame, (10, 10), (320, 65), alpha=0.6)
                    cv2.putText(display_frame, display_status, (18, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1, cv2.LINE_AA)
                    cv2.putText(display_frame, f"{vol} | {fname[:20]}", (18, 54),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

                    # 2. Superior Derecha
                    self._draw_hud_box(display_frame, (w_frame - 180, 10), (w_frame - 10, 65), alpha=0.6)
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (w_frame - 170, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 1, cv2.LINE_AA)
                    cv2.putText(display_frame, f"Umbral: {int(CONFIDENCE_THRESHOLD*100)}%", (w_frame - 170, 53),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

                    # 3. Inferior Izquierda: Telemetría anatómica con indicación [+]
                    self._draw_hud_box(display_frame, (10, h_frame - 145), (290, h_frame - 10), alpha=0.6)
                    y_base = h_frame - 128
                    line_step = 21

                    badge_izq = "[+]" if saved_izq else "[-]"
                    badge_der = "[+]" if saved_der else "[-]"
                    badge_boca = "[+]" if saved_boca else "[-]"

                    labels_info = [
                        (f"Ojo Izq: {ojo_izq} ({fmt_conf(conf_o_izq)}) {badge_izq}", saved_izq),
                        (f"Ojo Der: {ojo_der} ({fmt_conf(conf_o_der)}) {badge_der}", saved_der),
                        (f"Ceja Izq: {ceja_izq} ({fmt_conf(conf_c_izq)})", False),
                        (f"Ceja Der: {ceja_der} ({fmt_conf(conf_c_der)})", False),
                        (f"Apertura Boca: {ap_boca} ({fmt_conf(conf_ap)}) {badge_boca}", saved_boca),
                        (f"Comisuras: {com_boca} ({fmt_conf(conf_com)})", False)
                    ]

                    for i, (text, is_saved) in enumerate(labels_info):
                        col = (0, 255, 0) if is_saved else (255, 255, 255)
                        cv2.putText(display_frame, text, (18, y_base + (i * line_step)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

                    # 4. Inferior Derecha: Miniaturas
                    if patches is not None:
                        self._draw_hud_box(display_frame, (w_frame - 232, h_frame - 92), (w_frame - 5, h_frame - 8), alpha=0.6)
                        p_izq_bgr = cv2.cvtColor(patches["periocular_izq"], cv2.COLOR_GRAY2BGR)
                        p_der_bgr = cv2.cvtColor(patches["periocular_der"], cv2.COLOR_GRAY2BGR)
                        p_boca_bgr = cv2.cvtColor(patches["boca"], cv2.COLOR_GRAY2BGR)

                        margin_y = h_frame - 76
                        display_frame[margin_y:margin_y + 64, w_frame - 74:w_frame - 10] = p_boca_bgr
                        display_frame[margin_y:margin_y + 64, w_frame - 148:w_frame - 84] = p_der_bgr
                        display_frame[margin_y:margin_y + 64, w_frame - 222:w_frame - 158] = p_izq_bgr

                        col_izq = (0, 255, 0) if saved_izq else (150, 150, 150)
                        col_der = (0, 255, 0) if saved_der else (150, 150, 150)
                        col_boca = (0, 255, 0) if saved_boca else (150, 150, 150)

                        cv2.putText(display_frame, "Izq", (w_frame - 205, margin_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col_izq, 1, cv2.LINE_AA)
                        cv2.putText(display_frame, "Der", (w_frame - 130, margin_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col_der, 1, cv2.LINE_AA)
                        cv2.putText(display_frame, "Boca", (w_frame - 60, margin_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col_boca, 1, cv2.LINE_AA)

                    try:
                        win_rect = cv2.getWindowImageRect(window_name)
                        win_w, win_h = win_rect[2], win_rect[3]
                    except Exception:
                        win_w, win_h = w_frame, h_frame

                    if win_w > 100 and win_h > 100 and (win_w != w_frame or win_h != h_frame):
                        render_frame = self._apply_letterboxing(display_frame, win_w, win_h)
                    else:
                        render_frame = display_frame

                    cv2.imshow(window_name, render_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("\n[AVISO] Proceso interrumpido por el usuario.")
                        cv2.destroyAllWindows()
                        return

        if ENABLE_PREVIEW:
            cv2.destroyAllWindows()

        self._write_metadata(voluntarios, total_analizadas, total_descartes_pose, stats_parches)

        print("\n==================================================")
        print("RESUMEN DE PSEUDO-ETIQUETADO (RETRAINER)")
        print(f"  - Total fotos analizadas   : {total_analizadas}")
        print(f"  - Descartes por pose/rostro: {total_descartes_pose}")
        print(f"  - Parches P. Izquierdo     : {stats_parches['izq']}")
        print(f"  - Parches P. Derecho       : {stats_parches['der']}")
        print(f"  - Parches Boca             : {stats_parches['boca']}")
        print(f"  - Total parches rescatados : {sum(stats_parches.values())}")
        print(f"  - Directorio de salida     : {DIR_DATASET_READY}")
        print("==================================================\n")


if __name__ == "__main__":
    retrainer = RetrainerPipeline()
    retrainer.process_all()