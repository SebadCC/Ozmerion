 # retrainer.py
import os
import time
import cv2
import numpy as np
from patch import GeometricPatchExtractor

# ==============================================================================
# CONFIGURACIÓN Y PARÁMETROS PRINCIPALES
# ==============================================================================
VERSION = 9  # Versión del modelo ONNX a utilizar como juez/oráculo
CONFIDENCE_THRESHOLD = 0.80  # Umbral mínimo de certeza matemática para aceptar un parche
ENABLE_PREVIEW = True  # True: HUD visual fluido (~30-40 fps). False: máxima velocidad por consola

ROOT_DATA = "Data"
DIR_VERSION_ONNX = os.path.join(ROOT_DATA, "versiones", f"v{VERSION}", "onnx")
PERIOCULAR_MODEL_PATH = os.path.join(DIR_VERSION_ONNX, "model_periocular.onnx")
BOCA_MODEL_PATH = os.path.join(DIR_VERSION_ONNX, "model_boca.onnx")

DIR_MUESTRAS = os.path.join(ROOT_DATA, "muestras")
DIR_REDATASET = os.path.join(ROOT_DATA, "Redataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Mapeos canónicos de clases
EYEBROW_LABELS = {0: "descendida", 1: "neutra", 2: "elevada"}
EYE_LABELS = {0: "cerrado", 1: "abierto"}
COMMISSURE_LABELS = {0: "abajo", 1: "neutra", 2: "arriba"}
MOUTH_OPEN_LABELS = {0: "cerrada", 1: "abierta"}


def softmax_raw(logits):
    """Softmax a temperatura natural (T=1.0) para evaluar la certeza real del modelo."""
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

        print(f"[INFO] Cargando modelos ONNX de la versión oráculo: v{VERSION}...")
        self.net_periocular = cv2.dnn.readNetFromONNX(PERIOCULAR_MODEL_PATH)
        self.net_mouth = cv2.dnn.readNetFromONNX(BOCA_MODEL_PATH)

        self.net_periocular.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net_periocular.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.net_mouth.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net_mouth.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.out_names_p = self.net_periocular.getUnconnectedOutLayersNames()
        self.out_names_m = self.net_mouth.getUnconnectedOutLayersNames()

        self.extractor = GeometricPatchExtractor(target_size=(64, 64))

        # Crear estructura base de Redataset
        os.makedirs(os.path.join(DIR_REDATASET, "periocular_izq"), exist_ok=True)
        os.makedirs(os.path.join(DIR_REDATASET, "periocular_der"), exist_ok=True)
        os.makedirs(os.path.join(DIR_REDATASET, "boca"), exist_ok=True)

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
            prob_ceja[idx_ceja],
            EYE_LABELS.get(idx_ojo, "abierto"),
            prob_ojo[idx_ojo]
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
            prob_com[idx_com],
            MOUTH_OPEN_LABELS.get(idx_ap, "cerrada"),
            prob_ap[idx_ap]
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

    def process_all(self):
        if not os.path.exists(DIR_MUESTRAS):
            print(f"[ERROR] No se encontró el directorio de muestras en: '{DIR_MUESTRAS}'")
            return

        voluntarios = sorted([
            v for v in os.listdir(DIR_MUESTRAS)
            if os.path.isdir(os.path.join(DIR_MUESTRAS, v))
        ])

        if not voluntarios:
            print(f"[AVISO] No se encontraron subcarpetas en: '{DIR_MUESTRAS}'")
            return

        window_name = f"Retrainer Oráculo - v{VERSION}"
        if ENABLE_PREVIEW:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        total_analizadas = 0
        total_p_izq_guardados = 0
        total_p_der_guardados = 0
        total_boca_guardados = 0
        total_descartes_pose = 0

        fps = 0.0
        prev_time = time.time()

        print(f"\n[INFO] Iniciando pseudo-etiquetado con umbral de certeza >= {int(CONFIDENCE_THRESHOLD * 100)}%...")

        for vol in voluntarios:
            vol_dir = os.path.join(DIR_MUESTRAS, vol)
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

                saved_izq = False
                saved_der = False
                saved_boca = False

                if patches is None or status != "OK":
                    total_descartes_pose += 1
                else:
                    preds_ready = True

                    # 1. Análisis Periocular Izquierdo
                    ceja_izq, conf_c_izq, ojo_izq, conf_o_izq = self._infer_periocular(patches["periocular_izq"])
                    if conf_c_izq >= CONFIDENCE_THRESHOLD and conf_o_izq >= CONFIDENCE_THRESHOLD:
                        subfolder_izq = f"ceja_{ceja_izq}_ojo_{ojo_izq}"
                        dir_out_izq = os.path.join(DIR_REDATASET, "periocular_izq", subfolder_izq)
                        os.makedirs(dir_out_izq, exist_ok=True)
                        cv2.imwrite(
                            os.path.join(dir_out_izq, f"{vol}_{idx}_periocular_izq.png"),
                            patches["periocular_izq"]
                        )
                        total_p_izq_guardados += 1
                        saved_izq = True

                    # 2. Análisis Periocular Derecho
                    ceja_der, conf_c_der, ojo_der, conf_o_der = self._infer_periocular(patches["periocular_der"])
                    if conf_c_der >= CONFIDENCE_THRESHOLD and conf_o_der >= CONFIDENCE_THRESHOLD:
                        subfolder_der = f"ceja_{ceja_der}_ojo_{ojo_der}"
                        dir_out_der = os.path.join(DIR_REDATASET, "periocular_der", subfolder_der)
                        os.makedirs(dir_out_der, exist_ok=True)
                        cv2.imwrite(
                            os.path.join(dir_out_der, f"{vol}_{idx}_periocular_der.png"),
                            patches["periocular_der"]
                        )
                        total_p_der_guardados += 1
                        saved_der = True

                    # 3. Análisis Boca
                    com_boca, conf_com, ap_boca, conf_ap = self._infer_mouth(patches["boca"])
                    if conf_com >= CONFIDENCE_THRESHOLD and conf_ap >= CONFIDENCE_THRESHOLD:
                        subfolder_boca = f"comisuras_{com_boca}_{ap_boca}"
                        dir_out_boca = os.path.join(DIR_REDATASET, "boca", subfolder_boca)
                        os.makedirs(dir_out_boca, exist_ok=True)
                        cv2.imwrite(
                            os.path.join(dir_out_boca, f"{vol}_{idx}_boca.png"),
                            patches["boca"]
                        )
                        total_boca_guardados += 1
                        saved_boca = True

                # --------------------------------------------------------------
                # RENDERIZADO VISUAL SI PREVIEW ESTÁ ACTIVO
                # --------------------------------------------------------------
                if ENABLE_PREVIEW:
                    display_frame = frame.copy()

                    # 1. Superior Izquierda: Estatus y archivo activo
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

                    self._draw_hud_box(display_frame, (10, 10), (320, 65), alpha=0.6)
                    cv2.putText(display_frame, display_status, (18, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1, cv2.LINE_AA)
                    cv2.putText(display_frame, f"{vol} | {fname[:20]}", (18, 54),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

                    # 2. Superior Derecha: FPS y Aceptación
                    self._draw_hud_box(display_frame, (w_frame - 180, 10), (w_frame - 10, 65), alpha=0.6)
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (w_frame - 170, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 1, cv2.LINE_AA)
                    cv2.putText(display_frame, f"Umbral: {int(CONFIDENCE_THRESHOLD*100)}%", (w_frame - 170, 53),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

                    # 3. Inferior Izquierda: Telemetría anatómica
                    self._draw_hud_box(display_frame, (10, h_frame - 145), (280, h_frame - 10), alpha=0.6)
                    y_base = h_frame - 128
                    line_step = 21
                    labels_info = [
                        f"Ojo Izq: {ojo_izq} ({conf_o_izq * 100:.0f}%) {'[+]' if saved_izq else '[-]'}",
                        f"Ojo Der: {ojo_der} ({conf_o_der * 100:.0f}%) {'[+]' if saved_der else '[-]'}",
                        f"Ceja Izq: {ceja_izq} ({conf_c_izq * 100:.0f}%)",
                        f"Ceja Der: {ceja_der} ({conf_c_der * 100:.0f}%)",
                        f"Apertura Boca: {ap_boca} ({conf_ap * 100:.0f}%) {'[+]' if saved_boca else '[-]'}",
                        f"Comisuras: {com_boca} ({conf_com * 100:.0f}%)"
                    ]

                    for i, text in enumerate(labels_info):
                        col = (0, 255, 0) if "[+]" in text else (255, 255, 255)
                        cv2.putText(display_frame, text, (18, y_base + (i * line_step)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

                    # 4. Inferior Derecha: Miniaturas 64x64
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

                    # Redimensionamiento seguro manteniendo el aspect ratio
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
                        if ENABLE_PREVIEW:
                            cv2.destroyAllWindows()
                        return

        if ENABLE_PREVIEW:
            cv2.destroyAllWindows()

        print("\n==================================================")
        print("RESUMEN DE PSEUDO-ETIQUETADO (RETRAINER)")
        print(f"  - Total fotos analizadas   : {total_analizadas}")
        print(f"  - Descartes por pose/rostro: {total_descartes_pose}")
        print(f"  - Parches P. Izquierdo     : {total_p_izq_guardados}")
        print(f"  - Parches P. Derecho       : {total_p_der_guardados}")
        print(f"  - Parches Boca             : {total_boca_guardados}")
        print(f"  - Total parches rescatados : {total_p_izq_guardados + total_p_der_guardados + total_boca_guardados}")
        print(f"  - Directorio de salida     : {DIR_REDATASET}")
        print("==================================================\n")


if __name__ == "__main__":
    retrainer = RetrainerPipeline()
    retrainer.process_all()