# realtime.py
import os
import time
import queue
import threading
from datetime import datetime
import cv2
import numpy as np
from patch import GeometricPatchExtractor

# ==============================================================================
# CONFIGURACIÓN DE VERSIÓN Y RUTAS
# ==============================================================================
VERSION = 10

ROOT_DATA = "Data"
DIR_VERSION_ONNX = os.path.join(ROOT_DATA, "versiones", f"v{VERSION}", "onnx")
PERIOCULAR_MODEL_PATH = os.path.join(DIR_VERSION_ONNX, "model_periocular.onnx")
BOCA_MODEL_PATH = os.path.join(DIR_VERSION_ONNX, "model_boca.onnx")

DIR_TRANSMISIONES = os.path.join(ROOT_DATA, "transmisiones")
DIR_CAPTURAS = os.path.join(DIR_TRANSMISIONES, "capturas")
DIR_RAFAGAS = os.path.join(DIR_TRANSMISIONES, "rafagas")

DIR_RDATASET = os.path.join(ROOT_DATA, "Rdataset")

# Intervalo para ráfaga (5 capturas por segundo -> 0.2s)
BURST_INTERVAL_SECONDS = 0.2

# Calibración de temperatura para softmax
SOFTMAX_TEMPERATURE = 3.5

# Mapeos de clases
EYEBROW_LABELS = {0: "Descendida", 1: "Neutra", 2: "Elevada"}
EYE_LABELS = {0: "Cerrado", 1: "Abierto"}
COMMISSURE_LABELS = {0: "Abajo", 1: "Neutra", 2: "Arriba"}
MOUTH_OPEN_LABELS = {0: "Cerrada", 1: "Abierta"}


def softmax(logits, temperature=SOFTMAX_TEMPERATURE):
    scaled_logits = logits / temperature
    exp_z = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_z / np.sum(exp_z)


class DiskWriterWorker:
    """Escritor asíncrono en segundo plano para evitar bloqueos del hilo de video."""
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running or not self.queue.empty():
            try:
                task = self.queue.get(timeout=0.1)
                filepath, img_data = task
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                cv2.imwrite(filepath, img_data)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR ASÍNCRONO] Error guardando archivo: {e}")

    def add_task(self, filepath, img_data):
        self.queue.put((filepath, img_data))

    def stop(self):
        self.running = False
        self.queue.join()


class RealtimeInferenceEngine:
    def __init__(self, periocular_model_path, mouth_model_path):
        if not os.path.exists(periocular_model_path) or not os.path.exists(mouth_model_path):
            raise FileNotFoundError(
                f"[ERROR] No se encontraron los modelos de la versión v{VERSION} en:\n"
                f"  - {periocular_model_path}\n"
                f"  - {mouth_model_path}\n"
                "Verifica que la carpeta exista o entrena primero con trainer.py."
            )

        print(f"[INFO] Cargando modelos ONNX para versión: v{VERSION}...")
        self.net_periocular = cv2.dnn.readNetFromONNX(periocular_model_path)
        self.net_mouth = cv2.dnn.readNetFromONNX(mouth_model_path)

        self.net_periocular.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net_periocular.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.net_mouth.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net_mouth.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.out_names_p = self.net_periocular.getUnconnectedOutLayersNames()
        self.out_names_m = self.net_mouth.getUnconnectedOutLayersNames()

        self.extractor = GeometricPatchExtractor(target_size=(64, 64))
        self.writer = DiskWriterWorker()

        os.makedirs(DIR_CAPTURAS, exist_ok=True)
        os.makedirs(DIR_RAFAGAS, exist_ok=True)
        os.makedirs(DIR_RDATASET, exist_ok=True)

        self.current_minute_str = ""
        self.minute_index = 0

        self.single_capture_count = 0
        self.burst_capture_count = 0
        self.last_burst_time = 0.0

        # Tags activos para grabación manual supervisada
        self.tags = {
            "ojo": None,        # 'abierto', 'cerrado'
            "ceja": None,       # 'neutra', 'arriba', 'abajo'
            "boca": None,       # 'abierta', 'cerrada'
            "comisuras": None   # 'neutra', 'arriba', 'abajo'
        }

    def _infer_periocular(self, patch):
        blob = (patch.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]
        self.net_periocular.setInput(blob)
        outputs = self.net_periocular.forward(self.out_names_p)

        out_ceja = outputs[0] if outputs[0].shape[-1] == 3 else outputs[1]
        out_ojo = outputs[1] if outputs[0].shape[-1] == 3 else outputs[0]

        prob_ceja = softmax(out_ceja[0])
        prob_ojo = softmax(out_ojo[0])

        idx_ceja = int(np.argmax(prob_ceja))
        idx_ojo = int(np.argmax(prob_ojo))

        return (
            EYEBROW_LABELS.get(idx_ceja, "N/A"),
            prob_ceja[idx_ceja],
            EYE_LABELS.get(idx_ojo, "N/A"),
            prob_ojo[idx_ojo]
        )

    def _infer_mouth(self, patch):
        blob = (patch.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]
        self.net_mouth.setInput(blob)
        outputs = self.net_mouth.forward(self.out_names_m)

        out_com = outputs[0] if outputs[0].shape[-1] == 3 else outputs[1]
        out_ap = outputs[1] if outputs[0].shape[-1] == 3 else outputs[0]

        prob_com = softmax(out_com[0])
        prob_ap = softmax(out_ap[0])

        idx_com = int(np.argmax(prob_com))
        idx_ap = int(np.argmax(prob_ap))

        return (
            COMMISSURE_LABELS.get(idx_com, "N/A"),
            prob_com[idx_com],
            MOUTH_OPEN_LABELS.get(idx_ap, "N/A"),
            prob_ap[idx_ap]
        )

    def _get_timestamp_and_index(self):
        now = datetime.now()
        minute_key = now.strftime("%d_%m_%y_%H_%M")
        if minute_key == self.current_minute_str:
            self.minute_index += 1
        else:
            self.current_minute_str = minute_key
            self.minute_index = 1
        return minute_key, self.minute_index

    def _has_complete_tags(self):
        return all(v is not None for v in self.tags.values())

    def _save_data(self, raw_frame, patches, is_burst=False):
        now = datetime.now()

        # CASO 1: Grabación manual hacia Rdataset (solo con parches extraídos)
        if self._has_complete_tags() and patches is not None:
            minute_key, idx = self._get_timestamp_and_index()

            ceja_folder_map = {"arriba": "elevada", "abajo": "descendida", "neutra": "neutra"}
            comisuras_folder_map = {"arriba": "arriba", "abajo": "abajo", "neutra": "neutra"}

            ceja_estado = ceja_folder_map[self.tags["ceja"]]
            ojo_estado = self.tags["ojo"]
            subfolder_periocular = f"ceja_{ceja_estado}_ojo_{ojo_estado}"

            comisuras_estado = comisuras_folder_map[self.tags["comisuras"]]
            boca_estado = self.tags["boca"]
            subfolder_boca = f"comisuras_{comisuras_estado}_{boca_estado}"

            # 1. Periocular Izquierdo
            path_izq = os.path.join(
                DIR_RDATASET, "periocular_izq", subfolder_periocular,
                f"{minute_key}_{idx}_periocular_izq.png"
            )
            self.writer.add_task(path_izq, patches["periocular_izq"])

            # 2. Periocular Derecho
            path_der = os.path.join(
                DIR_RDATASET, "periocular_der", subfolder_periocular,
                f"{minute_key}_{idx}_periocular_der.png"
            )
            self.writer.add_task(path_der, patches["periocular_der"])

            # 3. Boca
            path_boca = os.path.join(
                DIR_RDATASET, "boca", subfolder_boca,
                f"{minute_key}_{idx}_boca.png"
            )
            self.writer.add_task(path_boca, patches["boca"])

        # CASO 2: Grabación cruda a transmisiones (sin tags completos)
        else:
            timestamp = now.strftime("%d_%m_%Y_%H_%M")
            if is_burst:
                self.burst_capture_count += 1
                filename = f"{timestamp}_{self.burst_capture_count}.jpg"
                filepath = os.path.join(DIR_RAFAGAS, filename)
            else:
                self.single_capture_count += 1
                filename = f"{timestamp}_{self.single_capture_count}.jpg"
                filepath = os.path.join(DIR_CAPTURAS, filename)

            self.writer.add_task(filepath, raw_frame)

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

    def run(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[ERROR] No se pudo abrir la cámara {camera_index}.")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        window_name = f"Inferencia en Vivo - v{VERSION}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n==================================================")
        print(f"SISTEMA EN VIVO ACTIVO - VERSIÓN: v{VERSION}")
        print("  - Tecla 'C': Captura única (asíncrona)")
        print("  - Barra ESPACIO: Ráfaga continua (5 fps)")
        print("  - Tags Ojos      : [1] Abierto  | [2] Cerrado")
        print("  - Tags Cejas     : [3] Neutra   | [4] Arriba   | [5] Abajo")
        print("  - Tags Boca      : [6] Abierta  | [7] Cerrada")
        print("  - Tags Comisuras : [8] Neutra   | [9] Arriba   | [0] Abajo")
        print("  - Tecla BACKSPACE: Limpiar tags")
        print("  - Tecla 'Q' o ESC: Salir")
        print("==================================================\n")

        fps = 0.0
        prev_time = time.time()
        single_flash_until = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
            prev_time = curr_time

            raw_frame_copy = frame.copy()
            h_frame, w_frame, _ = frame.shape

            patches, status = self.extractor.extract_patches(frame)

            preds_ready = False
            avg_conf = 0.0
            ceja_izq = ceja_der = ojo_izq = ojo_der = com_boca = ap_boca = "N/A"
            conf_c_izq = conf_c_der = conf_o_izq = conf_o_der = conf_com = conf_ap = 0.0

            if patches is not None:
                preds_ready = True
                ceja_izq, conf_c_izq, ojo_izq, conf_o_izq = self._infer_periocular(patches["periocular_izq"])
                ceja_der, conf_c_der, ojo_der, conf_o_der = self._infer_periocular(patches["periocular_der"])
                com_boca, conf_com, ap_boca, conf_ap = self._infer_mouth(patches["boca"])

                avg_conf = (conf_c_izq + conf_o_izq + conf_c_der + conf_o_der + conf_com + conf_ap) / 6.0

            # ------------------------------------------------------------------
            # HUD: 4 ESQUINAS CON FONDOS SEMITRANSPARENTES
            # ------------------------------------------------------------------

            # 1. Superior Izquierda: Estado limpio + Cadena de tags
            if status == "OK":
                display_status = "Reconociendo"
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

            tag_str = (
                f"{self.tags['ojo'] or '-'} "
                f"{self.tags['ceja'] or '-'} "
                f"{self.tags['boca'] or '-'} "
                f"{self.tags['comisuras'] or '-'}"
            )

            self._draw_hud_box(frame, (10, 10), (330, 70), alpha=0.6)
            cv2.putText(frame, display_status, (18, 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1, cv2.LINE_AA)

            tag_color = (0, 255, 255) if self._has_complete_tags() else (180, 180, 180)
            cv2.putText(frame, tag_str, (18, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, tag_color, 1, cv2.LINE_AA)

            # 2. Superior Derecha: FPS y Confianza
            self._draw_hud_box(frame, (w_frame - 180, 10), (w_frame - 10, 65), alpha=0.6)
            fps_text = f"FPS: {fps:.1f}"
            conf_text = f"Confianza: {avg_conf * 100:.1f}%" if preds_ready else "Confianza: --"

            cv2.putText(frame, fps_text, (w_frame - 170, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, conf_text, (w_frame - 170, 53),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

            # 3. Inferior Izquierda: Telemetría anatómica
            self._draw_hud_box(frame, (10, h_frame - 145), (260, h_frame - 10), alpha=0.6)
            y_base = h_frame - 128
            line_step = 21
            labels_info = [
                f"Ojo Izq: {ojo_izq} ({conf_o_izq * 100:.0f}%)",
                f"Ojo Der: {ojo_der} ({conf_o_der * 100:.0f}%)",
                f"Ceja Izq: {ceja_izq} ({conf_c_izq * 100:.0f}%)",
                f"Ceja Der: {ceja_der} ({conf_c_der * 100:.0f}%)",
                f"Apertura Boca: {ap_boca} ({conf_ap * 100:.0f}%)",
                f"Comisuras: {com_boca} ({conf_com * 100:.0f}%)"
            ]

            for i, text in enumerate(labels_info):
                cv2.putText(frame, text, (18, y_base + (i * line_step)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

            # 4. Inferior Derecha: Miniaturas 64x64
            if patches is not None:
                self._draw_hud_box(frame, (w_frame - 232, h_frame - 92), (w_frame - 5, h_frame - 8), alpha=0.6)
                p_izq_bgr = cv2.cvtColor(patches["periocular_izq"], cv2.COLOR_GRAY2BGR)
                p_der_bgr = cv2.cvtColor(patches["periocular_der"], cv2.COLOR_GRAY2BGR)
                p_boca_bgr = cv2.cvtColor(patches["boca"], cv2.COLOR_GRAY2BGR)

                margin_y = h_frame - 76
                frame[margin_y:margin_y + 64, w_frame - 74:w_frame - 10] = p_boca_bgr
                frame[margin_y:margin_y + 64, w_frame - 148:w_frame - 84] = p_der_bgr
                frame[margin_y:margin_y + 64, w_frame - 222:w_frame - 158] = p_izq_bgr

                cv2.putText(frame, "Izq", (w_frame - 205, margin_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(frame, "Der", (w_frame - 130, margin_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(frame, "Boca", (w_frame - 60, margin_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)

            # Flash de confirmación al guardar
            if curr_time < single_flash_until:
                dest_str = "RDATASET" if self._has_complete_tags() else "TRANSMISIONES"
                cv2.putText(frame, f"[GUARDADO ASÍNCRONO -> {dest_str}]", (w_frame // 2 - 160, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

            # Redimensionamiento con letterboxing manteniendo el aspect ratio original
            try:
                win_rect = cv2.getWindowImageRect(window_name)
                win_w, win_h = win_rect[2], win_rect[3]
            except Exception:
                win_w, win_h = w_frame, h_frame

            if win_w > 100 and win_h > 100 and (win_w != w_frame or win_h != h_frame):
                display_frame = self._apply_letterboxing(frame, win_w, win_h)
            else:
                display_frame = frame

            cv2.imshow(window_name, display_frame)

            key_raw = cv2.waitKey(1)
            key = key_raw & 0xFF

            # Salida
            if key == ord('q') or key == 27:
                break

            # Limpiar selección de tags: Backspace exclusivo (ASCII 8)
            elif key == 8:
                self.tags = {"ojo": None, "ceja": None, "boca": None, "comisuras": None}

            # Ojos: 1 abierto, 2 cerrado
            elif key == ord('1'):
                self.tags["ojo"] = "abierto"
            elif key == ord('2'):
                self.tags["ojo"] = "cerrado"

            # Cejas: 3 neutra, 4 arriba, 5 abajo
            elif key == ord('3'):
                self.tags["ceja"] = "neutra"
            elif key == ord('4'):
                self.tags["ceja"] = "arriba"
            elif key == ord('5'):
                self.tags["ceja"] = "abajo"

            # Boca: 6 abierta, 7 cerrada
            elif key == ord('6'):
                self.tags["boca"] = "abierta"
            elif key == ord('7'):
                self.tags["boca"] = "cerrada"

            # Comisuras: 8 neutra, 9 arriba, 0 abajo
            elif key == ord('8'):
                self.tags["comisuras"] = "neutra"
            elif key == ord('9'):
                self.tags["comisuras"] = "arriba"
            elif key == ord('0'):
                self.tags["comisuras"] = "abajo"

            # Captura única
            elif key == ord('c') or key == ord('C'):
                self._save_data(raw_frame_copy, patches, is_burst=False)
                single_flash_until = curr_time + 0.4

            # Ráfaga continua (Barra ESPACIO)
            elif key == 32:
                if (curr_time - self.last_burst_time) >= BURST_INTERVAL_SECONDS:
                    self._save_data(raw_frame_copy, patches, is_burst=True)
                    self.last_burst_time = curr_time
                    single_flash_until = curr_time + 0.2

        cap.release()
        cv2.destroyAllWindows()
        self.writer.stop()


if __name__ == "__main__":
    engine = RealtimeInferenceEngine(PERIOCULAR_MODEL_PATH, BOCA_MODEL_PATH)
    engine.run(camera_index=0)