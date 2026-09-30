# balancer.py
import os
import re
import math
import random
import shutil
import json
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, PngImagePlugin

# ==============================================================================
# CONFIGURACIÓN GENERAL
# ==============================================================================
ROOT_DATA = "Data"
DIR_READY = os.path.join(ROOT_DATA, "dataset", "ready")

# Clases fijas por región (las 6 combinaciones esperadas)
CLASSES_PERIOCULAR = [
    "ceja_descendida_ojo_abierto",
    "ceja_descendida_ojo_cerrado",
    "ceja_elevada_ojo_abierto",
    "ceja_elevada_ojo_cerrado",
    "ceja_neutra_ojo_abierto",
    "ceja_neutra_ojo_cerrado"
]

CLASSES_BOCA = [
    "comisuras_abajo_abierta",
    "comisuras_abajo_cerrada",
    "comisuras_arriba_abierta",
    "comisuras_arriba_cerrada",
    "comisuras_neutra_abierta",
    "comisuras_neutra_cerrada"
]

REGIONES = ["periocular_izq", "periocular_der", "boca"]

# Rango de operaciones sintéticas
MAX_ROTATION_DEG = 3.0
MAX_TRANSLATION_PX = 2
ALPHA_CONTRAST_RANGE = (0.90, 1.10)
BETA_BRIGHTNESS_RANGE = (-10, 10)
PROB_GAUSSIAN_BLUR = 0.20


def natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


# ==============================================================================
# MÓDULO SINTETIZADOR CON REGISTRO DE METADATOS PNG
# ==============================================================================
class PatchSynthesizer:
    @staticmethod
    def apply_transform(img_gray: np.ndarray):
        h, w = img_gray.shape
        params_record = []

        # 1. Rotación suave
        angle = random.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
        params_record.append(f"rot:{angle:.2f}")
        m_rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        img_trans = cv2.warpAffine(img_gray, m_rot, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # 2. Micro-traslación
        tx = random.randint(-MAX_TRANSLATION_PX, MAX_TRANSLATION_PX)
        ty = random.randint(-MAX_TRANSLATION_PX, MAX_TRANSLATION_PX)
        params_record.append(f"tx:{tx},ty:{ty}")
        m_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        img_trans = cv2.warpAffine(img_trans, m_trans, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # 3. Variación fotométrica (brillo / contraste)
        alpha = random.uniform(*ALPHA_CONTRAST_RANGE)
        beta = random.uniform(*BETA_BRIGHTNESS_RANGE)
        params_record.append(f"alpha:{alpha:.2f},beta:{beta:.1f}")
        img_trans = np.clip(alpha * img_trans + beta, 0, 255).astype(np.uint8)

        # 4. Desenfoque gaussiano suave ocasional
        if random.random() < PROB_GAUSSIAN_BLUR:
            img_trans = cv2.GaussianBlur(img_trans, (3, 3), 0.5)
            params_record.append("blur:true")
        else:
            params_record.append("blur:false")

        metadata_str = ";".join(params_record)
        return img_trans, metadata_str

    @staticmethod
    def save_with_metadata(img_np: np.ndarray, dest_path: str, src_name: str, op_type: str, details: str):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        pil_img = Image.fromarray(img_np)
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("source_file", src_name)
        png_info.add_text("operation", op_type)
        png_info.add_text("details", details)
        pil_img.save(dest_path, pnginfo=png_info)


# ==============================================================================
# MOTOR PRINCIPAL DE BALANCEO MULTI-DATASET
# ==============================================================================
class DatasetBalancer:
    def __init__(self):
        self.datasets = self._discover_valid_datasets()

    def _discover_valid_datasets(self):
        if not os.path.exists(DIR_READY):
            return []
        
        valid = []
        for entry in sorted(os.listdir(DIR_READY), key=natural_sort_key):
            full_path = os.path.join(DIR_READY, entry)
            # Ignoramos carpetas ya balanceadas o temporales
            if os.path.isdir(full_path) and not entry.endswith("_bal") and entry not in {"invalidas", "incertidumbre"}:
                # Verificamos que tenga estructura mínima
                if any(os.path.exists(os.path.join(full_path, r)) for r in REGIONES):
                    valid.append(entry)
        return valid

    def _scan_dataset_files(self):
        """
        Estructura de retorno:
        data[dataset][region][class_name] = [lista de rutas absolutas]
        """
        tree = {}
        for ds in self.datasets:
            tree[ds] = {r: {} for r in REGIONES}
            ds_dir = os.path.join(DIR_READY, ds)
            for r in REGIONES:
                r_dir = os.path.join(ds_dir, r)
                target_classes = CLASSES_BOCA if r == "boca" else CLASSES_PERIOCULAR
                for c in target_classes:
                    c_dir = os.path.join(r_dir, c)
                    if os.path.exists(c_dir):
                        files = sorted([
                            os.path.join(c_dir, f) for f in os.listdir(c_dir)
                            if f.lower().endswith(".png")
                        ], key=natural_sort_key)
                        tree[ds][r][c] = files
                    else:
                        tree[ds][r][c] = []
        return tree

    def execute_balance(self):
        if not self.datasets:
            print(f"[AVISO] No se encontraron datasets procesables en: {DIR_READY}")
            return

        print("\n" + "=" * 60)
        print("INICIANDO BALANCEO MULTI-DATASET CON FUSIÓN Y SÍNTESIS")
        print(f"Datasets detectados: {', '.join(self.datasets)}")
        print("=" * 60)

        data = self._scan_dataset_files()

        # Preparar carpetas destino vacías
        for ds in self.datasets:
            out_dir = os.path.join(DIR_READY, f"{ds}_bal")
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)

        # Diccionario para estadísticas de metadata final
        meta_summary = {
            ds: {
                r: {c: {"originales": 0, "invertidas": 0, "sintetizadas": 0, "total": 0}
                    for c in (CLASSES_BOCA if r == "boca" else CLASSES_PERIOCULAR)}
                for r in REGIONES
            } for ds in self.datasets
        }

        # ----------------------------------------------------------------------
        # PROCESAMIENTO POR REGIÓN
        # ----------------------------------------------------------------------
        for r in REGIONES:
            target_classes = CLASSES_BOCA if r == "boca" else CLASSES_PERIOCULAR
            print(f"\n--- Balanceando Región: {r} ---")

            # 1. Total global por clase
            global_counts = {c: sum(len(data[ds][r][c]) for ds in self.datasets) for c in target_classes}
            total_region_samples = sum(global_counts.values())

            if total_region_samples == 0:
                print(f"[AVISO] Región {r} vacía en todos los datasets. Se omite.")
                continue

            # N unificado perfecto por clase
            n_target = total_region_samples // len(target_classes)
            print(f"Total global en {r}: {total_region_samples} imágenes. Target exacto por clase: {n_target}")

            for c in target_classes:
                current_total = global_counts[c]
                b_deficit = n_target - current_total  # >0: falta sintetizar/invertir | <0: sobra (podar)

                # ==============================================================
                # FASE A: Inversión en espejo para Perioculares si hay déficit
                # ==============================================================
                inverted_assigned = {ds: [] for ds in self.datasets}
                if b_deficit > 0 and r in {"periocular_izq", "periocular_der"}:
                    opposite_r = "periocular_der" if r == "periocular_izq" else "periocular_izq"
                    # Recolectar parches disponibles del ojo contrario
                    opp_pools = {ds: list(data[ds][opposite_r][c]) for ds in self.datasets}
                    total_opp_avail = sum(len(v) for v in opp_pools.values())

                    to_take = min(b_deficit, total_opp_avail)
                    taken = 0

                    # Extracción Round-Robin balanceada entre datasets
                    while taken < to_take:
                        progress = False
                        for ds in self.datasets:
                            if opp_pools[ds] and taken < to_take:
                                patch_path = opp_pools[ds].pop(0)
                                inverted_assigned[ds].append(patch_path)
                                taken += 1
                                progress = True
                        if not progress:
                            break

                    b_deficit -= taken
                    if taken > 0:
                        print(f"  [{c}] Se incorporaron {taken} muestras invertidas desde {opposite_r}.")

                # ==============================================================
                # FASE B: Reparto proporcional de la carga restante (b_deficit)
                # ==============================================================
                # Calculamos el peso relativo de cada dataset en esta región
                ds_weights = {}
                for ds in self.datasets:
                    ds_region_total = sum(len(data[ds][r][k]) for k in target_classes)
                    ds_weights[ds] = (ds_region_total / total_region_samples) if total_region_samples > 0 else (1.0 / len(self.datasets))

                # Asignación proporcional de b_deficit
                b_per_ds = {}
                accum = 0
                ds_list = list(self.datasets)
                for i, ds in enumerate(ds_list):
                    if i == len(ds_list) - 1:
                        b_per_ds[ds] = b_deficit - accum
                    else:
                        alloc = int(round(b_deficit * ds_weights[ds]))
                        b_per_ds[ds] = alloc
                        accum += alloc

                # ==============================================================
                # FASE C: Ejecución por Dataset (Copia, Espejo, Poda o Síntesis)
                # ==============================================================
                for ds in self.datasets:
                    out_c_dir = os.path.join(DIR_READY, f"{ds}_bal", r, c)
                    os.makedirs(out_c_dir, exist_ok=True)

                    orig_files = list(data[ds][r][c])
                    num_orig = len(orig_files)
                    inv_files = inverted_assigned[ds]
                    num_inv = len(inv_files)

                    # Guardar parches invertidos asignados
                    for inv_src in inv_files:
                        img = cv2.imread(inv_src, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            flipped = cv2.flip(img, 1)
                            base_name = os.path.splitext(os.path.basename(inv_src))[0]
                            dest_name = f"{base_name}_inv.png"
                            dest_path = os.path.join(out_c_dir, dest_name)
                            PatchSynthesizer.save_with_metadata(
                                flipped, dest_path, os.path.basename(inv_src), "horizontal_flip", "source:opposite_eye"
                            )
                            meta_summary[ds][r][c]["invertidas"] += 1

                    local_b = b_per_ds[ds]

                    if local_b < 0:
                        # ------------------------------------------------------
                        # PODA UNIFORME DETERMINISTA
                        # ------------------------------------------------------
                        needed = max(0, num_orig + local_b)
                        if needed == 0:
                            kept_files = []
                        elif needed >= num_orig:
                            kept_files = orig_files
                        else:
                            step = num_orig / float(needed)
                            indices = [int(k * step) for k in range(needed)]
                            kept_files = [orig_files[idx] for idx in indices]

                        for fpath in kept_files:
                            fname = os.path.basename(fpath)
                            shutil.copy2(fpath, os.path.join(out_c_dir, fname))
                            meta_summary[ds][r][c]["originales"] += 1

                    else:
                        # ------------------------------------------------------
                        # COPIA DE TODOS LOS ORIGINALES + SÍNTESIS
                        # ------------------------------------------------------
                        for fpath in orig_files:
                            fname = os.path.basename(fpath)
                            shutil.copy2(fpath, os.path.join(out_c_dir, fname))
                            meta_summary[ds][r][c]["originales"] += 1

                        if local_b > 0 and num_orig > 0:
                            # Generación sintética estocástica
                            for syn_idx in range(1, local_b + 1):
                                chosen_src = random.choice(orig_files)
                                img = cv2.imread(chosen_src, cv2.IMREAD_GRAYSCALE)
                                if img is not None:
                                    syn_img, meta_desc = PatchSynthesizer.apply_transform(img)
                                    base_name = os.path.splitext(os.path.basename(chosen_src))[0]
                                    dest_name = f"{base_name}_syn_{syn_idx}.png"
                                    dest_path = os.path.join(out_c_dir, dest_name)
                                    PatchSynthesizer.save_with_metadata(
                                        syn_img, dest_path, os.path.basename(chosen_src), "stochastic_augmentation", meta_desc
                                    )
                                    meta_summary[ds][r][c]["sintetizadas"] += 1

                    total_in_class = (
                        meta_summary[ds][r][c]["originales"] +
                        meta_summary[ds][r][c]["invertidas"] +
                        meta_summary[ds][r][c]["sintetizadas"]
                    )
                    meta_summary[ds][r][c]["total"] = total_in_class

        # ----------------------------------------------------------------------
        # GENERACIÓN DE METADATA POR DATASET
        # ----------------------------------------------------------------------
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M")
        for ds in self.datasets:
            ds_out_dir = os.path.join(DIR_READY, f"{ds}_bal")
            total_ds_samples = sum(
                meta_summary[ds][r][c]["total"]
                for r in REGIONES
                for c in (CLASSES_BOCA if r == "boca" else CLASSES_PERIOCULAR)
            )

            manifest = {
                "origen": "balancer",
                "dataset_base": ds,
                "timestamp": timestamp,
                "parametros_sintesis": {
                    "max_rotation_deg": MAX_ROTATION_DEG,
                    "max_translation_px": MAX_TRANSLATION_PX,
                    "alpha_contrast_range": ALPHA_CONTRAST_RANGE,
                    "beta_brightness_range": BETA_BRIGHTNESS_RANGE,
                    "prob_gaussian_blur": PROB_GAUSSIAN_BLUR
                },
                "total_muestras_balanceadas": total_ds_samples,
                "desglose_clases": meta_summary[ds]
            }

            manifest_path = os.path.join(ds_out_dir, "metadata.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=4, ensure_ascii=False)
            print(f"[INFO] Manifiesto guardado en: {manifest_path}")

        print("\n" + "=" * 60)
        print("BALANCEO COMPLETADO CON ÉXITO")
        print(f"Salidas generadas en: {[f'{ds}_bal' for ds in self.datasets]}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    balancer = DatasetBalancer()
    balancer.execute_balance()