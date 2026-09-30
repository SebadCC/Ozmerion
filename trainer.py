# trainer.py
import os
import re
import json
import stat
import time
import shutil
import random
from datetime import datetime
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# CONFIGURACIÓN GENERAL DEL ENTRENAMIENTO
# ==============================================================================
BATCH_SIZE = 8
EPOCHS = 8
LR = 0.001
WEIGHT_DECAY = 0.01

VAL_SPLIT = 15                 # Porcentaje entero de validación (0 = sin validación, 15 = 15%)
AUTO_BALANCE_DATASET = True    # Nivelar clases al mínimo común antes de particionar
ADAPTIVE_STAGES = 3            # Etapas de entrenamiento decrecientes (0 o 1 para pasada única)

# ==============================================================================
# RUTAS DEL SISTEMA DE ARCHIVOS
# ==============================================================================
ROOT_DATA = "Data"
DIR_DATASET_READY = os.path.join(ROOT_DATA, "dataset", "ready")
DIR_MODELS = os.path.join(ROOT_DATA, "models")


def remove_readonly(func, path, exc_info):
    """Manejador de excepciones para Windows: remueve flag de solo lectura y reintenta."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


# ==============================================================================
# ARQUITECTURAS TINY-CNN CON DROPOUT
# ==============================================================================
class TinyCNNPeriocular(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16

            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 8x8
        )
        self.head_ceja = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(48 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 0: Descendida, 1: Neutra, 2: Elevada
        )
        self.head_ojo = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(48 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 0: Cerrado, 1: Abierto
        )

    def forward(self, x):
        feat = self.features(x)
        return self.head_ceja(feat), self.head_ojo(feat)


class TinyCNNBoca(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2),  # 32x32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16

            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 8x8
        )
        self.head_comisuras = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(48 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 0: Abajo, 1: Neutra, 2: Arriba
        )
        self.head_apertura = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(48 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 0: Cerrada, 1: Abierta
        )

    def forward(self, x):
        feat = self.features(x)
        return self.head_comisuras(feat), self.head_apertura(feat)


# ==============================================================================
# DATASET EN MEMORIA PARA MUESTREO DINÁMICO
# ==============================================================================
class DynamicPatchDataset(Dataset):
    def __init__(self, sample_list):
        # sample_list: [(path, label1, label2, compound_id)]
        self.samples = sample_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, l1, l2, _ = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 64), dtype=np.uint8)
        tensor = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        return tensor, torch.tensor(l1, dtype=torch.long), torch.tensor(l2, dtype=torch.long)


# ==============================================================================
# GESTIÓN Y ESTRUCTURACIÓN DE MUESTRAS MULTI-FUENTE
# ==============================================================================
def collect_and_balance_samples(target_folders, region_type, auto_balance=True):
    """
    Escanea las carpetas físicas, asocia cada muestra a su configuración muscular
    compuesta (0 a 5) y aplica balanceo por mínimo común si está habilitado.
    """
    if region_type == "periocular":
        c1_map = {"descendida": 0, "neutra": 1, "elevada": 2}
        c2_map = {"cerrado": 0, "abierto": 1}
    else:  # boca
        c1_map = {"abajo": 0, "neutra": 1, "arriba": 2}
        c2_map = {"cerrada": 0, "abierta": 1}

    class_bins = {i: [] for i in range(6)}

    for folder_path in target_folders:
        if not os.path.exists(folder_path):
            continue
        for class_name in os.listdir(folder_path):
            class_dir = os.path.join(folder_path, class_name)
            if not os.path.isdir(class_dir):
                continue

            parts = class_name.split("_")
            try:
                if region_type == "periocular":
                    l1 = c1_map[parts[1]]
                    l2 = c2_map[parts[3]]
                else:
                    l1 = c1_map[parts[1]]
                    l2 = c2_map[parts[2]]
                compound_id = l1 * 2 + l2
            except (IndexError, KeyError):
                continue

            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    full_path = os.path.join(class_dir, fname)
                    class_bins[compound_id].append((full_path, l1, l2, compound_id))

    # Filtrar clases que tengan al menos 1 muestra
    available_bins = {k: v for k, v in class_bins.items() if len(v) > 0}
    if not available_bins:
        return [], {}

    balanced_samples = []
    distribution = {}

    if auto_balance:
        min_count = min(len(v) for v in available_bins.values())
        random.seed(42)
        for comp_id, items in available_bins.items():
            selected = random.sample(items, min_count)
            balanced_samples.extend(selected)
            distribution[comp_id] = min_count
    else:
        for comp_id, items in available_bins.items():
            balanced_samples.extend(items)
            distribution[comp_id] = len(items)

    return balanced_samples, distribution


def stratify_split(samples, val_percent):
    """Divide las muestras en train y val manteniendo la proporción exacta de cada clase."""
    if val_percent <= 0:
        return samples, []

    bins = {}
    for item in samples:
        comp_id = item[3]
        bins.setdefault(comp_id, []).append(item)

    train_set = []
    val_set = []
    random.seed(42)

    for comp_id, items in bins.items():
        random.shuffle(items)
        n_val = int(round(len(items) * (val_percent / 100.0)))
        val_set.extend(items[:n_val])
        train_set.extend(items[n_val:])

    return train_set, val_set


# ==============================================================================
# EVALUACIÓN CONJUNTA Y MÉTRICAS DE ESTADO COMPUESTO (F1)
# ==============================================================================
def evaluate_compound_metrics(model, val_loader):
    """
    Evalúa el rendimiento considerando la muestra correcta únicamente si
    ambos cabezales musculares aciertan simultáneamente.
    """
    if len(val_loader.dataset) == 0:
        return 0.0, {i: 1.0 for i in range(6)}, np.zeros((6, 6), dtype=int)

    model.eval()
    confusion_matrix = np.zeros((6, 6), dtype=int)

    with torch.no_grad():
        for x, y1, y2 in val_loader:
            out1, out2 = model(x)
            pred1 = torch.argmax(out1, dim=1)
            pred2 = torch.argmax(out2, dim=1)

            for p1, p2, t1, t2 in zip(pred1, pred2, y1, y2):
                true_comp = t1.item() * 2 + t2.item()
                pred_comp = p1.item() * 2 + p2.item()
                confusion_matrix[true_comp, pred_comp] += 1

    total_samples = np.sum(confusion_matrix)
    correct_predictions = np.trace(confusion_matrix)
    accuracy = float(correct_predictions / total_samples) if total_samples > 0 else 0.0

    f1_per_class = {}
    for c in range(6):
        tp = confusion_matrix[c, c]
        fp = np.sum(confusion_matrix[:, c]) - tp
        fn = np.sum(confusion_matrix[c, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1_per_class[c] = float(2 * (precision * recall) / (precision + recall))
        else:
            f1_per_class[c] = 0.0

    return accuracy, f1_per_class, confusion_matrix


# ==============================================================================
# BUCLE DE ENTRENAMIENTO MULTI-ETAPA DECRECIENTE
# ==============================================================================
def run_adaptive_training(model, train_pool, val_pool, epochs, lr, weight_decay, stages):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    val_loader = DataLoader(DynamicPatchDataset(val_pool), batch_size=BATCH_SIZE, shuffle=False)
    history = []

    # Progresión geométrica decreciente normalizada: w_k = (1/2)^(k-1)
    num_stages = max(1, stages)
    weights_stages = [0.5 ** k for k in range(num_stages)]
    sum_weights = sum(weights_stages)
    stage_fractions = [w / sum_weights for w in weights_stages]

    # Agrupar train_pool por estado muscular compuesto
    pool_bins = {i: [] for i in range(6)}
    for item in train_pool:
        pool_bins[item[3]].append(item)

    # Inicializar pesos de muestreo uniformes
    current_sampling_weights = {i: 1.0 for i in range(6)}

    for stage_idx in range(num_stages):
        fraction = stage_fractions[stage_idx]
        target_stage_samples = int(round(len(train_pool) * fraction))

        # Construir subconjunto ponderado por F1 inverso del estado compuesto
        stage_samples = []
        weight_sum = sum(current_sampling_weights[c] for c in pool_bins if len(pool_bins[c]) > 0)
        
        if weight_sum > 0:
            for comp_id, items in pool_bins.items():
                if len(items) == 0:
                    continue
                ratio = current_sampling_weights[comp_id] / weight_sum
                quota = int(round(target_stage_samples * ratio))
                quota = min(quota, len(items))

                if quota > 0:
                    stage_samples.extend(random.sample(items, quota))

        # Respaldo si los redondeos dejan la cuota baja
        if len(stage_samples) < target_stage_samples and len(train_pool) > 0:
            faltan = target_stage_samples - len(stage_samples)
            stage_samples.extend(random.sample(train_pool, min(faltan, len(train_pool))))

        random.shuffle(stage_samples)
        stage_loader = DataLoader(DynamicPatchDataset(stage_samples), batch_size=BATCH_SIZE, shuffle=True)

        print(f"\n  [ETAPA {stage_idx+1}/{num_stages}] Fracción: {fraction*100:.1f}% | Muestras: {len(stage_samples)}")

        # Optimización
        model.train()
        for ep in range(epochs):
            running_loss = 0.0
            for x, y1, y2 in stage_loader:
                optimizer.zero_grad()
                out1, out2 = model(x)
                loss = criterion(out1, y1) + criterion(out2, y2)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)

            ep_loss = running_loss / max(1, len(stage_loader.dataset))
            print(f"    Época [{ep+1:02d}/{epochs:02d}] - Loss: {ep_loss:.4f}")

        # Evaluación sobre el conjunto de validación completo
        acc, f1_dict, conf_mat = evaluate_compound_metrics(model, val_loader)
        print(f"    -> Validación Accuracy Compuesto: {acc*100:.2f}%")

        stage_record = {
            "stage": stage_idx + 1,
            "muestras_etapa": len(stage_samples),
            "accuracy_compuesto": round(acc, 4),
            "f1_estados": {f"clase_{k}": round(v, 4) for k, v in f1_dict.items()}
        }
        history.append(stage_record)

        # Actualizar pesos para la siguiente etapa: W_c = (1.0 - F1_c) + 0.05
        for comp_id in current_sampling_weights:
            current_sampling_weights[comp_id] = (1.0 - f1_dict.get(comp_id, 0.0)) + 0.05

    return history, acc, conf_mat.tolist()


# ==============================================================================
# EXPORTACIÓN Y CONTROL DE VERSIONES
# ==============================================================================
def get_next_version(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = os.listdir(base_dir)
    versions = []
    for d in existing_dirs:
        match = re.match(r"^v(\d+)$", d)
        if match:
            versions.append(int(match.group(1)))
    next_ver_num = max(versions) + 1 if versions else 1
    return f"v{next_ver_num}"


def export_to_onnx(model, dummy_input, output_path, out_names):
    model.eval()
    dynamic_axes_dict = {"input": {0: "batch_size"}}
    for name in out_names:
        dynamic_axes_dict[name] = {0: "batch_size"}

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=out_names,
        dynamic_axes=dynamic_axes_dict
    )


def test_onnx_model(onnx_path, model_name):
    print(f"\n[VALIDACIÓN ONNX] Comprobando {model_name}...")
    try:
        net = cv2.dnn.readNetFromONNX(onnx_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        dummy_patch = np.zeros((1, 1, 64, 64), dtype=np.float32)
        net.setInput(dummy_patch)

        t0 = time.perf_counter()
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        latencia = (time.perf_counter() - t0) * 1000.0

        shapes = [out.shape for out in outputs]
        print(f"  [OK] Carga exitosa en OpenCV DNN (Opset 18).")
        print(f"  [OK] Salidas: {shapes}")
        print(f"  [OK] Latencia CPU: {latencia:.2f} ms")
        return True, latencia
    except Exception as e:
        print(f"  [FALLO] Incompatibilidad: {e}")
        return False, 0.0


def consolidate_and_purge(source_dirs, target_dataset_dir):
    """Mueve los contenidos de todas las fuentes hacia target_dataset_dir y purga ready."""
    for s_dir in source_dirs:
        for root, dirs, files in os.walk(s_dir):
            rel_path = os.path.relpath(root, s_dir)
            dest_folder = os.path.join(target_dataset_dir, rel_path)
            os.makedirs(dest_folder, exist_ok=True)

            for file in files:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_folder, file)
                if os.path.exists(dest_file):
                    base, ext = os.path.splitext(file)
                    dest_file = os.path.join(dest_folder, f"{base}_{int(time.time()*1000)%10000}{ext}")
                shutil.move(src_file, dest_file)

        shutil.rmtree(s_dir, onerror=remove_readonly)


# ==============================================================================
# PIPELINE PRINCIPAL DE ENTRENAMIENTO
# ==============================================================================
def main():
    if not os.path.exists(DIR_DATASET_READY):
        print(f"[ERROR] No existe el directorio: '{DIR_DATASET_READY}'")
        return

    available_sources = sorted([
        d for d in os.listdir(DIR_DATASET_READY)
        if os.path.isdir(os.path.join(DIR_DATASET_READY, d))
    ])

    if not available_sources:
        print(f"[AVISO] No hay datasets en '{DIR_DATASET_READY}' para entrenar.")
        return

    print(f"\n[INFO] Datasets detectados en ready ({len(available_sources)}): {available_sources}")

    p_dirs = []
    b_dirs = []
    source_metadatas = {}
    valid_sources = []

    for src in available_sources:
        src_path = os.path.join(DIR_DATASET_READY, src)
        p_izq = os.path.join(src_path, "periocular_izq")
        p_der = os.path.join(src_path, "periocular_der")
        boca = os.path.join(src_path, "boca")

        meta_file = os.path.join(src_path, "metadata.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    source_metadatas[src] = json.load(f)
            except Exception as e:
                source_metadatas[src] = f"Error al cargar metadata: {e}"
        else:
            source_metadatas[src] = "No metadata.json provided"

        if os.path.exists(p_izq):
            p_dirs.append(p_izq)
        if os.path.exists(p_der):
            p_dirs.append(p_der)
        if os.path.exists(boca):
            b_dirs.append(boca)

        valid_sources.append(src_path)

    # 1. Recolección y balanceo por mínimo común
    periocular_samples, dist_p = collect_and_balance_samples(p_dirs, "periocular", AUTO_BALANCE_DATASET)
    boca_samples, dist_b = collect_and_balance_samples(b_dirs, "boca", AUTO_BALANCE_DATASET)

    print(f"[DATOS] Periocular: {len(periocular_samples)} muestras | Balance: {dist_p}")
    print(f"[DATOS] Boca      : {len(boca_samples)} muestras | Balance: {dist_b}")

    if len(periocular_samples) == 0 or len(boca_samples) == 0:
        print("[ERROR] Muestras insuficientes para completar el entrenamiento.")
        return

    # 2. Partición estratificada con VAL_SPLIT entero
    p_train, p_val = stratify_split(periocular_samples, VAL_SPLIT)
    b_train, b_val = stratify_split(boca_samples, VAL_SPLIT)

    print(f"[SPLIT] Periocular -> Train: {len(p_train)} | Val: {len(p_val)} ({VAL_SPLIT}%)")
    print(f"[SPLIT] Boca       -> Train: {len(b_train)} | Val: {len(b_val)} ({VAL_SPLIT}%)")

    # 3. Creación del directorio de versión
    version_id = get_next_version(DIR_MODELS)
    dir_version_root = os.path.join(DIR_MODELS, version_id)
    dir_onnx_out = os.path.join(dir_version_root, "onnx")
    dir_dataset_out = os.path.join(dir_version_root, "dataset")

    os.makedirs(dir_onnx_out, exist_ok=True)
    os.makedirs(dir_dataset_out, exist_ok=True)

    print("\n==================================================")
    print(f"CONSTRUYENDO VERSIÓN MAESTRA -> {version_id.upper()}")
    print("==================================================")

    # 4. Entrenamiento Adaptativo Periocular
    print("\n[ENTRENAMIENTO] Modelo Periocular (F1 Estado Compuesto)...")
    model_p = TinyCNNPeriocular()
    p_history, p_final_acc, p_conf_mat = run_adaptive_training(
        model=model_p,
        train_pool=p_train,
        val_pool=p_val,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        stages=ADAPTIVE_STAGES
    )

    path_onnx_p = os.path.join(dir_onnx_out, "model_periocular.onnx")
    export_to_onnx(model_p, torch.randn(1, 1, 64, 64), path_onnx_p, ["out_ceja", "out_ojo"])
    print(f"[EXPORTADO] {path_onnx_p}")

    # 5. Entrenamiento Adaptativo Boca
    print("\n[ENTRENAMIENTO] Modelo Boca (F1 Estado Compuesto)...")
    model_b = TinyCNNBoca()
    b_history, b_final_acc, b_conf_mat = run_adaptive_training(
        model=model_b,
        train_pool=b_train,
        val_pool=b_val,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        stages=ADAPTIVE_STAGES
    )

    path_onnx_b = os.path.join(dir_onnx_out, "model_boca.onnx")
    export_to_onnx(model_b, torch.randn(1, 1, 64, 64), path_onnx_b, ["out_comisuras", "out_apertura"])
    print(f"[EXPORTADO] {path_onnx_b}")

    # 6. Validación técnica en CPU con OpenCV DNN
    ok_p, lat_p = test_onnx_model(path_onnx_p, "Periocular")
    ok_b, lat_b = test_onnx_model(path_onnx_b, "Boca")

    if not (ok_p and ok_b):
        print("\n[PELIGRO] Falló la validación técnica en OpenCV DNN. Abortando congelación.")
        return

    # 7. Consolidación atómica de datasets y purga de Data/dataset/ready/
    print(f"\n[CONSOLIDACIÓN] Moviendo datasets a {dir_dataset_out} y purgando ready...")
    consolidate_and_purge(valid_sources, dir_dataset_out)

    # 8. Manifiesto maestro auditado
    master_metadata = {
        "version": version_id,
        "timestamp": datetime.now().strftime("%d_%m_%y_%H_%M"),
        "configuracion": {
            "batch_size": BATCH_SIZE,
            "epocas_por_etapa": EPOCHS,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "val_split_porcentaje": VAL_SPLIT,
            "auto_balance_activado": AUTO_BALANCE_DATASET,
            "etapas_adaptativas": ADAPTIVE_STAGES
        },
        "fuentes_utilizadas": available_sources,
        "metadatos_origenes": source_metadatas,
        "auditoria_muestras": {
            "periocular": {
                "total_balanceado": len(periocular_samples),
                "distribucion_clases": dist_p,
                "train": len(p_train),
                "val": len(p_val)
            },
            "boca": {
                "total_balanceado": len(boca_samples),
                "distribucion_clases": dist_b,
                "train": len(b_train),
                "val": len(b_val)
            }
        },
        "rendimiento_modelos": {
            "periocular": {
                "accuracy_compuesto_final": round(p_final_acc, 4),
                "latencia_cpu_ms": round(lat_p, 2),
                "historial_etapas": p_history,
                "matriz_confusion_final": p_conf_mat
            },
            "boca": {
                "accuracy_compuesto_final": round(b_final_acc, 4),
                "latencia_cpu_ms": round(lat_b, 2),
                "historial_etapas": b_history,
                "matriz_confusion_final": b_conf_mat
            }
        }
    }

    metadata_path = os.path.join(dir_version_root, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(master_metadata, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Manifiesto maestro guardado en: {metadata_path}")
    print(f"\n[ÉXITO] Versión {version_id} generada y espacio 'ready' liberado.")


if __name__ == "__main__":
    main()