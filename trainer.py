# trainer.py
import os
import re
import shutil
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# RUTAS PRINCIPALES
# ==============================================================================
ROOT_DATA = "Data"
DIR_DATASET = os.path.join(ROOT_DATA, "dataset")
DIR_VERSIONES = os.path.join(ROOT_DATA, "versiones")

# ==============================================================================
# ARQUITECTURAS TINY-CNN CON DROPOUT (ANTI-OVERFITTING)
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
# DATASETS PYTORCH BASADOS EN ESTRUCTURA CANÓNICA
# ==============================================================================
class PeriocularDataset(Dataset):
    def __init__(self, roots):
        self.samples = []
        ceja_map = {"descendida": 0, "neutra": 1, "elevada": 2}
        ojo_map = {"cerrado": 0, "abierto": 1}

        for root_dir in roots:
            if not os.path.exists(root_dir):
                continue
            for folder_name in os.listdir(root_dir):
                folder_path = os.path.join(root_dir, folder_name)
                if not os.path.isdir(folder_path):
                    continue

                # Formato esperado de carpeta: ceja_{estado}_ojo_{estado}
                parts = folder_name.split("_")
                try:
                    c_label = ceja_map[parts[1]]
                    o_label = ojo_map[parts[3]]
                except (IndexError, KeyError):
                    continue

                for f in os.listdir(folder_path):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.samples.append((os.path.join(folder_path, f), c_label, o_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, c_label, o_label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 64), dtype=np.uint8)
        tensor = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        return tensor, torch.tensor(c_label, dtype=torch.long), torch.tensor(o_label, dtype=torch.long)


class BocaDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        com_map = {"abajo": 0, "neutra": 1, "arriba": 2}
        ape_map = {"cerrada": 0, "abierta": 1}

        if os.path.exists(root_dir):
            for folder_name in os.listdir(root_dir):
                folder_path = os.path.join(root_dir, folder_name)
                if not os.path.isdir(folder_path):
                    continue

                # Formato esperado: comisuras_{posicion}_{estado}
                parts = folder_name.split("_")
                try:
                    com_label = com_map[parts[1]]
                    ape_label = ape_map[parts[2]]
                except (IndexError, KeyError):
                    continue

                for f in os.listdir(folder_path):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.samples.append((os.path.join(folder_path, f), com_label, ape_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, com_label, ape_label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 64), dtype=np.uint8)
        tensor = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        return tensor, torch.tensor(com_label, dtype=torch.long), torch.tensor(ape_label, dtype=torch.long)


# ==============================================================================
# GESTIÓN DE VERSIONES Y EXPORTACIÓN ONNX
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


def train_loop(model, dataloader, epochs=8, lr=0.001, weight_decay=1e-2):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for x, y1, y2 in dataloader:
            optimizer.zero_grad()
            out1, out2 = model(x)
            loss = criterion(out1, y1) + criterion(out2, y2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / max(1, len(dataloader.dataset))
        print(f"    Época [{epoch+1:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f}")


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
        print(f"  [OK] Dimensiones de salida: {shapes}")
        print(f"  [OK] Latencia CPU: {latencia:.2f} ms")
        return True
    except Exception as e:
        print(f"  [FALLO] Incompatibilidad detectada: {e}")
        return False


# ==============================================================================
# FLUJO PRINCIPAL
# ==============================================================================
def main():
    if not os.path.exists(DIR_DATASET) or not os.listdir(DIR_DATASET):
        print(f"[ERROR] No se encontraron parches en '{DIR_DATASET}'. Ejecuta builder.py primero.")
        return

    p_roots = [os.path.join(DIR_DATASET, "periocular_izq"), os.path.join(DIR_DATASET, "periocular_der")]
    b_root = os.path.join(DIR_DATASET, "boca")

    periocular_data = PeriocularDataset(p_roots)
    boca_data = BocaDataset(b_root)

    print(f"[DATOS] Muestras perioculares: {len(periocular_data)}")
    print(f"[DATOS] Muestras boca        : {len(boca_data)}")

    if len(periocular_data) == 0 or len(boca_data) == 0:
        print("[ERROR] No hay muestras suficientes para entrenar.")
        return

    version_id = get_next_version(DIR_VERSIONES)
    dir_version_root = os.path.join(DIR_VERSIONES, version_id)
    dir_onnx_out = os.path.join(dir_version_root, "onnx")
    dir_dataset_out = os.path.join(dir_version_root, "dataset")

    os.makedirs(dir_onnx_out, exist_ok=True)
    os.makedirs(dir_dataset_out, exist_ok=True)

    print("\n==================================================")
    print(f"INICIANDO PIPELINE DE ENTRENAMIENTO -> {version_id.upper()}")
    print("==================================================")

    loader_periocular = DataLoader(periocular_data, batch_size=8, shuffle=True)
    loader_boca = DataLoader(boca_data, batch_size=8, shuffle=True)

    # 1. Entrenamiento Periocular
    print("\n[ENTRENAMIENTO] Modelo Periocular...")
    model_periocular = TinyCNNPeriocular()
    train_loop(model_periocular, loader_periocular, epochs=8, lr=0.001, weight_decay=1e-2)

    path_onnx_p = os.path.join(dir_onnx_out, "model_periocular.onnx")
    export_to_onnx(model_periocular, torch.randn(1, 1, 64, 64), path_onnx_p, ["out_ceja", "out_ojo"])
    print(f"[EXPORTADO] {path_onnx_p}")

    # 2. Entrenamiento Boca
    print("\n[ENTRENAMIENTO] Modelo Boca...")
    model_boca = TinyCNNBoca()
    train_loop(model_boca, loader_boca, epochs=8, lr=0.001, weight_decay=1e-2)

    path_onnx_b = os.path.join(dir_onnx_out, "model_boca.onnx")
    export_to_onnx(model_boca, torch.randn(1, 1, 64, 64), path_onnx_b, ["out_comisuras", "out_apertura"])
    print(f"[EXPORTADO] {path_onnx_b}")

    # 3. Validación ONNX en CPU
    ok_p = test_onnx_model(path_onnx_p, "Periocular")
    ok_b = test_onnx_model(path_onnx_b, "Boca")

    if not (ok_p and ok_b):
        print("\n[PELIGRO] Falló la validación de los modelos ONNX.")
        return

    # 4. Congelación atómica del dataset
    print(f"\n[ARCHIVADO] Congelando dataset en {dir_dataset_out}...")
    for item in os.listdir(DIR_DATASET):
        src_item = os.path.join(DIR_DATASET, item)
        dst_item = os.path.join(dir_dataset_out, item)
        shutil.move(src_item, dst_item)

    print(f"[OK] Pipeline finalizado con éxito. Nueva versión creada: {version_id}.")


if __name__ == "__main__":
    main()