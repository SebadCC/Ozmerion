# diagnostico.py
import os
import cv2
import numpy as np
from patch2 import TasksGeometricPatchExtractor

DIR_PRUEBA = os.path.join("Data", "prueba")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ==============================================================================
# CONFIGURACIÓN IDÉNTICA A BUILDER2.PY
# ==============================================================================
MAX_PITCH_DEG = 20.0
MAX_YAW_DEG = 25.0
MAX_ROLL_DEG = 15.0

MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

PADDING_PERIOCULAR = 0.45
PADDING_BOCA = 0.40

# ==============================================================================
# UMBRALES BIOMECÁNICOS CALIBRADOS
# ==============================================================================
EAR_OPEN_THRESHOLD = 0.19
MAR_OPEN_THRESHOLD = 0.12

CORNER_UP_THRESHOLD = 0.08
CORNER_DOWN_THRESHOLD = -0.06

# Cejas: Colapsos verticales normalizados por IOD y elevación del arco
BROW_DIST_IN_LOW_THRESHOLD = 0.46
BROW_DIST_MID_LOW_THRESHOLD = 0.58
BROW_ARCH_HIGH_THRESHOLD = 0.32     # Elevación de curvatura respecto a su base


def calculate_ear(coords, eye_pts):
    p1, p2, p3, p4, p5, p6 = coords[eye_pts]
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    if h < 1e-4:
        return 0.0
    return float((v1 + v2) / (2.0 * h))


def classify_eyebrow_audit(p_in, p_mid, p_out, p_canto_int, p_parpado_sup, iod):
    brow_width = max(1e-4, float(np.linalg.norm(p_out - p_in)))

    # 1. Colapso vertical medial (hacia el lagrimal/canto interno)
    # Como Y crece hacia abajo en la imagen, canto_int[1] > p_in[1]
    dist_in_y = float((p_canto_int[1] - p_in[1]) / iod)

    # 2. Colapso vertical central (hacia el párpado superior)
    dist_mid_y = float((p_parpado_sup[1] - p_mid[1]) / iod)

    # 3. Elevación del arco ciliar
    y_base_interp = (p_in[1] + p_out[1]) / 2.0
    arch_height = float((y_base_interp - p_mid[1]) / brow_width)

    # Decisión biomecánica
    is_descendida = (dist_in_y < BROW_DIST_IN_LOW_THRESHOLD) or (dist_mid_y < BROW_DIST_MID_LOW_THRESHOLD)

    if is_descendida:
        state = "descendida"
    elif arch_height > BROW_ARCH_HIGH_THRESHOLD:
        state = "elevada"
    else:
        state = "neutra"

    debug = {
        "dist_in_y": round(dist_in_y, 4),
        "dist_mid_y": round(dist_mid_y, 4),
        "arch_height": round(arch_height, 4),
        "is_descendida": is_descendida
    }
    return state, debug


def run_diagnostico():
    if not os.path.exists(DIR_PRUEBA):
        print(f"[ERROR] No existe el directorio: '{DIR_PRUEBA}'")
        return

    archivos = sorted([
        f for f in os.listdir(DIR_PRUEBA)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])

    if not archivos:
        print(f"[AVISO] No hay imágenes en '{DIR_PRUEBA}'.")
        return

    extractor = TasksGeometricPatchExtractor(
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

    # Puntos de referencia canónicos (Topología MediaPipe 478 Mesh)
    CANTO_INT_IZQ, CANTO_INT_DER = 133, 362
    PARPADO_SUP_IZQ, PARPADO_SUP_DER = 159, 386

    CEJA_IN_IZQ, CEJA_MID_IZQ, CEJA_OUT_IZQ = 55, 105, 46
    CEJA_IN_DER, CEJA_MID_DER, CEJA_OUT_DER = 285, 334, 276

    # Topología corregida para ambos ojos (6 puntos canónicos)
    OJO_IZQ_PTS = [33, 160, 158, 133, 153, 144]
    OJO_DER_PTS = [263, 385, 387, 362, 373, 380]

    COMISURA_IZQ, COMISURA_DER = 61, 291
    LABIO_INT_SUP, LABIO_INT_INF = 13, 14

    print(f"\n{'='*75}")
    print(f"AUDITORÍA BIOMECÁNICA (COLAPSO VERTICAL CEJA-OJO)")
    print(f"{'='*75}\n")

    for fname in archivos:
        path = os.path.join(DIR_PRUEBA, fname)
        frame = cv2.imread(path)
        if frame is None:
            continue

        patches, status, telemetry = extractor.extract_patches(frame)

        if patches is None or status != "OK" or telemetry is None:
            print(f"[-] {fname:25s} | DESCARTE PATCH2 -> Status: {status}")
            continue

        coords = telemetry["landmarks"]
        pitch, yaw, roll = telemetry["angles"]
        iod = np.linalg.norm(coords[CANTO_INT_DER] - coords[CANTO_INT_IZQ])
        if iod < 1e-4:
            continue

        # 1. Párpados (EAR)
        ear_izq = calculate_ear(coords, OJO_IZQ_PTS)
        ear_der = calculate_ear(coords, OJO_DER_PTS)
        ojo_izq_st = "cerrado" if ear_izq < EAR_OPEN_THRESHOLD else "abierto"
        ojo_der_st = "cerrado" if ear_der < EAR_OPEN_THRESHOLD else "abierto"

        # 2. Cejas (Colapso vertical directo hacia el ojo correspondiente)
        ceja_izq_st, deb_c_izq = classify_eyebrow_audit(
            coords[CEJA_IN_IZQ], coords[CEJA_MID_IZQ], coords[CEJA_OUT_IZQ],
            coords[CANTO_INT_IZQ], coords[PARPADO_SUP_IZQ], iod
        )
        ceja_der_st, deb_c_der = classify_eyebrow_audit(
            coords[CEJA_IN_DER], coords[CEJA_MID_DER], coords[CEJA_OUT_DER],
            coords[CANTO_INT_DER], coords[PARPADO_SUP_DER], iod
        )

        # 3. Complejo Bucal (MAR y Comisuras)
        ancho_boca = np.linalg.norm(coords[COMISURA_DER] - coords[COMISURA_IZQ])
        alt_interna = np.linalg.norm(coords[LABIO_INT_INF] - coords[LABIO_INT_SUP])
        mar = float(alt_interna / ancho_boca) if ancho_boca > 1e-4 else 0.0
        boca_st = "abierta" if mar >= MAR_OPEN_THRESHOLD else "cerrada"

        y_centro = (coords[LABIO_INT_SUP][1] + coords[LABIO_INT_INF][1]) / 2.0
        y_comis = (coords[COMISURA_IZQ][1] + coords[COMISURA_DER][1]) / 2.0
        diff_comis = (y_centro - y_comis) / iod if iod > 1e-4 else 0.0

        if diff_comis > CORNER_UP_THRESHOLD:
            comis_st = "arriba"
        elif diff_comis < CORNER_DOWN_THRESHOLD:
            comis_st = "abajo"
        else:
            comis_st = "neutra"

        print(f"[+] {fname}")
        print(f"    Pose 3D     : Pitch={pitch:+.1f}° | Yaw={yaw:+.1f}° | Roll={roll:+.1f}° | IOD={iod:.1f}px")
        print(f"    Ojo Izq     : EAR={ear_izq:.4f} (Umbral {EAR_OPEN_THRESHOLD}) -> [{ojo_izq_st.upper()}]")
        print(f"    Ojo Der     : EAR={ear_der:.4f} (Umbral {EAR_OPEN_THRESHOLD}) -> [{ojo_der_st.upper()}]")
        print(f"    Ceja Izq    : in_y={deb_c_izq['dist_in_y']:.3f} (<0.16) | mid_y={deb_c_izq['dist_mid_y']:.3f} (<0.17) | arch={deb_c_izq['arch_height']:+.3f} -> [{ceja_izq_st.upper()}]")
        print(f"    Ceja Der    : in_y={deb_c_der['dist_in_y']:.3f} (<0.16) | mid_y={deb_c_der['dist_mid_y']:.3f} (<0.17) | arch={deb_c_der['arch_height']:+.3f} -> [{ceja_der_st.upper()}]")
        print(f"    Boca        : MAR={mar:.4f} (Umbral {MAR_OPEN_THRESHOLD}) -> [{boca_st.upper()}]")
        print(f"    Comisuras   : diff={diff_comis:+.4f} (Down<{CORNER_DOWN_THRESHOLD} / Up>{CORNER_UP_THRESHOLD}) -> [{comis_st.upper()}]")
        print(f"    >>> CLASIFICACIÓN FINAL:")
        print(f"        P_IZQ: ceja_{ceja_izq_st}_ojo_{ojo_izq_st}")
        print(f"        P_DER: ceja_{ceja_der_st}_ojo_{ojo_der_st}")
        print(f"        BOCA : comisuras_{comis_st}_{boca_st}\n")


if __name__ == "__main__":
    run_diagnostico()