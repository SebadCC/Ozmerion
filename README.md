# Project Ozmerion  
  
Ozmerion es un sistema modular y ligero de visión por computador diseñado para la detección, extracción y clasificación en tiempo real de dinámicas faciales y microexpresiones (región periocular y complejo bucal).  
  
El objetivo final del proyecto es construir un pipeline integral capaz de detectar microexpresiones espontáneas de alta velocidad (40–200 ms) combinando clasificadores neuromusculares locales basados en Action Units (FACS) con una capa temporal cinemática de autenticidad gestual. Actualmente el sistema se encuentra cerrando la Fase 1 (Extracción, curaduría de datasets y entrenamiento de clasificadores locales robustos), preparando la transición hacia la Fase 2 (Matriz estática de 36 estados de expresión) y la Fase 3 (Análisis dinámico y temporal en ventanas de 5 segundos a 10 FPS).  
  
El pipeline opera en CPU mediante MediaPipe Tasks (FaceLandmarker v2 con Blendshapes) para la alineación geométrica y modelos Tiny-CNN optimizados en formato ONNX ejecutados a través de OpenCV DNN, alcanzando latencias de inferencia de ~2-3 ms (~30-35+ FPS)[cite: 2, 3, 5].  
  
---  
  
## Novedades de la Versión  
  
* Transición de FaceMesh a MediaPipe Tasks (patch2.py): Integración del binario canónico face_landmarker_v2_with_blendshapes.task, incorporando cálculo de ángulos de Euler (Pitch, Yaw, Roll) a partir de la matriz de transformación 4x4 y normalización fotométrica CLAHE sobre parches de 64x64[cite: 3].  
* Evolución de Builders:  
  - builder.py: Generador base legacy sobre landmarks simples[cite: 5].  
  - builder2.py: Clasificación biomecánica estricta mediante distancias anatómicas relativas al IOD, vector corrugador compuesto y amortiguamiento por banda de incertidumbre[cite: 1].  
  - builder3.py: Clasificación neuromuscular de alta precisión basada en sinergias de FaceBlendshapes (AU4, AU1, AU2, AU12, AU15, aperturas palpebrales y bucales corregidas por mirada)[cite: 4].  
* Módulo de Balanceo Multi-Dataset (balancer.py): Balanceador determinista que calcula un N objetivo unificado sobre todos los datasets en Data/dataset/ready/. Incorpora inversión horizontal cruzada de ojos (efecto espejo entre ojos opuestos), submuestreo regular uniforme con paso variable y síntesis estocástica (data augmentation) con inyección de metadatos en bloques tEXt de archivos PNG.  
* Herramientas de Soporte:  
  - sampler.py: Extractor y muestreador de frames relevantes a partir de secuencias de video crudas.  
  - diagnostico.py: Inspector numérico de telemetría y scores de blendshapes para validación de umbrales sobre muestras dudosas.  
  
---  
  
## Instalación  
  
1. Clonar el repositorio localmente[cite: 5].  
2. Configurar un entorno virtual con Python 3.10 (versiones superiores presentan incompatibilidades con ciertas dependencias de MediaPipe y exportadores ONNX)[cite: 5].  
3. Asegurar la presencia del binario face_landmarker_v2_with_blendshapes.task en la raíz del proyecto[cite: 3].  
4. Instalar las dependencias requeridas[cite: 5]:  
   pip install opencv-python mediapipe torch torchvision onnx numpy pillow  
  
---  
  
## Flujo de Trabajo  
  
### 1. Extracción y Clasificación del Dataset (builder3.py)  
1. Ubicar los lotes de imágenes o videos procesados en carpetas dentro de Data/raw/ready/ (ej. Data/raw/ready/ADFES_F01/, ADFES_F02/, etc.)[cite: 4].  
2. Ejecutar builder3.py para extraer los parches de 64x64 y clasificarlos con las sinergias de blendshapes[cite: 4]:  
   python builder3.py  
3. Los parches clasificados se guardarán en Data/dataset/ready/builder3/ junto con su correspondiente metadata.json, derivando los casos dudosos a incertidumbre/[cite: 4].  
  
### 2. Balanceo y Síntesis de Datos (balancer.py)  
1. Una vez generados uno o más datasets en Data/dataset/ready/, ejecutar el balanceador:  
   python balancer.py  
2. El script calculará el target unificado, aplicará inversión cruzada en ojos, podará excedentes regulares y sintetizará las muestras faltantes.  
3. El resultado equilibrado se escribirá en Data/dataset/ready/{dataset}_bal/.  
  
### 3. Entrenamiento (trainer.py)  
1. Configurar la ruta hacia el dataset balanceado deseado dentro de trainer.py.  
2. Ejecutar el entrenamiento de las Tiny-CNNs[cite: 5]:  
   python trainer.py  
3. Los modelos resultantes se exportarán a Data/models/vX/onnx/ (model_periocular.onnx y model_boca.onnx)[cite: 2, 5].  
  
### 4. Inferencia en Tiempo Real (realtime.py)  
1. Indicar el número de versión generado (VERSION = X) en realtime.py (línea 13)[cite: 2].  
2. Verificar que PATCHER = 2 para operar con el extractor moderno de Tasks[cite: 2].  
3. Ejecutar[cite: 5]:  
   python realtime.py  
  
---  
  
## Herramientas Adicionales  
  
### Diagnóstico de Blendshapes (diagnostico.py)  
Permite inspeccionar los scores exactos de activación muscular sobre fotogramas individuales o lotes pequeños:  
python diagnostico.py  
  
### Pseudo-Labeling (retrainer.py)  
Si se desea utilizar un modelo ONNX ya entrenado para clasificar de forma autónoma un lote crudo masivo[cite: 5]:  
1. Colocar las imágenes en Data/raw/ready/[cite: 5].  
2. Definir la versión del modelo en retrainer.py[cite: 5].  
3. Ejecutar retrainer.py para generar un nuevo conjunto preetiquetado por inferencia[cite: 5].  
  
### Captura Supervisada en Vivo (realtime.py)  
Permite enriquecer el dataset capturando expresiones específicas directamente desde la cámara[cite: 5]:  
* Asignación de Tags en Teclado[cite: 2, 5]:  
  - Ojos: [1] Abierto | [2] Cerrado[cite: 2, 5]  
  - Cejas: [3] Neutra | [4] Arriba | [5] Abajo[cite: 2, 5]  
  - Boca: [6] Abierta | [7] Cerrada[cite: 2, 5]  
  - Comisuras: [8] Neutra | [9] Arriba | [0] Abajo[cite: 2, 5]  
  - Limpiar Tags: [BACKSPACE][cite: 2, 5]  
* Captura: Presionar C para foto única o barra ESPACIO para ráfagas continuas a 5 FPS[cite: 2, 5].  
* Si los 4 tags están activos, la muestra se almacena estructurada en Data/dataset/real/{sesion}/; si no hay tags completos, se guarda como crudo en Data/raw/archive/{sesion}/[cite: 2].  
* Salir con Q o ESC[cite: 2, 5].  
  
---  
  
## Estructura del Directorio Data/  
  
Data/  
├── raw/                         # Imágenes y videos de entrada crudos  
│   ├── ready/                   # Lotes organizados listos para procesar (ej. ADFES_F01, etc.)  
│   └── archive/                 # Capturas crudas guardadas desde realtime.py  
├── dataset/                     # Parches 64x64 generados y estructurados  
│   ├── ready/                   # Datasets clasificados por builders  
│   │   ├── builder2/            # Salida del clasificador biomecánico  
│   │   ├── builder3/            # Salida del clasificador por Blendshapes  
│   │   ├── builder3_bal/        # Salida procesada y balanceada por balancer.py  
│   │   └── ...  
│   └── real/                    # Muestras supervisadas capturadas en vivo  
└── models/                      # Pesos, checkpoints y modelos exportados  
    ├── v10/  
    ├── v11/  
    │   ├── checkpoints/  
    │   └── onnx/  
    │       ├── model_periocular.onnx  
    │       └── model_boca.onnx  
    └── ...  
  
---  
  
## Configuraciones Adicionales  
  
Todos los módulos del pipeline (patch2.py, builder2.py, builder3.py, balancer.py, trainer.py, realtime.py) mantienen sus parámetros de calibración, umbrales y tolerancias en bloques configurables al inicio del archivo, justo después de las importaciones, permitiendo ajustar umbrales anatómicos, tolerancias angulares y rutas de forma desacoplada[cite: 1, 2, 3, 4, 5].  
  
---  
  
La modularidad de Ozmerion permite iterar independientemente sobre la extracción geométrica, la curaduría del dato y la optimización de los modelos. Los avances consolidados en esta versión establecen una base de parches balanceada y libre de sesgos morfológicos basales, sentando el terreno para la integración del motor cinemático temporal.  
  
**Ozmerion** — *Desarrollado por Sebastián Cortés (SebadCC)*  



P.D. Este ReadMe lo realice con IA y no lo he revisado, cualquier duda estoy al pendiente, tengo sueño y no queria hacer el readme a mano como la otra vez  