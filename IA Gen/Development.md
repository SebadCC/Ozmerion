# HISTORIAL DE DESARROLLO Y TRANSFORMACIÓN ESTRATÉGICA (OZMERION)

> **PROPÓSITO DEL DOCUMENTO:**
> Este registro compila la evolución cronológica del proyecto **Ozmerion**, detallando tanto las transformaciones en la base de código como los cambios de paradigma conceptual, cuellos de botella superados y la maduración de la arquitectura biomecánica.

---

## 1. Línea de Tiempo y Evolución de Versiones

| Periodo / Fecha | Hito Principal | Versiones de Modelo | Módulos Clave | Paradigma Técnico Dominante |
| :--- | :--- | :--- | :--- | :--- |
| **15–16 Sep 2026** | Nacimiento del Pipeline Base | `v1` a `v9` | `patch.py`, `builder.py`, `trainer.py`, `realtime.py` | MediaPipe FaceMesh clásico, distancias 2D planas, recorte básico de parches. |
| **17–28 Sep 2026** | Transición a Tasks y Biomecánica | `v10`, `v11` | `patch2.py`, `builder2.py`, `sampler.py`, `diagnostico.py` | FaceLandmarker v2 3D, EAR/MAR normalizados por IOD, cinemática de video. |
| **28–30 Sep 2026** | Sinergias FACS y Balanceo Determinista | `v12` | `builder3.py`, `balancer.py`, `trainer.py` (adaptativo) | Co-activación muscular FACS, inversión cruzada L/R, metadatos PNG `tEXt`. |
| **1–15 Oct 2026** | Expansión Masiva 17x y Consolidación Canónica | `v14` (Hito Fase 1) | `trainer.py`, `realtime.py` (RAM preload) | 1.4M parches con 16 arquetipos de luz Blender. 12-17 ms de latencia total en CPU. |
| **Próxima Etapa** | Fusión Jerárquica y Cobertura Facial Estratégica | Fase 2 | Grafo de Congruencia, 6-7 parches | Expansión facial estratégica (Nasolabial AU9/10, Mentón AU17, Mejillas AU6/12/14) y grafo FACS. |
| **Etapa Posterior**| Motor Cinemático y Desambiguación Temporal | Fase 3 | Búfer FIFO, Derivadas cinemáticas | Dinámicas 30 FPS, derivadas físicas de velocidad/aceleración para microexpresiones (40-200 ms). |
| **Medio Plazo** | Soberanía Tecnológica Completa | Fase 4 | `patcher3.py` (Modelo Propio) | Extractor propio multi-tarea en CPU (detección, pose 3D reducida PnP, recorte directo). |

---

## 2. Evolución Cronológica Detallada

### Época 1: Prototipado Inicial y Validación de Factibilidad (15–16 Septiembre 2026)
* **Commits Clave:** `33398c1 ("holaaaaa")`, `733cf58`, `87c9067`, `652660f`.
* **Modelos Entrenados:** `v1` a `v9` en iteraciones sucesivas de calibración de hiperparámetros.
* **Componentes:**
  * `patch.py`: Extractor apoyado en MediaPipe FaceMesh estándar. Cálculo de rotación plana en 2D (Roll) y normalización básica.
  * `builder.py`: Clasificación elemental de muestras basada en distancias euclidianas directas en píxeles.
  * `trainer.py` (v1): Red Tiny-CNN simple en PyTorch y validación de exportación a grafos ONNX para ejecución en `cv2.dnn`.
  * `realtime.py`: Primer bucle de inferencia directa sobre cámara.
* **Cuellos de Botella y Aprendizajes:**
  * **Fragilidad de pose:** La dependencia de distancias euclidianas crudas provocaba que pequeñas inclinaciones de la cabeza (Pitch o Yaw) falsearan totalmente la detección de apertura bucal u ocular.
  * **Sesgo severo a clases neutras:** El rostro humano pasa la mayor parte del tiempo en reposo. Los datasets crudos generaban un desbalance abrumador hacia muestras neutras, debilitando la sensibilidad ante contracciones sutiles.

---

### Época 2: Adopción de MediaPipe Tasks y Enfoque Biomecánico (17–28 Septiembre 2026)
* **Commits Clave:** Inicio de la refactorización profunda hacia MediaPipe Tasks.
* **Modelos Entrenados:** `v10` (modelo de prueba oráculo) y `v11` (primer modelo basado en métricas geométricas normalizadas).
* **Innovaciones de Código:**
  * **Binario Canónico Local (`patch2.py`):** Integración de `face_landmarker_v2_with_blendshapes.task`. Descomposición de la matriz rígida de transformación 4×4 para derivar ángulos de Euler formales (Pitch, Yaw, Roll) en grados.
  * **Ampliación de Cobertura Anatómica:** El margen de recorte periocular (`PADDING_PERIOCULAR`) se incrementó de 0.35 a 0.45 para integrar obligatoriamente la cabeza de la ceja, el puente nasal y la zona de acción del corrugador superciliar.
  * **Métricas Biomecánicas Invariantes (`builder2.py`):** Reemplazo de umbrales en píxeles crudos por ratios anatómicos normalizados:
    * EAR (*Eye Aspect Ratio*) para apertura ocular.
    * MAR (*Mouth Aspect Ratio*) para apertura bucal.
    * IOD (*Inter-Ocular Distance* / Cantos internos) como unidad base de escala métrica del rostro.
  * **Ingesta Cinemática de Video (`sampler.py`):** Módulo de tracking temporal para extraer fotogramas clave en videos continuos: obligatoriamente inicio/fin, picos de aceleración muscular, valles y mesetas de actividad.
  * **Auditoría Matemática (`diagnostico.py`):** Script analítico para graficar e inspeccionar los ratios biomecánicos en muestras individuales dudosas sin ejecutar inferencia visual completa.
* **El Gran Cuello de Botella Superado:**
  * En las primeras pruebas de `builder2.py`, se descartaba hasta el **97% de los fotogramas** de video debido a un filtrado de pose 2D interno rígido (`yaw_ratio` basado en la posición nasal).
  * **Solución Estratégica:** Desacoplar completamente el extractor (`patch2.py`), dejando que solo compute y reporte la telemetría angular 3D, y delegando los criterios de tolerancia a las capas consumidoras.

---

### Época 3: Sinergias FACS, Balanceo Determinista y Consolidación de v12 (28–30 Septiembre 2026)
* **Commits Clave:** `dc98621 ("Pequeñas modificaciones")`, `f15e69b`.
* **Modelos Entrenados:** `v12` (Versión maestra actual del sistema).
* **Transformaciones Fundamentales:**
  * **De Distancias Euclidianas a Sinergias FACS (`builder3.py`):** Se identificó que las distancias geométricas puras fallaban en capturar interacciones musculares sutiles (ej. la subida de mejillas comprime pasivamente el párpado inferior simulando un pestañeo). Se reescribió el clasificador basándose en sinergias de Action Units (FACS):
    * *Ojo:* `eyeBlink` corregido por mirada inferior (`eyeLookDown`) y tensión orbitaria (`eyeSquint`).
    * *Ceja:* Puntuación compuesta de ceño fruncido (AU4 + `noseSneer` + `cheekSquint`) vs elevación (AU1 medial + AU2 lateral).
    * *Boca:* Apertura compuesta (mandíbula + elongación labial) y comisuras mediante competencia entre cigomáticos (AU12 `mouthSmile` + `dimple`) y depresores (AU15 `mouthFrown` + `mouthShrugLower`).
  * **Política "Zero Toxic Labels":** Se introdujo una franja de histeresis del $\pm 10\%$ (`UNCERTAINTY_MARGIN`). Toda muestra situada en zonas límite o de conflicto muscular se expulsa a carpetas de `incertidumbre/`, garantizando una distribución de entrenamiento sin ruido.
  * **Balanceo Determinista Multi-Dataset (`balancer.py`):**
    * *Espejo Cruzado Izquierda $\leftrightarrow$ Derecha:* Las muestras de ojo derecho se invierten horizontalmente para alimentar la clase equivalente del ojo izquierdo (y viceversa), duplicando la representatividad física de clases minoritarias.
    * *Poda Regular:* Submuestreo uniforme con paso variable sobre clases hiper-pobladas (neutras).
    * *Síntesis Estocástica:* Micro-rotaciones ($\pm 3^\circ$), micro-traslaciones ($\pm 2\text{ px}$), ajustes de contraste/brillo y desenfoques controlados.
    * *Trazabilidad en Metadatos PNG:* Parches sintéticos inyectan los parámetros exactos de generación en bloques `tEXt` del formato PNG.
  * **Entrenamiento Multi-Tarea Adaptativo (`trainer.py`):**
    * Estructuración definitiva de `TinyCNNPeriocular` y `TinyCNNBoca` con cabezas independientes (Ceja $\times$ Ojo y Comisura $\times$ Apertura).
    * Entrenamiento adaptativo en 3 etapas con reducción progresiva de tasa de aprendizaje y regularización por Dropout (0.3).
  * **Rendimiento de la Versión v12:**
    * Complejo Bucal: **94.26%** de exactitud compuesta.
    * Complejo Periocular: **88.33%** de exactitud compuesta.
    * Latencia de inferencia en CPU: **~1.17 a 7.47 ms** por frame completo.

---

### Época 4: Diagnóstico de Entornos Reales y Camino a v13 (Octubre 2026)
* **Diagnóstico del "Domain Shift":** Al someter a [v12](file:///c:/Users/JSCC2/OneDrive/Documentos/PROYECTOS/Ozmerion/Data/models/v12/metadata.json) a pruebas en vivo bajo iluminación cotidiana con [realtime.py](file:///c:/Users/JSCC2/OneDrive/Documentos/PROYECTOS/Ozmerion/realtime.py), se detectó que la región periocular presenta oscilaciones de falso parpadeo ante **sombras cenitales** (la sombra de las cejas o la cuenca ocular en penumbra atenúa el contraste del ojo abierto, confundiéndolo con cerrado).
* **Estrategia para v13:** Robustecer el clasificador periocular inyectando muestras ricas en penumbra orbital y contraste adverso, mediante:
  1. Capturas supervisadas en caliente con luz ambiental cotidiana mediante [realtime.py](file:///mnt/windows/Users/JSCC2/Documents/PROYECTOS/Ozmerion/realtime.py).
  2. Generación controlada con variaciones sistemáticas de luz cenital dura para forzar a la red a desacoplar el brillo orbital de la apertura palpebral real.

---

### Época 5: Expansión 17x e Hito Canónico v14 (Octubre 2026)

#### 1. Expansión Masiva del Dataset ADFES (1,399,117 Parches)
A partir de los 82,301 parches originales de la versión v12, se aplicó la tubería geométrica de sombreado con los 16 arquetipos de iluminación validados (cenitales, laterales, contraluz y luz de monitor con protección pupilar anatómica):
- **Periocular Izquierdo:** 459,612 parches en 6 clases.
- **Periocular Derecho:** 462,128 parches en 6 clases.
- **Boca:** 477,377 parches en 6 clases.
- **Total:** 1,399,117 parches guardados en `Data/dataset/ready/adfes_v12_17x/` con nomenclatura `{stem}.png` y `{stem}_ilu{1..16}.png`.

#### 2. Entrenamiento Canónico Multi-Etapa Decreciente (v14)
El entrenamiento se ejecutó en CPU utilizando precarga a memoria RAM (`preload_samples_to_cache` con 8 hilos, eliminando 5 millones de lecturas síncronas de disco):
- **Periocular:** 469,404 muestras balanceadas por mínimo común (78,234 por clase exacta en las 6 clases; 398,994 train / 70,410 val).
  - Etapa 1 (8 épocas, 57.1% muestras): Loss 0.1804 $\rightarrow$ Val Accuracy: 94.38%.
  - Etapa 2 (8 épocas, 28.6% muestras): Loss 0.1786 $\rightarrow$ Val Accuracy: 95.92%.
  - Etapa 3 (8 épocas, 14.3% muestras): Loss 0.1703 $\rightarrow$ Val Accuracy: **95.19%**.
  - ONNX exportado y validado en `cv2.dnn` (Opset 18, latencia CPU: 10.79 ms).
- **Boca:** 240,414 muestras balanceadas por mínimo común (40,069 por clase exacta en las 6 clases; 204,354 train / 36,060 val).
  - Etapa 1 (8 épocas, 57.1% muestras): Loss 0.0815 $\rightarrow$ Val Accuracy: 98.13%.
  - Etapa 2 (8 épocas, 28.6% muestras): Loss 0.0844 $\rightarrow$ Val Accuracy: 98.20%.
  - Etapa 3 (8 épocas, 14.3% muestras): Loss 0.0729 $\rightarrow$ Val Accuracy: **98.34%**.
  - ONNX exportado y validado en `cv2.dnn` (Opset 18, latencia CPU: 1.28 ms).

#### 3. Hallazgo Crítico y Normalización Adaptativa de Rango Dinámico
Al evaluar en tiempo real sobre cámaras web en entornos cotidianos, se descubrió la causa raíz de la persistencia del falso "ojo cerrado":
- **Brecha de Dominio Fotográfico:** ADFES fue capturado en estudio profesional bajo reflectores de alta potencia (rango dinámico [10, 254], media 167.9, esclerótica blanca >230). En contraste, las webcams de portátiles bajo iluminación doméstica producen imágenes con rango dinámico comprimido ([31, 186], media 121.3, desviación estándar 28.7 frente a 54.0).
- **Limitación de CLAHE:** `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` sobre bloques de $8 \times 8$ recorta la amplificación y no estira el histograma global, dejando la esclerótica en grises medios (120-140) que la CNN interpreta como párpado cerrado.
- **Solución Algorítmica en Inferencia:** Se incorporó estiramiento adaptativo min-max en `_infer_periocular` y `_infer_mouth` en `realtime.py`. En pruebas sobre 20 fotogramas reales del usuario, la detección correcta de ojos abiertos aumentó de 9/20 (45%) a **17/20 (85%)**, manteniendo un 100% de precisión (20/20) en ojos cerrados verdaderos.

#### 4. Validación en Vivo y Cierre Oficial de la Fase 1 (16 Octubre 2026)
Con la migración del entorno a Arch Linux (acceso directo a cámara mediante V4L2 sobre Intel Iris Xe sin capas de compresión de software):
- Se comprobó la inferencia de [v14](file:///home/sebadcc/Documentos/PROYECTOS/Ozmerion/Data/models/v14/metadata.json) en [realtime.py](file:///home/sebadcc/Documentos/PROYECTOS/Ozmerion/realtime.py) a ~10-12 ms totales en CPU.
- **Resultado:** Detección de ojos abiertos con confianza continua superior al **90%–95%+**, y cierre/parpadeo reactivo con alta fidelidad y sin falsos positivos.
- **Hito:** Se declara formalmente **FINALIZADA Y APROBADA LA FASE 1** de Ozmerion. El proyecto queda listo para abordar la **Fase 2 (Fusión Espacial Jerárquica y Grafo de Congruencia Anatómica)**.

---

## 3. Cambios en el Pensamiento y Paradigmas Estratégicos

### De Recortes Visuales a Complejos Neuromusculares Biológicos
Inicialmente se concebía la extracción facial como recortes geométricos genéricos. A partir del 22/09/2026 se consolidó el principio biológico: **los parches son unidades musculares funcionales acopladas**.
* No se puede aislar el párpado de la ceja porque el orbicular y el frontal co-determinan la textura de la piel.
* No se puede aislar la comisura de la mandíbula porque la apertura bucal deforma la orientación angular de los labios.
* Resolver la covarianza de forma local permite que redes ultra compactas (Tiny-CNNs) aprendan representaciones latentes precisas sin requerir visión holística pesada.

### De Redes de Caja Negra a Cinemática Física Explicable
Frente a la tendencia común de utilizar modelos recurrentes o 3D-CNNs para video, Ozmerion adoptó una postura inmutable:
* **El aprendizaje profundo termina en el frame estático.**
* La inferencia temporal de microexpresiones ($40\text{--}200\text{ ms}$) es una cuestión de **física, cinemática y estadística**: velocidad gestual ($\Delta x / \Delta t$), aceleración ($\Delta^2 x / \Delta t^2$), distancia a la línea base de reposo ($Z\text{-score}$) y máquinas de estados finitos explicables (Onset $\rightarrow$ Ápice $\rightarrow$ Offset).
* Esto garantiza ejecución en microsegundos sobre CPU y explicabilidad algorítmica total.

### El Tiempo como Desambiguador Supremo de la Morfología Estática
Una de las conclusiones teóricas más determinantes al planificar las Fases 2 y 3 es que **la imagen estática ($t$) por sí sola es morfológicamente ambigua**:
* Una apertura ocular extrema (AU5) y apertura mandibular (AU26) son geométricamente casi idénticas en **Sorpresa** y en **Pánico / Terror**.
* Las comisuras elevadas pueden representar una sonrisa genuina de Duchenne o una mueca social forzada.
* **El dominio temporal ($\Delta t$) resuelve la verdad:** La Sorpresa exhibe un tiempo de subida (*onset*) balístico ultra rápido (<100 ms) y una resolución inmediata al asimilar el estímulo; el Pánico o Terror presenta mesetas prolongadas con microoscilaciones de tensión muscular. El tiempo no es un mero accesorio: es la variable física que convierte una mueca morfológica en un estado psicológico interpretable.

### De Matrices Cartesianas Planas a Topologías Jerárquicas de Congruencia (Fase 2)
Al proyectar la incorporación de nuevos parches (Nasolabial AU9/10 para asco genuino, Mentoniano AU17 para contención/tensión y Glabela AU4 para ceño medial):
* Una matriz cartesiana plana multiplicaría combinatoriamente los estados ($6^N = \text{miles de estados}$), colapsando la interpretabilidad.
* El nuevo paradigma adopta una **topología jerárquica de congruencia anatómica / grafo bayesiano**: evalúa primero activaciones musculares locales continuas, verifica sinergias biológicas conocidas (ej. co-activación mejilla-párpado) y sintetiza el estado facial sin explosión combinatoria.

### La Genealogía `patcher` y la Soberanía Tecnológica con `patcher3` Propio (Fase 4)
La arquitectura consolidó el principio de **aislamiento geométrico total**:
$$\text{patch.py (FaceMesh)} \longrightarrow \text{patch2.py (Tasks 3D)} \longrightarrow \text{patcher3.py (Modelo Propio Ozmerion)}$$
* `patcher` es la **única frontera** que procesa la geometría cruda de la imagen. Ningún otro módulo (`builder`, `balancer`, `trainer`, `realtime`) conoce los detalles del sensor ni los puntos faciales.
* En la Fase 4, en lugar de adoptar detectores externos limitados, se entrenará un **modelo multi-tarea propio en CPU** (basado en MobileNetV4 o ShuffleNetV2) con un conjunto mínimo de 12–16 anclajes anatómicos PnP para resolver rostro, pose 3D y corte directo de parches en $< 4\text{ ms}$, otorgando independencia tecnológica absoluta.

