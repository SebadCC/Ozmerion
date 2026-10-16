# DIRECTRICES Y REGLAS MAESTRAS DEL PROYECTO OZMERION

> **IMPORTANTE PARA ASISTENTES IA (LEER ANTES DE RESPONDER):**
> Este documento define la filosofía, arquitectura y normas obligatorias de interacción para el proyecto **Ozmerion**. Lee estas directrices antes de cualquier análisis o intervención. Nunca violes la política de código bajo demanda ni propongas redes neuronales de caja negra para la capa temporal.

---

## 1. Introducción al Proyecto Ozmerion

**Ozmerion** es un sistema modular, ultra ligero y optimizado para **CPU comercial estándar**, diseñado para la detección, extracción y clasificación en tiempo real de **dinámicas faciales y microexpresiones espontáneas de alta velocidad (40–200 ms)**.

### Enfoque Biomecánico Fundamental
El sistema rechaza el procesamiento monolítico de rostros completos mediante redes neuronales pesadas. En su lugar, descompone la morfología facial en **complejos neuromusculares acoplados** mediante parches anatómicos locales de **$64 \times 64$ píxeles en escala de grises**:
1. **Complejo Periocular Izquierdo ($64 \times 64$):** Covarianza entre orbicular (apertura ocular) y corrugador/frontal (ceja descendida/elevada/neutra).
2. **Complejo Periocular Derecho ($64 \times 64$):** Mismo principio anatómico, evaluado de forma desacoplada para registrar asimetrías (duda, guiño).
3. **Complejo Bucal ($64 \times 64$):** Covarianza entre cigomáticos/depresores (comisuras arriba/abajo/neutra) y apertura mandibular (abierta/cerrada).
4. **Parche Nasal / Nasolabial Central ($64 \times 64$):** AU9/AU10 (arruga nasal y elevador del labio superior), marcador biológico canónico del asco genuino y la ira visceral.
5. **Parche Mentoniano / Barbilla ($64 \times 64$):** AU17 (mentalis / textura "piel de naranja"), marcador biológico supremo de la contención emocional reprimida, tristeza sofocada y duda/escepticismo.
6. **Parches de Mejilla / Surco Cigomático Bilateral ($64 \times 64$ L/R):** AU6/AU12/AU14 (abultamiento del pómulo y surco nasogeniano), confirmación biológica de sonrisa genuina de Duchenne vs fingida y desprecio unilateral.

---

## 2. Normas de Trabajo Obligatorias

### Regla 1: Política Estricta de Código Bajo Demanda
* **NO generar bloques de código ejecutable ni refactorizaciones completas a menos que el usuario lo solicite de forma explícita.**
* El rol por defecto del asistente es de **arquitecto de software, revisor crítico, analista de datos, documentador y asesor biomecánico**.
* Si se discute un cambio, explicar la lógica, ecuaciones o arquitectura primero; el código solo se emite cuando el usuario dé la orden expresa ("genera el código", "implementa esto", etc.).

### Regla 2: Cero Deep Learning en la Capa Temporal (Inferencia Explicable)
* Las redes neuronales (Tiny-CNNs) se detienen en la clasificación del fotograma estático ($t$).
* **La capa de decisión temporal (microexpresiones y dinámicas) NO utiliza redes neuronales de caja negra (ni LSTMs, ni GRUs, ni 3D-CNNs, ni Transformers).**
* La inferencia final se rige estrictamente por:
  * **Señales continuas:** La capa temporal se alimenta de vectores continuos de activación (probabilidades / logits) y no de índices discretizados prematuramente.
  * **Acondicionamiento y filtrado de señal:** Aplicar suavizado analítico de baja latencia (Savitzky-Golay o Kalman 1D) para eliminar el jitter estocástico del sensor antes de computar derivadas.
  * **Cinemática física:** Derivadas de velocidad y aceleración gestual ($\Delta x / \Delta t$, $\Delta^2 x / \Delta t^2$).
  * **Desambiguación Temporal:** Utilizar la firma temporal (rapidez de onset y persistencia de ápice) para resolver ambigüedades de la morfología estática (ej. sorpresa vs terror).
  * **Línea base neutra y $Z\text{-score}$:** Calibración estadística de la fisonomía de reposo del sujeto individual ($\mu, \sigma$).
  * **Máquinas de Estados Finitos (FSM):** Detección estructurada de fases: $\text{Línea Base} \rightarrow \text{Onset (40-200 ms)} \rightarrow \text{Ápice (pico)} \rightarrow \text{Offset (relajación)}$.
  * **Árboles de reglas explicables** fundamentados en la literatura FACS (Facial Action Coding System).

### Regla 3: Prioridad Absoluta a Inferencia en CPU (Baja Latencia a 30 FPS)
* El sistema completo (Fases 1, 2 y 3 integradas) debe ejecutarse a **$\ge 30\text{ FPS}$ en CPUs comerciales corrientes**, saturando el sensor de la cámara web sin requerir GPU dedicada.
* La latencia total del pipeline debe mantenerse en $\le 25\text{--}30\text{ ms}$ (en Fase 1 ya opera a **12–17 ms** totales gracias a grafos ONNX Opset 18 inferidos nativamente mediante `cv2.dnn`).

### Regla 4: Entrada Monocanal Estandarizada, Rango Dinámico y Parches Orgánicos
* Todo parche para clasificadores es estrictamente de dimensiones **$64 \times 64 \times 1$**, normalizado con ecualización de histograma adaptativa (**CLAHE**) y estiramiento de rango dinámico robusto (percentilar $P_2\text{--}P_{98}$) para blindar la detección ante cámaras domésticas con rango comprimido sin ser vulnerable a reflejos especulares.
* **Parches orgánicos:** No aplicar homografías agresivas, transformaciones afines extremas o desdoblamientos 3D que alteren la física y textura real del sensor.

### Regla 5: Desacoplamiento Modular y Soberanía Geométrica de `patcher`
Cada componente del repositorio cumple una única responsabilidad canónica:
* `patcher` (`patch.py` $\rightarrow$ `patch2.py` $\rightarrow$ futuro `patcher3.py`): **Es el único componente del sistema que interactúa con la geometría cruda de la imagen.** Realiza detección de rostro, filtrado de pose 3D (Pitch, Yaw, Roll) y recorte/estandarización de parches.
* `builder3.py`: Curaduría neuromuscular del dataset mediante sinergias FACS con bandas de incertidumbre. Consume directamente parches y telemetría de `patcher`.
* `balancer.py`: Nivelación determinista mediante espejo horizontal cruzado de ojos, podas regulares y síntesis estocástica con metadatos en chunks PNG `tEXt`.
* `trainer.py`: Entrenamiento agnóstico de Tiny-CNNs con cabezas multi-tarea sobre tensores $64 \times 64$.
* `realtime.py`: Bucle interactivo con escritor asíncrono desacoplado (`DiskWriterWorker`) para anotación y captura en caliente sin caídas de FPS.

### Regla 6: Cero Etiquetas Tóxicas (Zero Toxic Labels)
* Las muestras dudosas o situadas en bandas de solapamiento morfológico **no se asignan forzosamente a una clase activa**.
* Se canalizan siempre a carpetas de `incertidumbre/` para garantizar la pureza matemática de la distribución de entrenamiento.

### Regla 7: Independencia Tecnológica Mediante Modelo Propio (`patcher3`)
* MediaPipe Tasks (en `patch2.py`) es una solución transitoria de andamiaje geométrico.
* La meta de la Fase 4 es implementar `patcher3.py`, un **modelo propio multi-tarea de Ozmerion** optimizado en CPU (ej. ShuffleNetV2/MobileNetV4) que integre detección facial, estimación de ángulos 3D con un set reducido de 12–16 anclajes anatómicos PnP y extracción directa de parches en $< 4\text{ ms}$.

### Regla 8: Trazabilidad Completa (metadata.json)
* Toda etapa que genere o transforme datos debe registrar un archivo `metadata.json` con fecha, versión, parámetros biomecánicos, umbrales y conteo exhaustivo de muestras por clase.

### Regla 9: Ética, Privacidad y Habeas Data
* Cumplimiento riguroso de normativas de protección de datos personales. Rostros y muestras se tratan bajo principios estrictos de investigación y consentimiento informado.

---

## 3. Resumen Rápido de Fases Técnicas

* **Fase 1 (Culminada y Consolidada en v14):** Extracción geométrica de parches $64 \times 64$, curaduría FACS neuromuscular, balanceo multi-dataset 17x (1.4M parches) y clasificadores locales Tiny-CNN de alta precisión (>95% periocular, >98% boca) con latencia de 12–17 ms en CPU.
* **Fase 2 (Próxima etapa):** **Fusión Espacial Jerárquica y Cobertura Facial Estratégica (6-7 Parches Canónicos)**, combinando vectores continuos de activación bajo un grafo de congruencia anatómica (previniendo la explosión combinatoria $6^N$) y extendiendo la cobertura al parche nasolabial (AU9/10), mentoniano (AU17) y surco cigomático/mejillas bilaterales (AU6/12/14) para Duchenne y desprecio.
* **Fase 3 (Etapa final):** **Motor Cinemático y Desambiguación Temporal**, búfer circular FIFO a 30–33 FPS nativos, cálculo de derivadas cinemáticas filtradas ($\Delta x/\Delta t, \Delta^2 x/\Delta t^2$) y resolución temporal de ambigüedades morfológicas para microexpresiones de alta velocidad ($40\text{--}200\text{ ms}$).
* **Fase 4 (Soberanía tecnológica):** **Sustitución por `patcher3` propio**, eliminando MediaPipe y asumiendo internamente la detección, pose 3D y corte anatómico.

