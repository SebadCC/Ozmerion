# VISIÓN, ARQUITECTURA Y HOJA DE RUTA (OZMERION)

> **PROPÓSITO DEL DOCUMENTO:**
> Este archivo define la estructura de largo plazo, los principios no negociables y la hoja de ruta de fases del proyecto **Ozmerion**. Cualquier asistente IA o desarrollador que colabore en el proyecto debe consultar este mapa para mantener la coherencia y no desviar el enfoque.

---

## 1. La Visión Central (Norte Estratégico)

El objetivo de **Ozmerion** es construir un sistema de visión artificial en tiempo real, ligero y modular, capaz de detectar, clasificar y verificar la autenticidad de **microexpresiones faciales espontáneas de alta velocidad (40–200 ms)** operando exclusivamente sobre **CPU comercial estándar**, sin depender de hardware acelerador costoso ni de redes neuronales de caja negra en la toma de decisión final.

### El Principio Fundacional
> *"El aprendizaje profundo resuelve la visión en el instante; la cinemática física y la psicología de la expresión resuelven la verdad en el tiempo."*

Una vez que las redes neuronales locales extraen los estados de contracción muscular del fotograma estático, **la imagen visual deja de existir**. La inferencia temporal y emocional pasa a ser un problema puramente numérico, vectorial y cinemático, evaluable en microsegundos.

---

## 2. La Arquitectura en Tres Escalas

```
┌────────────────────────────────────────────────────────────────────────┐
│ ESCALA 1: ESPACIAL LOCAL (Frame t) - Inferencia Visual en CPU          │
│ Parches 64x64 CLAHE + MinMax ──> Tiny-CNNs Locales (ONNX en cv2.dnn)   │
│  ├─ Periocular Izquierdo: [Ceja (3) x Ojo (2)] ──> 6 Clases            │
│  ├─ Periocular Derecho:   [Ceja (3) x Ojo (2)] ──> 6 Clases            │
│  ├─ Complejo Bucal:       [Comisuras (3) x Apertura (2)] ──> 6 Clases  │
│  ├─ Nasolabial Central:   [Arruga Nasal (2) x Elevación Labial (2)]    │
│  ├─ Mentoniano / Barbilla:[Piel de Naranja / Tensión (2)] (AU17)       │
│  ├─ Mejilla / Cigomático: [Pómulo Duchenne x Hoyuelo (L/R)] (AU6/12/14)│
│  └─ Latencia Actual Fase 1: 10-12 ms en CPU (Meta Fase 2: <= 20 ms)    │
└────────────────────────────────────────────────────────────────────────┘
                                    │ (Vector continuo de probabilidades / logits)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ ESCALA 2: FUSIÓN ESPACIAL ESTÁTICA (Frame t) - Congruencia Anatómica    │
│ Grafo de Congruencia Biomecánica / Reglas FACS (Anti-Explosión 6^N)    │
│  ├─ Fusión continua de activaciones locales en vector facial unificado │
│  ├─ Evaluación de asimetría bilateral (desprecio unilateral, guiño)    │
│  ├─ Reglas de co-activación biológica (Duchenne: pómulo + párpado)     │
│  ├─ Reglas de contención (tristeza/ira reprimida: ceño + mentón AU17)  │
│  └─ Salida: Vector de Estado Facial Continuo (sin argmax prematuro)    │
└────────────────────────────────────────────────────────────────────────┘
                                    │ (Vectores continuos estabilizados)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ ESCALA 3: CINEMÁTICA TEMPORAL (Δt) - Desambiguación Psico-Física       │
│ Búfer Circular FIFO a Tasa Nativa (30-33 FPS, ventana 3-5s = 90-150 v) │
│  ├─ Suavizado analítico previo (Savitzky-Golay / Kalman 1D) anti-ruido │
│  ├─ Derivadas Físicas: Velocidad (Δx/Δt) y Aceleración (Δ²x/Δt²)       │
│  ├─ FSM Cinemática: Línea Base ──> Onset ──> Ápice ──> Offset          │
│  ├─ Desambiguación Temporal: Resuelve ambigüedad morfológica estática  │
│  └─ Filtro de Autenticidad: 40-200 ms (Genuina) vs >500 ms (Fingida)   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Hoja de Ruta de Fases Técnicas

### Fase 1: Extracción Canónica, Invarianza y Clasificadores Locales Robustos
* **Estado:** **FINALIZADA Y VALIDADA CON ÉXITO (Hito v14 consolidado y verificado en tiempo real sobre webcam).**
* **Logros Clave:**
  * Pipeline desacoplado con `patch2.py` (análisis angular 3D Euler Pitch, Yaw, Roll).
  * Curaduría neuromuscular FACS (`builder3.py`) con estricta política "Zero Toxic Labels" ($\pm 10\%$ de histeresis a `incertidumbre/`).
  * Balanceo determinista multi-dataset con cruce especular de ojos (`balancer.py`).
  * Expansión masiva 17x con 16 arquetipos físicos de iluminación Blender (1,399,117 parches).
  * Entrenamiento canónico decreciente v14: Exactitud Periocular **95.19%**, Exactitud Bucal **98.34%**.
  * Latencia de inferencia optimizada sobre CPU: **10 a 12 ms por frame completo** (saturando webcams comerciales y superando con creces los 70-80 ms de frameworks monolíticos).
  * Normalización adaptativa de rango dinámico Min-Max + CLAHE para salvar la brecha fotográfica entre estudios de iluminación y webcams domésticas.
  * Verificación en vivo en webcam bajo Linux (V4L2 + Intel Iris Xe): Detección estable de ojos abiertos con confianza >90–95% y respuesta precisa al parpadeo/cierre.

---

### Fase 2: Fusión Espacial Jerárquica y Expansión Multi-Parche Estratégica
* **Estado:** **Siguiente Etapa Inmediata (En planificación).**
* **Objetivo:** Unificar las lecturas locales de cada frame en un vector de estado facial anatómicamente coherente y expandir la cobertura a todo el rostro sin dispersar el cómputo en micro-parches innecesarios.
* **Principio de Cobertura Facial Estratégica (Por qué NO 10 parches):**
  * Fragmentar el rostro en 10 o 12 micro-parches colapsaría el presupuesto de CPU ($\le 25\text{--}30\text{ ms}$) y destruiría el contexto anatómico local (cada parche necesita suficiente área circundante para que la CNN distinga la contracción muscular relativa al tejido circundante).
  * En su lugar, el sistema adopta una **topología canónica de 6 a 7 parches estratégicos**, que cubre el 100% de los complejos neuromusculares esenciales del rostro humano bajo FACS:
    1. **Periocular Izquierdo ($64 \times 64$):** Ojo + Ceja izquierda (AU4 ceño, AU1/2 elevación, AU45 parpadeo/cierre).
    2. **Periocular Derecho ($64 \times 64$):** Mismo complejo evaluado de forma desacoplada para registrar asimetrías (duda, guiño, desprecio).
    3. **Complejo Bucal ($64 \times 64$):** Labios, apertura mandibular y comisuras (AU12 risa, AU15 tristeza, AU25/26 apertura).
    4. **Parche Nasal / Nasolabial Central ($64 \times 64$):** Dorso de la nariz y arruga nasolabial superior (AU9 `noseWrinkle` + AU10 `upperLipRaiser`). Marcador biológico inconfundible del **asco (disgust) genuino y la ira visceral**, distinguiéndolo de meras aperturas bucales.
    5. **Parche Mentoniano / Barbilla ($64 \times 64$):** Músculo mentalis (AU17 `chinRaiser`). Textura arrugada de la barbilla ("piel de naranja"). Marcador biológico supremo de la **contención emocional reprimida, tristeza sofocada y duda/escepticismo**.
    6. **Parche Mejilla / Surco Cigomático Izquierdo ($64 \times 64$):** Abultamiento del pómulo, pliegue nasogeniano y hoyuelo lateral (AU6 `cheekRaiser` + AU12 lateral + AU14 `dimpler`). Confirmación biológica de la **sonrisa genuina de Duchenne** (co-activación de mejilla con compresión palpebral inferior) frente a la sonrisa social fingida.
    7. **Parche Mejilla / Surco Cigomático Derecho ($64 \times 64$):** Mismo complejo evaluado de forma bilateral independiente para detectar asimetrías (desprecio unilateral, sonrisa burlona o tensión asimétrica).
* **Pilares de la Fusión Jerárquica:**
  1. **Prevención de la Explosión Combinatoria:** En lugar de una matriz cartesiana plana gigantesca ($6^N = \text{decenas de miles de combinaciones}$), la fusión opera como un **Grafo de Congruencia Anatómica / Árbol de Reglas Biomecánicas**:
     * *Sinergia Duchenne:* Párpado inferior comprimido (AU6/AU44) + Pómulo elevado (AU6) + Comisura elevada (AU12).
     * *Sinergia de Tristeza Contenida:* Ceño medial elevado (AU1) + Comisura descendida (AU15) + Mentón contraído en piel de naranja (AU17).
     * *Sinergia de Asco Genuino:* Nariz arrugada (AU9) + Labio superior elevado (AU10) + Entrecerramiento palpebral (AU7).
     * *Asimetría Unilateral:* Disociación de mejilla/comisura izquierda vs derecha (marcador canónico de desprecio AU14 o ironía).
  2. **Preservación de Señal Continua:** La capa de fusión no descarta información convirtiendo prematuramente a índices discretos (`argmax`); alimenta a la capa temporal con **vectores continuos de probabilidad / logits de activación**.

---

### Fase 3: Motor Cinemático y Desambiguación Temporal de Microexpresiones
* **Estado:** **Fase de Diseño y Especificación Matemática.**
* **Objetivo:** Detección y verificación de dinámicas espontáneas en el tiempo mediante física, cinemática y estadística explicable (CERO cajas negras).
* **El Principio de Desambiguación Temporal:**
  * *La imagen estática ($t$) es morfológicamente ambigua:* Unos ojos hiper-abiertos y boca abierta pueden representar tanto **Sorpresa** como **Pánico/Terror**.
  * *El dominio temporal ($\Delta t$) resuelve la verdad:* La **Sorpresa** exhibe un tiempo de subida (*onset*) balístico ultra rápido (<100 ms) con decaimiento inmediato tras procesar el estímulo cognitivo; el **Pánico** exhibe mesetas de alta tensión sostenida y oscilaciones de microtensión muscular.
* **Pilares de Implementación:**
  1. **Búfer Circular FIFO a Tasa Nativa:** Ventana temporal deslizante de 3 a 5 segundos procesada a **30–33 FPS** (90 a 150 vectores numéricos en RAM; costo computacional despreciable).
  2. **Acondicionamiento y Filtrado de Señal:** Filtrado temporal analítico de baja latencia (Savitzky-Golay de ventana corta o filtro Kalman 1D) para eliminar el jitter estocástico del sensor antes de derivar.
  3. **Derivadas Físicas:** Cálculo estable de velocidad gestual ($\Delta x / \Delta t$) y aceleración ($\Delta^2 x / \Delta t^2$).
  4. **FSM de Activación:**
     $$\text{Línea Base (Reposo)} \xrightarrow{\Delta x / \Delta t > \text{Umbral}} \text{Onset (40-200 ms)} \xrightarrow{\text{Pico}} \text{Ápice} \xrightarrow{\text{Retorno}} \text{Offset} \rightarrow \text{Reposo}$$
  5. **Presupuesto de Rendimiento:** Mantener el cómputo total de las Fases 1, 2 y 3 en $\le 25\text{--}30\text{ ms}$, garantizando $\ge 30\text{ FPS}$ en vivo en CPU.

---

### Fase 4: Desacoplamiento Total con `patcher3` (Modelo Propio Ozmerion)
* **Estado:** **Visión y Meta a Medio Plazo.**
* **Objetivo:** Lograr soberanía tecnológica absoluta reemplazando dependencias de Google/MediaPipe mediante un modelo propio multi-tarea ultraligero (`patcher3.py`).
* **Genealogía y Principio de Aislamiento:**
  $$\text{patch.py (FaceMesh 2D)} \longrightarrow \text{patch2.py (Landmarker 3D)} \longrightarrow \text{patcher3.py (Modelo Propio)}$$
  * `patcher` es el **único módulo del repositorio** que conoce y manipula la geometría cruda de la imagen. Los módulos consumidores (`builder`, `balancer`, `trainer`, `realtime`) permanecen 100% aislados y agnósticos.
* **Alcance de `patcher3`:**
  * Backbone ultra liviano sobre CPU (ej. ShuffleNetV2 o MobileNetV4-Small en ONNX, latencia $< 4\text{ ms}$).
  * Detección directa de rostro y estimación de pose 3D rígida (Euler Pitch, Yaw, Roll) mediante un conjunto reducido y canónico de 12 a 16 anclajes anatómicos con PnP.
  * Extracción y normalización directa de los parches útiles ($64 \times 64$) requeridos por el sistema.

---

## 4. Métricas de Éxito y Criterios No Negociables

1. **Latencia Total de Pipeline:** $\le 25\text{--}30\text{ ms}$ por cuadro completo en CPU estándar (saturando los $30\text{--}33\text{ FPS}$ de cámaras web comerciales).
2. **Explicabilidad Total:** Cada microexpresión detectada debe poder desglosarse en su curva física temporal ($\Delta x / \Delta t$), duración de fases y sus Action Units activas (CERO cajas negras recurrentes).
3. **Invarianza a Iluminación y Pose:** Tolerancia a giros de cabeza de hasta $\pm 20^\circ$ en Pitch/Yaw y resistencia probada ante sombras cenitales/orbitales y rango dinámico comprimido.
4. **Zero Toxic Labels:** Pureza matemática en las distribuciones de datos; la incertidumbre es una clase de control y auditoría.

