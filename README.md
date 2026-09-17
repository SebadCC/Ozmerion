# Project Ozmerion

Sistema modular y ligero de visión por computador para el análisis y clasificación en tiempo real de microexpresiones faciales (región periocular y bucal).

El pipeline opera completamente en **CPU** optimizado con **MediaPipe (Face Mesh)** para la alineación geométrica y redes **Tiny-CNN** exportadas a **ONNX** ejecutadas a través de **OpenCV DNN**, alcanzando latencias de inferencia de ~2-3 ms (~30-35+ FPS).

---

## Instalacion
Clonar el repositorio localmente.  
Descargar y elegir como entorno Python 3.10 (mediapipe y torch no funcionan correctamente con versiones superiores de Python).  
Descargar las dependencias (pip install opencv-python mediapipe torch torchvision onnx numpy).  

## Uso
Ubicar las imagenes de muestra dentro de la carpeta /Data/muestras/ejemplo1/.  
Ejecutar builder.py.  
Ejecutar trainer.py.  
Verificar la version del modelo .onnx que arrojo trainer.py.  
Colocar el numero de la version del modelo .onnxx en la linea 14 de realtime.py.  
Ejecutar realtime.py.  

## Retrainer
Si desea que la clasificacion deje de hacerla mediapipe, puede usar un modelo y con base en el clasificar un nuevo dataset.  
Esto se usa para Pseudo-labeling, a fin de filtrar muestras masivas y evitar los sesgos de mediapipe, para ello debe:  

Colocar las imagenes de muestra dentro de la carpeta /Data/muestras/ejemplo1/.  
Colocar el numero de la version del modelo .onnxx con la cual clasificar las imagenes en la linea 11 de realtime.py.  
Verificar o modificar el umbral de aceptacion de un parche en la linea 12 de retrainer.py.  
Habilitar o deshabilitar la visualizacion del clasificador, si desea una mayor velocidad dejar en False.  
Ejecutar retrainer.py.  
Tomar las carpetas dentro de Data/Redataset/ y ubicarlas en Data/dataset/.  
Ejecutar trainer.py.  
Verificar la version del modelo .onnx que arrojo trainer.py.  
Colocar el numero de la version del modelo .onnxx en la linea 14 de realtime.py.  
Ejecutar realtime.py.  

## Realtime Dataset
Si desea tomar nuevas imagenes de muestra en tiempo real, puede hacerlo usando realtime.py, generando datasets personalizados manualmente.  
Esto se realiza con el fin de corregir deficiencias o sesgos en datasets, complementandolos para la creacion de un mejor modelo, para ello debe:  

Verificar el numero de la version del modelo .onnxx en la linea 14 de realtime.py.  
Ejecutar realtime.py.  
Seleccionar las etiquetas manuales para las capturas a tomar, usando el codigo de la seccion inferior.  
Preparar el entorno visual de la camara, acorde a las etiquetas seleccionadas.  
Presionar la tecla C para realizar una captura, o la tecla Espacio para tomar una rafaga de imagenes.  
Tomar las carpetas generadas dentro de Data/Rdataset, y moverlas hacia Data/dataset/ (mezclandolas con el dataset previo).  
Ejecutar trainer.py.  
Verificar la version del modelo .onnx que arrojo trainer.py.  
Colocar el numero de la version del modelo .onnxx en la linea 14 de realtime.py.  
Ejecutar realtime.py .  


Codigo de realtime.py para Transmision.  
  - Tags Ojos      : [1] Abierto  | [2] Cerrado
  - Tags Cejas     : [3] Neutra   | [4] Arriba   | [5] Abajo
  - Tags Boca      : [6] Abierta  | [7] Cerrada
  - Tags Comisuras : [8] Neutra   | [9] Arriba   | [0] Abajo

Despues de haber seleccionado correctamente las 4 etiquetas, el sistema mostara el nombre de las etiquetas seleccionadas en amarillo.  

## Estructura del directiorio Data/
Debido a que el directorio se genera a medida que se ejecutan ciertos scripts, esta la guia visual para su generacion manual.  

Data/  
├── muestras/                    # Fotos crudas de entrada por voluntario/lote  
│   ├── voluntario_1/  
│   │   ├── 001.jpg  
│   │   └── ...  
│   └── ...
├── dataset/                     # Parches 64x64 generados por builder.py  
│   ├── periocular_izq/  
│   ├── periocular_der/  
│   └── boca/
├── Rdataset/                    # Parches recolectados con etiquetas manuales en vivo  
│   ├── periocular_izq/  
│   ├── periocular_der/  
│   └── boca/  
├── Redataset/                   # Parches autoetiquetados mediante retrainer.py  
│   ├── periocular_izq/  
│   ├── periocular_der/  
│   └── boca/  
├── transmisiones/               # Capturas crudas tomadas desde realtime.py  
│   ├── capturas/  
│   └── rafagas/  
└── versiones/                   # Modelos congelados y pesos por versión  
    ├── v1/  
    │   ├── dataset/  
    │   └── onnx/  
    │       ├── model_periocular.onnx  
    │       └── model_boca.onnx  
    └── ...  

## Configuraciones adicionales

Todos los archivos tienen las variables de configuracion al inicio del mismo, despues de las importaciones, a fin de modificar con libertad, ya sea para pruebas, mejoras o personalizacion de los modelos y datasets generados  

