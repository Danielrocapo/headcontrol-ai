# HeadControl AI — versión 2 (seguimiento facial)

Versión experimental del proyecto destinada a controlar Windows con movimientos de cabeza y expresiones faciales, **sin Teachable Machine**. Se conserva la primera versión en la raíz del repositorio; esta versión está en la carpeta `v2/`.

## Funciones implementadas

- Seguimiento continuo de la cabeza con MediaPipe Face Landmarker.
- Calibración del centro mirando al frente (30 fotogramas).
- Movimiento suave por velocidad, con zona muerta, sensibilidad y velocidad configurables.
- Guiño del ojo izquierdo = clic izquierdo; guiño del ojo derecho = clic derecho (sensible a la cámara y a la persona).
- Mantener ambos ojos cerrados intencionalmente durante unos 0,5 segundos y abrirlos = clic izquierdo (un parpadeo normal no basta).
- Mantener la boca abierta aproximadamente 1 segundo = activar/desactivar modo scroll.
- Clic izquierdo por permanencia del cursor durante 1,5 segundos, **activado por defecto**. Se hace un solo clic hasta mover el cursor a otra posición.
- Indicadores en vivo de las probabilidades de ojo izquierdo, ojo derecho, apertura de boca, progreso de permanencia y última acción.
- Pausa global F8 y recalibración global F9 en Windows.
- Parada al perder el rostro y parada de emergencia de PyAutoGUI.

**Importante:** los gestos no estarán calibrados para todo el mundo. Prueba el prototipo sin documentos importantes y ajusta la sensibilidad. No constituye un dispositivo médico ni una tecnología asistiva validada.

## Requisitos

- Windows 10 u 11, cámara web y Python 3.11 o 3.12 recomendados.
- Conexión una vez para instalar dependencias y descargar el archivo oficial del modelo.
- Después, la inferencia y las acciones se ejecutan localmente.

## Instalación

Desde la raíz del repositorio, en PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r v2/requirements.txt
python v2/download_model.py
python v2/app.py
```

Si no tienes Python 3.12, utiliza `py -3.11` o el comando `python` correspondiente. Si PowerShell bloquea la activación, puedes ejecutar los comandos directamente con `.venv\Scripts\python.exe` sin activar el entorno.

## Guía de uso

1. Colócate frente a la cámara. Durante los primeros 30 fotogramas mantén una posición neutral.
2. Pulsa **F8** o el botón «Activar control».
3. Mueve la cabeza a derecha/izquierda o arriba/abajo para desplazar el cursor, y vuelve al centro para detenerlo.
4. Para asegurar un primer clic, coloca el cursor sobre un botón y **déjalo quieto 1,5 segundos**: verás el progreso de «Permanencia» hasta 100 %. Por defecto esta modalidad está activada.
5. También puedes mantener ambos ojos cerrados alrededor de 0,5 segundos y abrirlos para un clic izquierdo. Los guiños individuales hacen clic izquierdo/derecho si MediaPipe puede distinguirlos.
6. Si no funcionan los gestos, mira los valores de «Ojos» en la interfaz. Los valores cercanos a 1.0 indican mayor probabilidad de ojo cerrado. Si al guiñar un ojo ambos valores cambian a la vez, usa el clic por permanencia.
7. Para hacer scroll, mantén la boca abierta alrededor de 1 segundo; repite el gesto para regresar al modo cursor.
8. Si el centro quedó mal calibrado, pulsa **F9** y mira al frente.
9. Para pausar desde cualquier programa, pulsa **F8**.

El archivo `v2/assets/face_landmarker.task` no se incluye en el repositorio; `download_model.py` lo descarga desde el alojamiento oficial de MediaPipe. Su licencia y uso corresponden a sus respectivos titulares.

## Arquitectura

```text
Webcam (OpenCV)
    ↓
MediaPipe Face Landmarker
    ↓
478 puntos faciales + expresiones
    ↓
Calibración + filtro de suavizado + reconocimiento de gestos
    ↓
Python (PyAutoGUI)
    ↓
Cursor / clic / scroll de Windows
```

La interfaz se construye con PySide6. Esta versión no utiliza HTML, Flask ni TensorFlow.js.

## Comprobar la lógica

Desde la raíz del repositorio:

```powershell
python -m unittest discover -s v2/tests -v
```

Las pruebas automatizadas comprueban lógica matemática y gestos, **no** el funcionamiento con una webcam real. Para la entrega final se necesita comprobarlo en el equipo del estudiante, documentar los resultados y capturar un video de uso.

## Limitaciones conocidas

- La iluminación, el ángulo de la cámara y la anatomía facial pueden afectar precisión y gestos.
- Los umbrales de guiño y apertura de boca son valores iniciales, no mediciones validadas.
- El movimiento utiliza control relativo (tipo joystick), no seguimiento exacto de la mirada.
- La pausa global F8/F9 y la automatización de Windows se diseñaron para un equipo Windows; otras plataformas no están verificadas.

## Diagnóstico de clics

La interfaz muestra las señales `Ojos: izq. / der.` y `Permanencia`. Si las señales de ambos ojos nunca se acercan a 0,55 al cerrar los ojos, utiliza el modo por permanencia o ajusta los umbrales de detección en `engine.py` después de tomar mediciones. El movimiento del cursor puede funcionar perfectamente aunque el reconocimiento de gestos falle.

Comprueba siempre el clic en un botón inofensivo antes de intentar manejar otras aplicaciones. Si el clic por permanencia tampoco funciona, observa si el progreso llega a 100 %. Si llega a 100 % y no hace clic, puede tratarse de un problema de automatización de Windows o de permisos del programa de destino. PyAutoGUI no siempre puede controlar ventanas elevadas con privilegios de administrador desde un proceso sin esos privilegios.
