# HeadControl AI

HeadControl AI es un prototipo de interacción humano-computadora que permite controlar el cursor mediante movimientos de la cabeza.

El proyecto utiliza un modelo de **Teachable Machine Pose** entrenado con cinco clases y una aplicación local que interpreta las predicciones para ejecutar movimientos del mouse en Windows.

## Clases del modelo

- `CENTRO`
- `IZQUIERDA`
- `DERECHA`
- `ARRIBA`
- `ABAJO`

## Funcionamiento

```text
Webcam
  ↓
Teachable Machine Pose
  ↓
Clase + nivel de confianza
  ↓
Validación de la predicción
  ↓
Python
  ↓
PyAutoGUI
  ↓
Movimiento del cursor
```

Para reducir movimientos accidentales, el prototipo utiliza un umbral de confianza y exige varias predicciones consecutivas antes de ejecutar una acción.

## Tecnologías

- Teachable Machine Pose
- TensorFlow.js
- Python
- Flask
- PySide6
- PyAutoGUI
- HTML / JavaScript

## Estructura

```text
headcontrol-ai/
├── app.py
├── requirements.txt
├── model/
│   ├── model.json
│   ├── metadata.json
│   └── weights.bin
├── static/
│   ├── index.html
│   ├── tf.min.js
│   └── teachablemachine-pose.min.js
└── docs/
    └── images/
```

## Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/Danielrocapo/headcontrol-ai.git
cd headcontrol-ai
```

2. Instalar las dependencias de Python:

```bash
pip install -r requirements.txt
```

3. Ejecutar la aplicación:

```bash
python app.py
```

## Modelo

El modelo fue entrenado en Teachable Machine Pose con ejemplos capturados mediante webcam. Los archivos exportados se almacenan localmente en la carpeta `model/`.

La aplicación está planteada para funcionar de forma local; el servidor en `127.0.0.1` se utiliza únicamente como comunicación interna dentro del equipo.

## Estado del proyecto

Versión inicial funcional: reconocimiento de cinco posturas de cabeza y control direccional del cursor.

### Mejoras futuras

- Movimiento continuo y más suave del cursor.
- Calibración personalizada.
- Clic y doble clic mediante gestos.
- Scroll y arrastre.
- Evaluación con más usuarios y diferentes condiciones de iluminación.

## Uso académico

Proyecto desarrollado con fines académicos para la asignatura de Inteligencia Artificial.
