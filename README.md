# HeadControl AI

HeadControl AI controla el puntero con movimientos de cabeza mediante una cámara frontal. La inferencia se ejecuta localmente. Windows y Android comparten nueve características faciales y el contrato de coordenadas absolutas, con modelos Tracker D de distintas revisiones.

## Usar los binarios

Descarga los paquetes en [GitHub Releases](https://github.com/Danielrocapo/headcontrol-ai/releases) cuando estén publicados. Verifica los archivos con el SHA-256 del paquete de entrega.

- **Windows x64.** Extrae toda la carpeta portátil y abre `HeadControlAI.exe`. El directorio `_internal` permanece junto al ejecutable. Inicia la cámara, mira al frente para calibrar y activa el cursor. La permanencia es optativa. F8 o Esc pausa el control.
- **Android ARM64 (Android 8.0 o posterior).** Instala el APK mediante `adb install -r HeadControlAI-Android-arm64.apk` o desde Android. Habilita HeadControl AI en Accesibilidad y concede cámara y notificaciones. Calibra el neutral y activa el control. Puedes seleccionar las flechas, Arrastrar, Atrás, Inicio y Pausa por permanencia. La notificación permite detener la cámara.

Una instalación anterior con el mismo identificador debe tener una firma compatible. Los paquetes de demostración usan una firma de depuración. La compilación requiere descargar dependencias si la máquina no las tiene en caché. El uso de los binarios es local.

## Ejecutar y compilar desde un clon

Windows requiere Python 3.12 x64 y una cámara compatible con OpenCV. Desde PowerShell, en la raíz:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-build-windows.txt
.\.venv\Scripts\python.exe -X utf8 -B packaging\windows_entry.py --self-test
.\.venv\Scripts\python.exe -X utf8 -B packaging\windows_entry.py
.\.venv\Scripts\python.exe -X utf8 -B packaging\build_windows.py ..\headcontrol-artifacts
```

Android requiere JDK 17 o superior, Android SDK con plataforma API 37 y herramientas de compilación compatibles. Desde PowerShell:

```powershell
cd product/platform/android
.\gradlew.bat :app:assembleDebug
```

El APK queda en `app/build/outputs/apk/debug/app-debug.apk`. La tarea de recursos verifica los hashes del ONNX, su normalizador y Face Landmarker. El envoltorio de Gradle fija la versión de Gradle y el proyecto fija las dependencias de Android.

## Arquitectura y alcance

1. MediaPipe Face Landmarker obtiene puntos faciales y pose. El usuario establece una referencia neutral para cada sesión.
2. Nueve entradas relativas alimentan ONNX Runtime CPU. Windows inicia con Tracker D guiado v2. Android empaqueta Tracker D original.
3. El cursor absoluto utiliza las coordenadas X/Y y la fijación por consenso como configuración inicial. Cada aplicación aplica las protecciones de su plataforma.
4. Windows usa Python, OpenCV, PySide6 y las API de Windows. Android usa Java, Camera2, un servicio de cámara en primer plano y AccessibilityService para la superposición y los gestos.

El código del MVP de Windows está en `product/mvp/` y la app Android en `product/platform/android/`. Algunas dependencias de extracción, calibración y ejecución mantienen sus rutas históricas bajo `product/training/`, `product/experiments/` y `prueba_ojos/` para conservar el contrato y la procedencia. Los pesos finales y el normalizador están versionados; las capturas, perfiles personales, checkpoints y binarios de distribución quedan fuera del código.

[Auditoría, pruebas y checklist de demostración](docs/ENTREGA.md).
