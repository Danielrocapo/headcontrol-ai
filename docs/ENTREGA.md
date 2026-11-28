# Auditoría y demostración

La entrega combina el worktree de presentación de Windows con Android 0.5.0-demo, basado en la versión de consenso 0.4.8 probada por el usuario. El historial público se conserva desde `v2-mediapipe`; el prototipo anterior y sus documentos permanecen en el historial y en el respaldo público. El tag local estable de Windows sigue en `87d9153`.

Se conservaron las fuentes de ejecución, tres ONNX seleccionables en Windows, el normalizador común y el recurso local de MediaPipe. Android usa el ONNX original. Se conservaron los modelos alternativos de Windows porque el selector y la comprobación de compatibilidad los usan. La rama de entrega prescinde de APK históricos, modelos de entrenamiento, copias de prueba, sesiones CSV, registros, perfiles personales, cachés y entornos. Los worktrees de origen y sus artefactos de respaldo se conservaron.

## Checklist de tres minutos

1. **0:00–1:00 · Windows.** Abre el ejecutable, inicia cámara, calibra y activa el cursor. Usa permanencia en un botón pequeño.
2. **1:00–2:00 · Windows.** Abre el teclado, escribe una palabra, prueba scroll y arrastre en una ventana de ensayo y pausa con F8.
3. **2:00–3:00 · Android.** Calibra, activa el control, abre una aplicación segura, usa permanencia y una flecha, y termina con Detener cámara en la notificación.

Las pruebas automatizadas del ejecutable usan imágenes sintéticas y prueban los recursos locales. El APK se valida por compilación, firma, contenido e integridad de recursos. La prueba física de los paquetes de entrega con cámara y los gestos globales del teléfono debe realizarse antes de declararlos aprobados para una publicación estable.
