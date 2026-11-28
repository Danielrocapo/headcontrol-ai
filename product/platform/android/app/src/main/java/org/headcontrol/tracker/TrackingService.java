package org.headcontrol.tracker;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.KeyguardManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.hardware.display.DisplayManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.Looper;
import android.os.PowerManager;
import android.os.SystemClock;
import android.util.Log;
import android.view.Display;

import org.opencv.android.OpenCVLoader;

import static org.headcontrol.tracker.PipelineTrace.Stage.*;


public final class TrackingService extends Service {
    static final String STOP = "org.headcontrol.tracker.STOP_TRACKING";
    static final String TOGGLE = "org.headcontrol.tracker.TOGGLE_CONTROL";
    static TrackingService instance;
    static String status = "Cámara detenida";
    static String stoppedDiagnostics = "Sin sesión";
    private static final long PAUSE_AFTER_GAP_MS = 750;
    final WinkPolicy wink = new WinkPolicy();
    private final PointerPolicy gate = new PointerPolicy();
    private final PipelineTrace trace = new PipelineTrace();
    private final ActivationRequest activation = new ActivationRequest();
    private final Handler ui = new Handler(Looper.getMainLooper());
    private HandlerThread thread;
    private Handler worker;
    private volatile CameraPreview sink;
    private Tracker tracker;
    private TrackingImage image;
    private double[] baseline;
    private NeutralCalibration neutral;
    private volatile NeutralCalibration.Feedback neutralFeedback;
    private long lastTimestamp, lastNeutralLog;
    private long startupDeadline;
    private volatile long lastFrame;
    private long lastUsable, lastDelivered;
    private long lastControlFrameId;
    private int width, height, rotation, displayRotation;
    private volatile int generation;
    private boolean armed, winkEnabled, smooth, closing;
    private int selectedEye;
    private float[] lastEyes;
    private boolean ocularRead;
    private String performance = "Esperando mediciones de rendimiento";
    private int samples, eyeSamples, winkSamples;
    private String eyeRate = "Ojos: esperando mediciones";
    private long statsStart;
    private final FrameTiming.Latest<Reading> latest = new FrameTiming.Latest<>();

    private static final class Reading {
        final int token;
        final float[] point, eyes;
        final StablePointer.Pose pose;
        final long captured, acquired, finished, mediaPipeMs, onnxMs;
        final long workerWaitMs, preparationMs;
        final boolean sensorClock;
        final String label, rejection;

        Reading(int token, float[] point, float[] eyes, long captured, long acquired, long finished,
                long mediaPipeMs, long onnxMs, long workerWaitMs, long preparationMs,
                boolean sensorClock, String label, String rejection, StablePointer.Pose pose) {
            this.token = token;
            this.point = point;
            this.eyes = eyes;
            this.captured = captured;
            this.acquired = acquired;
            this.finished = finished;
            this.mediaPipeMs = mediaPipeMs;
            this.onnxMs = onnxMs;
            this.workerWaitMs = workerWaitMs;
            this.preparationMs = preparationMs;
            this.sensorClock = sensorClock;
            this.label = label;
            this.rejection = rejection;
            this.pose = pose;
        }
    }

    private final Runnable watchdog = new Runnable() {
        @Override public void run() {
            if (closing) return;
            long now = SystemClock.elapsedRealtime();
            if (PointerService.instance == null) stop("accessibility_disconnected");
            else if (displayRotation != currentRotation()) stop("display_rotation_changed");
            else if (!screenAvailable()) stop("screen_locked_or_off");
            else if (lastFrame == 0 ? now > startupDeadline : now - lastFrame > 3000)
                stop("camera_frames_timeout");
            else {
                expireActivation(now);
                if (armed && !FrameTiming.live(lastUsable, now)) {
                    PointerService.instance.suspendPointer();
                    if (now - lastDelivered >= PAUSE_AFTER_GAP_MS) {
                        gate.reset();
                        pause("frame_stale");
                        status = "Seguimiento degradado · control pausado";
                    } else {
                        status = "Seguimiento degradado · puntero inmóvil, sin acciones";
                    }
                }
                ui.postDelayed(this, 100);
            }
        }
    };

    @Override public int onStartCommand(Intent intent, int flags, int id) {
        if (intent != null && STOP.equals(intent.getAction())) {
            stop("notification_stop");
            return START_NOT_STICKY;
        }
        if (intent != null && TOGGLE.equals(intent.getAction())) {
            if (instance == this && !closing) toggle();
            else stopSelf();
            return START_NOT_STICKY;
        }
        if (thread != null || closing) return START_NOT_STICKY;
        try {
            showNotification();
        } catch (RuntimeException error) {
            Log.e("HeadControl", "camera_foreground_denied", error);
            stop("camera_foreground_denied");
            return START_NOT_STICKY;
        }
        instance = this;
        if (PointerService.instance == null) {
            stop("accessibility_required");
            return START_NOT_STICKY;
        }
        displayRotation = currentRotation();
        startupDeadline = SystemClock.elapsedRealtime() + 15000;
        status = "Mantén la cabeza en posición neutral";
        thread = new HandlerThread("HeadControlCamera");
        thread.start();
        worker = new Handler(thread.getLooper());
        int token = ++generation;
        worker.post(() -> open(token));
        ui.post(watchdog);
        return START_NOT_STICKY;
    }

    private void showNotification() {
        String channel = "headcontrol_camera";
        getSystemService(NotificationManager.class).createNotificationChannel(
                new NotificationChannel(channel, "Seguimiento de cabeza", NotificationManager.IMPORTANCE_LOW));
        Notification notification = notification();
        if (Build.VERSION.SDK_INT >= 29) startForeground(1, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA);
        else startForeground(1, notification);
    }

    private Notification notification() {
        String channel = "headcontrol_camera";
        int flags = PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE;
        PendingIntent stop = PendingIntent.getService(this, 1, new Intent(this, TrackingService.class).setAction(STOP), flags);
        PendingIntent show = PendingIntent.getActivity(this, 2, new Intent(this, MainActivity.class), flags);
        PendingIntent toggle = PendingIntent.getService(this, 3, new Intent(this, TrackingService.class).setAction(TOGGLE), flags);
        return new Notification.Builder(this, channel)
                .setSmallIcon(android.R.drawable.ic_menu_camera).setContentTitle("HeadControl AI")
                .setContentText("La cámara frontal está activa para el seguimiento")
                .setContentIntent(show).setOngoing(true)
                .addAction(new Notification.Action.Builder(null, armed ? "Pausar" : "Activar", toggle).build())
                .addAction(new Notification.Action.Builder(null, "Detener cámara", stop).build()).build();
    }

    private int currentRotation() {
        Display display = getSystemService(DisplayManager.class).getDisplay(Display.DEFAULT_DISPLAY);
        return display == null ? -1 : display.getRotation();
    }

    private boolean screenAvailable() {
        return getSystemService(PowerManager.class).isInteractive()
                && !getSystemService(KeyguardManager.class).isKeyguardLocked();
    }

    private void open(int token) {
        if (token != generation) return;
        try {
            if (!OpenCVLoader.initLocal()) throw new IllegalStateException("opencv_init");
            tracker = new Tracker(this);
            if (token != generation) return;
            sink = new CameraPreview(this);
            width = sink.width;
            height = sink.height;
            rotation = PointerPolicy.rotation(sink.orientation, displayRotation * 90);
            image = new TrackingImage(width, height, rotation);
            sink.start(worker, (data, frameId, captured, acquired, packed, sensorClock) -> {
                if (token != generation) return;
                lastFrame = acquired;
                if (neutral == null) resetNeutral(acquired);
                if (baseline == null && !neutral.active()) return;
                if (baseline != null) trace.frame(captured, acquired, sensorClock);
                if (data == null) {
                    if (baseline == null) collectNeutral(frameId, captured, null, NeutralCalibration.Rejection.STALE);
                    else publish(new Reading(token, null, null, captured, acquired, acquired, -1, -1, -1, -1, sensorClock, "",
                            captured > acquired ? "Reloj de cámara adelantado" : "Predicción antigua · fotograma atrasado al adquirir", null));
                    return;
                }
                if (baseline != null && frameId <= lastControlFrameId) {
                    publish(new Reading(token, null, null, captured, acquired, acquired, -1, -1, -1, -1, sensorClock, "",
                            "Fotograma repetido o fuera de orden", null));
                    return;
                }
                lastControlFrameId = Math.max(lastControlFrameId, frameId);
                read(data, frameId, token, captured, acquired, packed, sensorClock);
            }, error -> fail(token, error));
            Log.i("HeadControl", "camera_started " + width + "x" + height + " rotation=" + rotation);
        } catch (Exception | LinkageError error) {
            fail(token, error);
        }
    }

    private void read(byte[] data, long frameId, int token, long captured, long acquired, long packed,
                      boolean sensorClock) throws Exception {
        long workStart = SystemClock.elapsedRealtime();
        if (baseline != null && !FrameTiming.usable(captured, acquired, workStart)) {
            publish(new Reading(token, null, null, captured, acquired, workStart, -1, -1,
                    workStart - packed, -1, sensorClock, "", "Predicción antigua · espera del worker", null));
            return;
        }
        var pixels = image.prepare(data);
        long now = Math.max(lastTimestamp + 1, captured);
        lastTimestamp = now;
        long mediaPipeStart = SystemClock.elapsedRealtime();
        if (baseline != null) trace.mark(MEDIAPIPE_INICIO, mediaPipeStart);
        double[] measured = tracker.measure(pixels, TrackingImage.WIDTH, TrackingImage.HEIGHT, now);
        long mediaPipeEnd = SystemClock.elapsedRealtime();
        if (baseline == null) {
            // Pose belongs to the acquired frame even if MediaPipe finished after the cursor deadline.
            collectNeutral(frameId, captured, measured, tracker.geometryStatus);
            return;
        }
        trace.mark(MEDIAPIPE_FIN, mediaPipeEnd);
        float[] point = null;
        String rejection = null;
        long onnxMs = -1;
        long finished = mediaPipeEnd;
        if (measured != null) {
            trace.mark(RAW_VALIDO, mediaPipeEnd);
            if (!FrameTiming.usable(captured, acquired, mediaPipeEnd)) {
                rejection = "Predicción antigua · presupuesto agotado tras MediaPipe";
            } else {
                float[] input = tracker.normalize(measured, baseline);
                long onnxStart = SystemClock.elapsedRealtime();
                trace.mark(NORMALIZADOR, onnxStart);
                trace.mark(ONNX_INICIO, onnxStart);
                try {
                    point = tracker.infer(input);
                } catch (Exception | LinkageError error) {
                    trace.reject("Error ONNX: " + error.getClass().getSimpleName(), SystemClock.elapsedRealtime());
                    throw error;
                } finally {
                    finished = SystemClock.elapsedRealtime();
                    trace.mark(ONNX_FIN, finished);
                }
                onnxMs = finished - onnxStart;
                if (point == null) rejection = "ONNX devolvió coordenadas no finitas";
                else {
                    trace.prediction(point, captured, finished);
                    if (!FrameTiming.usable(captured, acquired, finished))
                        rejection = "Predicción antigua · presupuesto agotado tras ONNX";
                }
            }
        } else {
            trace.reject(tracker.geometryStatus.message, mediaPipeEnd);
        }
        publish(new Reading(token, point, tracker.eyes, captured, acquired, finished,
                mediaPipeEnd - mediaPipeStart, onnxMs, workStart - packed, mediaPipeStart - workStart,
                sensorClock, tracker.lastMeasurement, rejection,
                point == null || rejection != null ? null : StablePointer.Pose.from(measured, baseline[9])));
    }

    private void publish(Reading reading) {
        if (reading.rejection != null) trace.reject(reading.rejection, reading.finished);
        trace.mark(BUZON, reading.finished);
        if (latest.offer(reading)) ui.post(() -> receive(latest.take()));
        else trace.mark(REEMPLAZADO, reading.finished);
    }

    private void collectNeutral(long frameId, long capture, double[] raw, NeutralCalibration.Rejection reason) {
        long now = SystemClock.elapsedRealtime();
        neutral.record(frameId, capture, now, raw, reason);
        baseline = neutral.baseline();
        neutralFeedback = neutral.feedback();
        if (baseline != null) {
            int token = generation;
            ui.post(() -> {
                if (token == generation && !closing) status = "Esperando rostro para la primera predicción";
            });
        }
        if (!neutral.active() || now - lastNeutralLog >= 2000) {
            Log.i("HeadControl", neutralFeedback.state + " " + neutralFeedback.detail.replace('\n', ' '));
            lastNeutralLog = now;
        }
    }

    private void resetNeutral(long now) {
        neutral = new NeutralCalibration(now);
        neutralFeedback = neutral.feedback();
        lastNeutralLog = now;
    }

    NeutralCalibration.Feedback neutralFeedback() { return neutralFeedback; }
    boolean cameraActive() { return lastFrame != 0 && SystemClock.elapsedRealtime() - lastFrame < 3000; }

    void retryNeutral() {
        if (worker == null || !cameraActive() || closing) return;
        int token = generation;
        worker.post(() -> {
            if (token == generation && baseline == null) resetNeutral(SystemClock.elapsedRealtime());
        });
    }

    private void receive(Reading reading) {
        PointerService pointer = PointerService.instance;
        if (reading.token != generation || pointer == null) return;
        long now = SystemClock.elapsedRealtime();
        trace.mark(SERVICIO, now);
        if (!screenAvailable() || displayRotation != currentRotation()) {
            stop("screen_unavailable_or_rotated");
            return;
        }
        if (reading.mediaPipeMs >= 0) {
            ocularRead = true;
            lastEyes = reading.eyes;
            if (reading.eyes != null) eyeSamples++;
        }
        if (statsStart == 0) statsStart = now;
        samples++;
        if (now - statsStart >= 2000) {
            performance = String.format(java.util.Locale.ROOT,
                    "%.1f FPS de procesamiento · edad %d ms (%s) · entrada %d ms · espera %d ms · preparar %d ms · MediaPipe %d ms · ONNX %d ms · UI %d ms",
                    samples * 1000.0 / (now - statsStart), now - reading.captured,
                    reading.sensorClock ? "sensor" : "desde adquisición", reading.acquired - reading.captured,
                    reading.workerWaitMs, reading.preparationMs, reading.mediaPipeMs, reading.onnxMs, now - reading.finished);
            eyeRate = String.format(java.util.Locale.ROOT, "Ojos %.1f Hz · detector %.1f Hz · guiños %s",
                    eyeSamples * 1000.0 / (now - statsStart), winkSamples * 1000.0 / (now - statsStart),
                    winkEnabled ? "habilitados" : "deshabilitados");
            Log.i("HeadControl", performance + " · " + sink.stats() + " · " + eyeRate + " · " + pointer.timing());
            statsStart = now;
            samples = eyeSamples = winkSamples = 0;
        }
        if (reading.rejection != null || !FrameTiming.usable(reading.captured, reading.acquired, now)) {
            status = reading.rejection == null ? "Predicción antigua · entrega tardía al servicio" : reading.rejection;
            if (reading.rejection == null) trace.reject(status, now);
            pointer.suspendPointer();
            wink.reset();
            gate.reset();
            return;
        }
        PointerPolicy.Sample state = gate.observe(reading.captured, now, reading.point != null);
        if (state != PointerPolicy.Sample.VALID) {
            pointer.suspendPointer();
            wink.reset();
            if (armed && state != PointerPolicy.Sample.HOLD) pause(reading.label + "_" + state);
            status = "face_missing".equals(reading.label) ? "Esperando rostro"
                    : "Pose no válida · " + reading.label;
            return;
        }
        wink.sample(reading.eyes, reading.captured);
        if (pointer.update(reading.point, reading.pose, reading.captured, armed, !winkEnabled,
                wink.eyesClosed(reading.eyes), smooth)) {
            lastUsable = reading.captured;
            lastDelivered = now;
            trace.mark(PUNTERO, SystemClock.elapsedRealtime());
            tryActivate(now);
            if (armed && winkEnabled) {
                if (reading.eyes != null) winkSamples++;
                WinkPolicy.Event event = wink.update(reading.eyes, reading.captured);
                if (event == WinkPolicy.Event.BEGIN || event == WinkPolicy.Event.TAP || event == WinkPolicy.Event.HOLD
                        || event == WinkPolicy.Event.RELEASE) Log.i("HeadControl", "wink_event=" + event + " " + wink.gestureTiming());
                pointer.wink(event);
            }
        } else {
            pointer.cancelWink();
            wink.reset();
            status = "Predicción válida · esperando la superposición del puntero";
            trace.reject(status, now);
            return;
        }
        status = armed ? "Control activo · RAW" : "Predicción reciente · puntero en vista previa";
    }

    void toggle() {
        if (armed || activation.pending()) {
            pause("user_pause");
            status = "Control pausado";
        } else if (neutralFeedback != null && neutralFeedback.state == NeutralCalibration.State.READY
                && PointerService.instance != null && screenAvailable() && displayRotation == currentRotation()
                && !wink.calibrating()) {
            long now = SystemClock.elapsedRealtime();
            activation.request(now);
            Log.i("HeadControl", "activation_requested");
            tryActivate(now);
        }
    }

    private void tryActivate(long now) {
        expireActivation(now);
        if (!activation.consume(now, lastUsable, gate)) return;
        armed = true;
        wink.reset();
        if (PointerService.instance != null) PointerService.instance.resetInteraction();
        status = "Control activo · RAW";
        Log.i("HeadControl", "control_armed");
        refreshNotification();
    }

    private void expireActivation(long now) {
        if (!activation.expire(now)) return;
        status = "Activación vencida · " + status;
        trace.reject("Activación sin predicción reciente y estable", now);
        Log.i("HeadControl", "activation_expired");
    }

    boolean armed() { return armed; }
    boolean activating() { return activation.pending(); }

    String trackingStatus() {
        String current = lastUsable == 0 && trace.runningInference() ? "Ejecutando inferencia ONNX" : status;
        return (activation.pending() ? "Activación solicitada · " : "") + current;
    }
    boolean winkEnabled() { return winkEnabled; }
    int selectedEye() { return selectedEye; }
    boolean smooth() { return smooth; }

    void setSmooth(boolean value) { smooth = value; }

    String diagnostics() {
        PointerService pointer = PointerService.instance;
        String eyeData = !ocularRead ? "Ojos: sin fotogramas MediaPipe en seguimiento"
                : lastEyes == null ? "Ojos: MediaPipe no devolvió ambos canales"
                : !wink.ready() ? "Ojos: canales presentes · falta calibración ocular" : "Ojos: canales presentes y calibrados";
        return (sink == null ? "" : sink.stats() + "\n") + performance + "\n" + trace.summary()
                + "\n\n" + eyeRate + "\n" + eyeData + "\n" + wink.diagnostic(lastEyes, winkEnabled, armed)
                + (pointer == null ? "" : "\n" + pointer.timing() + "\n" + pointer.gestureStatus());
    }

    void setWinks(boolean enabled, int eye) {
        boolean effective = enabled && wink.ready();
        if (effective == winkEnabled && eye == selectedEye) return;
        selectedEye = eye;
        winkEnabled = effective;
        wink.select(eye);
        if (PointerService.instance != null) PointerService.instance.resetInteraction();
    }

    private void refreshNotification() {
        if (!closing) getSystemService(NotificationManager.class).notify(1, notification());
    }

    void calibrateWinks(int stage) {
        pause("eye_calibration");
        winkEnabled = false;
        wink.calibrate(stage, SystemClock.elapsedRealtime());
    }

    private void pause(String cause) {
        boolean changed = armed || activation.pending();
        activation.cancel();
        armed = false;
        if (changed) {
            Log.i("HeadControl", "control_paused cause=" + cause);
            refreshNotification();
        }
        wink.reset();
        if (PointerService.instance != null) PointerService.instance.resetInteraction();
    }

    private void fail(int token, Throwable error) {
        Log.e("HeadControl", "camera_pipeline_failed", error);
        ui.post(() -> { if (token == generation) stop("Error de cámara o inferencia: " + error.getClass().getSimpleName()); });
    }

    void stop(String cause) {
        if (closing) return;
        closing = true;
        generation++;
        status = "Cámara detenida · " + cause;
        stoppedDiagnostics = trace.summary();
        Log.i("HeadControl", "tracking_stopped cause=" + cause);
        ui.removeCallbacks(watchdog);
        pause(cause);
        wink.cancelCalibration();
        if (PointerService.instance != null) PointerService.instance.hide(cause);
        if (instance == this) instance = null;
        if (worker != null) worker.post(() -> {
            release();
            thread.quitSafely();
            ui.post(() -> {
                stopForeground(STOP_FOREGROUND_REMOVE);
                stopSelf();
            });
        }); else {
            stopForeground(STOP_FOREGROUND_REMOVE);
            stopSelf();
        }
    }

    private void release() {
        if (sink != null) sink.close();
        if (tracker != null) {
            try { tracker.close(); } catch (Exception error) { Log.w("HeadControl", "tracker_close", error); }
        }
        if (image != null) image.close();
    }

    @Override public IBinder onBind(Intent intent) { return null; }
    @Override public void onDestroy() {
        stop("service_destroyed");
        super.onDestroy();
    }
}
