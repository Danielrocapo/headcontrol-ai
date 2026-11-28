package org.headcontrol.tracker;

import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.AccessibilityServiceInfo;
import android.accessibilityservice.GestureDescription;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.WindowInsets;
import android.view.WindowManager;
import android.view.accessibility.AccessibilityEvent;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

public final class PointerService extends AccessibilityService {
    static PointerService instance;
    private static final String TAG = "HeadControl";
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final DwellPolicy dwell = new DwellPolicy();
    private final DwellPolicy controlDwell = new DwellPolicy();
    private final EdgeGestures.Repeat repeat = new EdgeGestures.Repeat();
    private final EdgeGestures.Selection selection = new EdgeGestures.Selection();
    private final Object cancelControl = new Object();
    private final PointerContinuity continuity = new PointerContinuity();
    private final StablePointer stabilizer = new StablePointer();
    private PointerTrace motionTrace;
    private String traceMessage = "Registro desactivado";
    private boolean traceWriting;
    private final Runnable watchdog = this::suspendPointer;
    private WindowManager windows;
    private SharedPreferences controls;
    private EdgeGestures.ZoneLayout zones;
    private EdgeGestures.Zone highlighted;
    private Rect zoneScreen;
    private int overlayX, overlayY;
    private String dragFeedback = "";
    private long dragFeedbackUntil;
    private PointDrag pointDrag;
    private View overlay;
    private double x, y;
    private long last, lastDrawn, drawLatency = -1, displayAge = -1, draws, lastDrawAt;
    private float screenX, screenY, onsetX, onsetY;
    private float progress;
    private boolean tapPending;
    private boolean winkOnsetValid;
    private Drag drag;
    private String request = "—", result = "—";
    private long requestId;
    private long lastPublishedCapture;
    private boolean diagnosticPassed;

    @Override protected void onServiceConnected() {
        instance = this;
        diagnosticPassed = false;
        windows = getSystemService(WindowManager.class);
        controls = getSharedPreferences(EdgeGestures.SETTINGS, MODE_PRIVATE);
        Log.i(TAG, "accessibility_connected");
    }

    @SuppressWarnings("deprecation")
    private Rect usableScreen() {
        if (Build.VERSION.SDK_INT >= 30) {
            var metrics = windows.getCurrentWindowMetrics();
            Rect bounds = new Rect(metrics.getBounds());
            var bars = metrics.getWindowInsets().getInsetsIgnoringVisibility(
                    WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
            bounds.left += bars.left;
            bounds.top += bars.top;
            bounds.right -= bars.right;
            bounds.bottom -= bars.bottom;
            return bounds;
        }
        Rect screen = new Rect();
        overlay.getWindowVisibleDisplayFrame(screen);
        return screen;
    }

    private boolean ensureOverlay() {
        if (overlay != null) return true;
        float density = getResources().getDisplayMetrics().density;
        float radius = 14 * density;
        overlay = new View(this) {
            private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
            private final RectF arc = new RectF();

            @Override protected void onDraw(Canvas canvas) {
                long now = SystemClock.elapsedRealtime();
                var display = continuity.display(now);
                if (display == PointerContinuity.Display.HIDDEN) return;
                drawControls(canvas, paint, density, display, now);
                if (display != PointerContinuity.Display.FROZEN && last != lastDrawn) {
                    lastDrawn = last;
                    draws++;
                    lastDrawAt = now;
                    displayAge = now - last;
                    drawLatency = now - lastPublishedCapture;
                    if (motionTrace != null) motionTrace.drawn(last, now);
                }
                float cx = (float)x, cy = (float)y;
                paint.setStyle(Paint.Style.FILL);
                paint.setColor(0xff101722);
                canvas.drawCircle(cx, cy, radius, paint);
                paint.setColor(display == PointerContinuity.Display.FROZEN ? 0xffffb343 : 0xffffff55);
                canvas.drawCircle(cx, cy, radius - 3*density, paint);
                paint.setColor(display == PointerContinuity.Display.ACTIVE ? 0xff187433 : 0xff101722);
                canvas.drawCircle(cx, cy, 2.5f*density, paint);
                if (progress > 0 && (display == PointerContinuity.Display.ACTIVE
                        || display == PointerContinuity.Display.PREVIEW && highlighted == EdgeGestures.Zone.PAUSE)) {
                    float ring = radius + 3*density;
                    arc.set(cx-ring, cy-ring, cx+ring, cy+ring);
                    paint.setStyle(Paint.Style.STROKE);
                    paint.setStrokeWidth(3*density);
                    paint.setColor(0xffffffff);
                    canvas.drawArc(arc, -90, 360*progress, false, paint);
                }
            }
        };
        try {
            windows.addView(overlay, new WindowManager.LayoutParams(-1, -1,
                    WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE | WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE,
                    PixelFormat.TRANSLUCENT));
            return true;
        } catch (RuntimeException error) {
            overlay = null;
            Log.e(TAG, "overlay_add_failed", error);
            return false;
        }
    }

    boolean update(float[] point, StablePointer.Pose pose, long captured, boolean armed,
                   boolean dwellEnabled, boolean closed, boolean smooth) {
        long now = SystemClock.elapsedRealtime();
        if (!FrameTiming.live(captured, now) || !ensureOverlay()) return false;
        if (overlay.getWidth() < 2 || overlay.getHeight() < 2) return false;
        int[] origin = new int[2];
        overlay.getLocationOnScreen(origin);
        Rect safe = usableScreen();
        if (!safe.intersect(origin[0], origin[1], origin[0] + overlay.getWidth(), origin[1] + overlay.getHeight())
                || safe.width() < 2 || safe.height() < 2) return false;
        overlayX = origin[0];
        overlayY = origin[1];
        if (!safe.equals(zoneScreen)) {
            stabilizer.reset();
            zoneScreen = safe;
            zones = new EdgeGestures.ZoneLayout(safe.left, safe.top, safe.right, safe.bottom,
                    getResources().getDisplayMetrics().density);
        }
        double targetX = PointerPolicy.project(point[0], safe.left, safe.right) - origin[0];
        double targetY = PointerPolicy.project(point[1], safe.top, safe.bottom) - origin[1];
        var mode = StablePointer.Profile.at(controls.getInt(StablePointer.PREF, StablePointer.DEFAULT));
        var previousState = stabilizer.state;
        if (mode.adaptive()) {
            stabilizer.minimumCutoff = controls.getInt(OneEuro.CUTOFF, OneEuro.DEFAULT_CUTOFF)/10.0;
            stabilizer.beta = controls.getInt(OneEuro.BETA, OneEuro.DEFAULT_BETA)/100.0;
        }
        if (mode == StablePointer.Profile.OFF) {
            stabilizer.reset();
            x = PointerResponse.position(x, targetX, captured - last, smooth && last != 0);
            y = PointerResponse.position(y, targetY, captured - last, smooth && last != 0);
        } else {
            stabilizer.update(captured, pose, targetX, targetY, x, y,
                    getResources().getDisplayMetrics().density, mode, smooth,
                    point[0] <= 0 || point[0] >= 1, point[1] <= 0 || point[1] >= 1);
            x = stabilizer.x;
            y = stabilizer.y;
        }
        screenX = (float)x + origin[0];
        screenY = (float)y + origin[1];
        last = captured;
        continuity.update(captured, armed);
        if (drag != null && !drag.finishing && !drag.releaseRequested) {
            drag.x = screenX;
            drag.y = screenY;
        }
        updateControls(now, armed, dwellEnabled, closed);
        lastPublishedCapture = SystemClock.elapsedRealtime();
        if (motionTrace != null) motionTrace.add(new PointerTrace.Frame(captured, lastPublishedCapture, mode,
                smooth, previousState, stabilizer.state, pose, point[0], point[1], screenX, screenY));
        overlay.removeCallbacks(watchdog);
        overlay.postDelayed(watchdog, Math.max(0, PointerPolicy.FRESH_MS - (lastPublishedCapture - captured)));
        overlay.postInvalidateOnAnimation();
        return true;
    }

    boolean tracingMotion() { return motionTrace != null; }

    String motionTraceStatus() {
        return motionTrace == null ? traceMessage : "Registro activo · " + motionTrace.size() + " muestras";
    }

    void toggleMotionTrace() {
        if (traceWriting) return;
        if (motionTrace == null) motionTrace = new PointerTrace();
        else saveMotionTrace();
    }

    private void saveMotionTrace() {
        if (motionTrace == null) return;
        var rows = motionTrace.snapshot();
        motionTrace = null;
        traceWriting = true;
        traceMessage = "Guardando registro…";
        File path = new File(getFilesDir(), "pointer_trace.csv");
        new Thread(() -> {
            String message;
            try (var writer = Files.newBufferedWriter(path.toPath(), StandardCharsets.UTF_8)) {
                PointerTrace.write(rows, writer);
                message = "Registro guardado · " + rows.size() + " filas · pointer_trace.csv";
            } catch (IOException error) {
                message = "No se pudo guardar pointer_trace.csv";
                Log.e(TAG, "pointer_trace_write_failed", error);
            }
            String completed = message;
            handler.post(() -> {
                traceMessage = completed;
                traceWriting = false;
            });
        }, "HeadControlPointerTrace").start();
    }

    private void drawControls(Canvas canvas, Paint paint, float density, PointerContinuity.Display display, long now) {
        if (zones == null) return;
        canvas.save();
        canvas.translate(-overlayX, -overlayY);
        paint.setStyle(Paint.Style.FILL);
        paint.setTextAlign(Paint.Align.CENTER);
        for (EdgeGestures.Zone zone : EdgeGestures.Zone.values()) {
            float[] b = zones.bounds[zone.ordinal()];
            paint.setColor(zone == highlighted ? 0xe0255c69 : 0x8a101722);
            canvas.drawRoundRect(b[0], b[1], b[2], b[3], 9*density, 9*density, paint);
            paint.setColor(0xffffffff);
            paint.setTextSize((zone.arrow() ? 23 : 11) * density);
            String label = zone == EdgeGestures.Zone.DRAG && selectingDrag() ? "Cancelar"
                    : zone == EdgeGestures.Zone.PAUSE && display == PointerContinuity.Display.PREVIEW ? "Reanudar" : zone.label;
            canvas.drawText(label, (b[0]+b[2])/2, (b[1]+b[3]-paint.ascent()-paint.descent())/2, paint);
        }
        String hint = selection.phase == EdgeGestures.Selection.Phase.ORIGIN ? "Fija el origen"
                : selection.phase == EdgeGestures.Selection.Phase.DESTINATION ? "Fija el destino"
                : now < dragFeedbackUntil ? dragFeedback : "";
        if (!hint.isEmpty()) {
            float[] b = zones.bounds[EdgeGestures.Zone.DOWN.ordinal()];
            float[] row = zones.bounds[EdgeGestures.Zone.DRAG.ordinal()];
            paint.setTextSize(12*density);
            float cx = (b[0]+b[2])/2, base = row[1] - 9*density;
            float w = paint.measureText(hint)/2 + 7*density;
            paint.setColor(0xcd101722);
            canvas.drawRoundRect(cx-w, base-15*density, cx+w, base+4*density, 6*density, 6*density, paint);
            paint.setColor(0xffffffff);
            canvas.drawText(hint, cx, base, paint);
        }
        if (selection.phase == EdgeGestures.Selection.Phase.DESTINATION) {
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(2*density);
            paint.setColor(0xff73cbff);
            canvas.drawCircle(selection.x, selection.y, 8*density, paint);
        }
        canvas.restore();
    }

    private void updateControls(long now, boolean armed, boolean dwellEnabled, boolean closed) {
        EdgeGestures.Zone zone = zones.hit(screenX, screenY);
        float density = getResources().getDisplayMetrics().density;
        long hold = controls.getInt(EdgeGestures.TAP, EdgeGestures.DEFAULT_TAP);
        boolean busy = tapPending || drag != null || pointDrag != null;
        boolean allowed = armed && !closed && continuity.canAct(now);
        boolean isArrow = zone != null && zone.arrow();
        highlighted = allowed && (!isArrow || !selectingDrag()) || !closed && zone == EdgeGestures.Zone.PAUSE ? zone : null;
        EdgeGestures.Zone arrow = allowed && !selectingDrag() && isArrow ? zone : null;
        if (repeat.update(arrow, now, busy, controls.getInt(EdgeGestures.ARROW, EdgeGestures.DEFAULT_ARROW),
                controls.getInt(EdgeGestures.REPEAT, EdgeGestures.DEFAULT_REPEAT))) swipe(arrow);

        boolean controlAvailable = zone == EdgeGestures.Zone.PAUSE && !closed
                || allowed && zone != null && !isArrow && (!busy || zone == EdgeGestures.Zone.DRAG);
        Object control = controlAvailable ? zone == EdgeGestures.Zone.DRAG && selectingDrag() ? cancelControl : zone : null;
        if (controlDwell.update(control, screenX, screenY, now, 17*density, hold))
            activateZone(zone);

        Object target = allowed && dwellEnabled && !busy && zone == null
                ? selectingDrag() ? selection.phase : dwell : null;
        if (dwell.update(target, screenX, screenY, now, 18*density, hold)) {
            if (selectingDrag()) {
                float[] endpoints = selection.confirm(screenX, screenY);
                if (endpoints != null) startPointDrag(endpoints);
            } else {
                tap(screenX, screenY, "Toque por permanencia");
            }
        }
        progress = isArrow ? repeat.progress : zone != null ? controlDwell.progress : dwell.progress;
    }

    private boolean selectingDrag() {
        return selection.phase != EdgeGestures.Selection.Phase.IDLE || pointDrag != null;
    }

    private void activateZone(EdgeGestures.Zone zone) {
        switch (zone) {
            case DRAG:
                if (selectingDrag()) cancelPointDrag();
                else selection.start();
                dwell.reset();
                break;
            case BACK:
            case HOME:
                cancelPointDrag();
                boolean back = zone == EdgeGestures.Zone.BACK;
                boolean done = performGlobalAction(back ? GLOBAL_ACTION_BACK : GLOBAL_ACTION_HOME);
                requestId++;
                request = zone.label;
                result = done ? "Android aceptó la navegación" : "Android rechazó la navegación";
                Log.i(TAG, "global_action " + request + " accepted=" + done);
                break;
            case PAUSE:
                TrackingService session = TrackingService.instance;
                if (session != null && !session.activating()) session.toggle();
                break;
            default: break;
        }
    }

    private void swipe(EdgeGestures.Zone zone) {
        if (!continuity.canAct(SystemClock.elapsedRealtime()) || tapPending || drag != null || pointDrag != null) return;
        float[] ends = EdgeGestures.swipe(zone, zoneScreen.left, zoneScreen.top, zoneScreen.right, zoneScreen.bottom);
        Path path = at(ends[0], ends[1]);
        path.lineTo(ends[2], ends[3]);
        tapPending = dispatch(new GestureDescription.StrokeDescription(path, 0, 280),
                "Desplazamiento " + zone.label, completed -> tapPending = false);
    }

    private static final class PointDrag {
        final float[] ends;
        long holdRemaining = ViewConfiguration.getLongPressTimeout() + 90;
        GestureDescription.StrokeDescription previous;
        boolean cancelled;

        PointDrag(float[] ends) { this.ends = ends; }
    }

    private void startPointDrag(float[] ends) {
        if (!continuity.canAct(SystemClock.elapsedRealtime()) || tapPending || drag != null || pointDrag != null) return;
        pointDrag = new PointDrag(ends);
        showDragFeedback("Arrastrando…");
        pointDragStep(pointDrag);
    }

    private void pointDragStep(PointDrag current) {
        if (pointDrag != current) return;
        boolean first = current.previous == null;
        boolean cancel = current.cancelled || !continuity.canAct(SystemClock.elapsedRealtime());
        if (first && cancel) {
            finishPointDrag("Arrastre cancelado");
            return;
        }
        boolean finish = cancel || current.holdRemaining <= 0;
        Path path = at(current.ends[0], current.ends[1]);
        if (finish && !cancel) path.lineTo(current.ends[2], current.ends[3]);
        long duration = finish ? cancel ? 1 : 350 : Math.min(65, current.holdRemaining);
        try {
            GestureDescription.StrokeDescription stroke = first
                    ? new GestureDescription.StrokeDescription(path, 0, duration, true)
                    : current.previous.continueStroke(path, 0, duration, !finish);
            current.previous = stroke;
            if (!finish) current.holdRemaining -= duration;
            String name = first ? "Arrastre: pulsar" : !finish ? "Continuar arrastre"
                    : cancel ? "Arrastre: liberar por cancelación" : "Arrastre: deslizar y soltar";
            boolean accepted = dispatch(stroke, name, completed -> {
                if (pointDrag != current) return;
                if (!completed) finishPointDrag("Android canceló el arrastre");
                else if (finish) finishPointDrag(cancel ? "Arrastre cancelado" : "Arrastre completado");
                else pointDragStep(current);
            });
            if (!accepted) finishPointDrag("Error al enviar arrastre");
        } catch (IllegalArgumentException | IllegalStateException error) {
            Log.w(TAG, "point_drag_error", error);
            finishPointDrag("Error de continuación");
        }
    }

    private void showDragFeedback(String text) {
        dragFeedback = text;
        dragFeedbackUntil = SystemClock.elapsedRealtime() + 3000;
        if (overlay != null) overlay.postInvalidateOnAnimation();
    }

    private void cancelPointDrag() {
        if (selection.phase != EdgeGestures.Selection.Phase.IDLE) {
            selection.cancel();
            showDragFeedback("Arrastre cancelado");
        }
        if (pointDrag != null) pointDrag.cancelled = true;
    }

    private void finishPointDrag(String result) {
        pointDrag = null;
        showDragFeedback(result);
        Log.i(TAG, "point_drag_result " + result);
    }

    String timing() {
        String fixation = stabilizer.state == StablePointer.State.ANCHORED ? "anclado"
                : stabilizer.state == StablePointer.State.RELEASING ? "liberando"
                : stabilizer.state == StablePointer.State.SETTLING ? "estabilizando" : "en movimiento";
        return "Puntero " + continuity.display(SystemClock.elapsedRealtime()) + " · " + fixation + " · dibujo: " + draws
                + " @ " + lastDrawAt + " ms · actualización a dibujo "
                + drawLatency + " ms · edad al dibujar " + displayAge + " ms";
    }

    String gestureStatus() {
        return request.startsWith("TOQUE DE PRUEBA") ? "Último envío: prueba A"
                : "Solicitud: " + request + " · Android: " + result;
    }

    boolean canPerformGestures() {
        AccessibilityServiceInfo info = getServiceInfo();
        return instance == this && info != null
                && (info.getCapabilities() & AccessibilityServiceInfo.CAPABILITY_CAN_PERFORM_GESTURES) != 0;
    }

    boolean diagnosticPassed() { return diagnosticPassed; }

    void acceptDiagnostic(GestureProbe probe) {
        if (probe.passed()) diagnosticPassed = true;
    }

    void testTap(float x, float y, GestureProbe probe) {
        diagnosticPassed = false;
        if (!canPerformGestures()) {
            probe.error("Servicio desconectado o sin capacidad de gestos");
        } else if (drag != null || tapPending) {
            probe.error("Hay otro gesto en curso. Reintenta cuando termine.");
        } else {
            dispatchTap(x, y, "TOQUE DE PRUEBA", probe);
        }
    }

    void wink(WinkPolicy.Event event) {
        if (event == WinkPolicy.Event.BEGIN || event == WinkPolicy.Event.TAP || event == WinkPolicy.Event.HOLD) {
            TrackingService session = TrackingService.instance;
            if (!diagnosticPassed || session == null || !session.wink.previewConfirmed() || session.wink.preview()) {
                blocked("Guiño " + event, "Falta confirmar las pruebas A y B");
                return;
            }
            if (!continuity.canAct(SystemClock.elapsedRealtime())) {
                blocked("Guiño " + event, "Puntero pausado o predicción caducada");
                return;
            }
        }
        switch (event) {
            case BEGIN:
                onsetX = screenX;
                onsetY = screenY;
                winkOnsetValid = true;
                break;
            case TAP:
                if (winkOnsetValid) tap(onsetX, onsetY, "Toque por guiño");
                else blocked("Toque por guiño", "Inicio cancelado o sin posición reciente");
                winkOnsetValid = false;
                break;
            case HOLD:
                if (winkOnsetValid) beginDrag();
                else blocked("Pulsación larga", "Inicio cancelado o sin posición reciente");
                break;
            case RELEASE:
                winkOnsetValid = false;
                if (drag != null) drag.releaseRequested = true;
                break;
            case CANCEL:
                cancelWink();
                break;
            default:
                break;
        }
    }

    private static Path at(float x, float y) {
        Path path = new Path();
        path.moveTo(x, y);
        return path;
    }

    private void tap(float x, float y, String type) {
        if (!continuity.canAct(SystemClock.elapsedRealtime())) return;
        if (drag != null || pointDrag != null || tapPending) {
            blocked(type);
            return;
        }
        dispatchTap(x, y, type, null);
    }

    private void dispatchTap(float x, float y, String type, GestureProbe probe) {
        String label = type + " (" + Math.round(x) + ", " + Math.round(y) + ")";
        tapPending = dispatch(new GestureDescription.StrokeDescription(at(x, y), 0, 50),
                label, completed -> {
                    tapPending = false;
                    if (probe != null) {
                        probe.complete(completed);
                        acceptDiagnostic(probe);
                    }
                }, probe, x, y);
    }

    private void blocked(String type) {
        blocked(type, "Hay otro gesto en curso");
    }

    private void blocked(String type, String reason) {
        request = type;
        requestId++;
        result = "Omitido: " + reason;
        Log.w(TAG, "gesture_blocked " + type + " cause=" + reason);
    }

    private boolean dispatch(GestureDescription.StrokeDescription stroke, String type,
                             java.util.function.Consumer<Boolean> callback) {
        return dispatch(stroke, type, callback, null, 0, 0);
    }

    private boolean dispatch(GestureDescription.StrokeDescription stroke, String type,
                             java.util.function.Consumer<Boolean> callback, GestureProbe probe, float x, float y) {
        boolean segment = "Continuar arrastre".equals(type);
        long id = requestId;
        if (!segment) {
            id = ++requestId;
            request = type;
            result = "Solicitado";
        }
        final long activeRequest = id;
        try {
            if (probe != null) probe.send(x, y);
            boolean accepted = dispatchGesture(new GestureDescription.Builder().addStroke(stroke).build(),
                    new GestureResultCallback() {
                @Override public void onCompleted(GestureDescription gesture) {
                    if (!segment) {
                        if (activeRequest == requestId) result = "Completado";
                        Log.i(TAG, "gesture_completed " + type);
                    }
                    callback.accept(true);
                }
                @Override public void onCancelled(GestureDescription gesture) {
                    if (activeRequest == requestId) result = "Cancelado: " + type;
                    Log.w(TAG, "gesture_cancelled " + type);
                    callback.accept(false);
                }
            }, handler);
            if (probe != null) {
                probe.accepted(accepted);
                if (!accepted) probe.error("Android rechazó dispatchGesture");
            }
            if (!accepted) {
                result = "Rechazado: " + type;
                Log.w(TAG, "gesture_rejected " + type);
            } else if (!segment) {
                result = "Aceptado · esperando callback";
                Log.i(TAG, "gesture_accepted " + type);
            }
            return accepted;
        } catch (RuntimeException error) {
            if (probe != null) probe.error(error.getClass().getSimpleName());
            result = error.getClass().getSimpleName() + " al solicitar " + type;
            Log.e(TAG, "gesture_failed " + type, error);
            return false;
        }
    }

    private static final class Drag {
        GestureDescription.StrokeDescription previous;
        final long nativeHoldUntil = SystemClock.elapsedRealtime() + ViewConfiguration.getLongPressTimeout() + 80;
        float lastX, lastY, x, y;
        boolean finishing, releaseRequested;
        Drag(float x, float y) {
            lastX = this.x = x;
            lastY = this.y = y;
        }
    }

    private void beginDrag() {
        if (drag != null || pointDrag != null || tapPending) {
            blocked("Pulsación larga");
            return;
        }
        drag = new Drag(onsetX, onsetY);
        dragSegment(drag);
    }

    private void dragSegment(Drag current) {
        if (drag != current) return;
        long now = SystemClock.elapsedRealtime();
        boolean first = current.previous == null;
        var step = DragPolicy.step(now, current.nativeHoldUntil, 0,
                current.finishing || !continuity.canAct(now), current.releaseRequested);
        boolean finishing = step == DragPolicy.Step.END;
        if (first && finishing) {
            drag = null;
            return;
        }
        float nx = step == DragPolicy.Step.MOVE ? current.x : current.lastX;
        float ny = step == DragPolicy.Step.MOVE ? current.y : current.lastY;
        Path path = at(current.lastX, current.lastY);
        if (nx != current.lastX || ny != current.lastY) path.lineTo(nx, ny);
        try {
            var stroke = first ? new GestureDescription.StrokeDescription(path, 0, 55, true)
                    : current.previous.continueStroke(path, 0, finishing ? 1 : 55, !finishing);
            current.previous = stroke;
            current.lastX = nx;
            current.lastY = ny;
            boolean accepted = dispatch(stroke, first ? "Pulsación larga" : finishing ? "Soltar" : "Continuar arrastre",
                    completed -> {
                        if (drag != current) return;
                        if (!completed || finishing) drag = null;
                        else dragSegment(current);
                    });
            if (!accepted) drag = null;
        } catch (IllegalArgumentException | IllegalStateException error) {
            result = "Continuación inválida";
            Log.e(TAG, "drag_continuation_failed", error);
            drag = null;
        }
    }

    void cancelWink() {
        winkOnsetValid = false;
        if (drag != null) drag.finishing = true;
    }

    void resetInteraction() {
        continuity.pauseActions();
        cancelInteraction();
    }

    private void cancelInteraction() {
        cancelWink();
        cancelPointDrag();
        repeat.reset();
        controlDwell.reset();
        highlighted = null;
        dwell.reset();
        progress = 0;
        if (overlay != null) overlay.postInvalidateOnAnimation();
    }

    void suspendPointer() {
        continuity.freeze();
        stabilizer.suspend();
        cancelInteraction();
    }

    void hide(String cause) {
        saveMotionTrace();
        suspendPointer();
        continuity.hide();
        last = 0;
        zones = null;
        zoneScreen = null;
        if (overlay != null) {
            overlay.removeCallbacks(watchdog);
            try { windows.removeViewImmediate(overlay); }
            catch (RuntimeException error) { Log.w(TAG, "overlay_remove_failed", error); }
            overlay = null;
        }
        Log.i(TAG, "overlay_removed cause=" + cause);
    }

    private void disconnect(String cause) {
        if (instance == this) instance = null;
        if (TrackingService.instance != null) TrackingService.instance.stop(cause);
        hide(cause);
    }

    @Override public void onAccessibilityEvent(AccessibilityEvent event) {}
    @Override public void onInterrupt() {
        if (TrackingService.instance != null) TrackingService.instance.stop("accessibility_interrupted");
        else hide("accessibility_interrupted");
    }
    @Override public boolean onUnbind(Intent intent) {
        disconnect("accessibility_unbound");
        return super.onUnbind(intent);
    }
    @Override public void onDestroy() {
        disconnect("accessibility_destroyed");
        super.onDestroy();
    }
}
