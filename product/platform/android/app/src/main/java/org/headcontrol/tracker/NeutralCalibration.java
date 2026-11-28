package org.headcontrol.tracker;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.Locale;

final class NeutralCalibration {
    static final int GOAL = 12;
    static final long TIMEOUT_MS = 15_000;
    private static final long MIN_SPAN_MS = 1_000, WINDOW_MS = 5_000, MAX_FACE_GAP_MS = 1_200;
    private static final double ANGLE_LIMIT = 5.0, CENTER_LIMIT = .06, WIDTH_LIMIT = .15;

    enum State { WAITING, COLLECTING, UNSTABLE, READY, TIMEOUT }
    enum Face { UNKNOWN, MISSING, DETECTED }
    enum Rejection {
        NONE(""), DUPLICATE("Fotograma repetido o fuera de orden"), NO_TIMESTAMP("Sin marca de tiempo de cámara"),
        STALE("Fotograma ya atrasado al adquirirlo"), NO_FACE("MediaPipe no detectó rostro"),
        NO_POSE("MediaPipe no entregó pose facial"), MATRIX("Matriz de pose inválida"),
        LANDMARKS("Puntos faciales incompletos"), EYE_SPAN("Separación ocular insuficiente o inválida"),
        GEOMETRY("Geometría de pose inválida"), NONFINITE("Características faciales no finitas");
        final String message;
        Rejection(String message) { this.message = message; }
    }

    static final class Feedback {
        final State state;
        final Face face;
        final int progress;
        final String message, discarded, detail;

        Feedback(State state, Face face, int progress, String message, String discarded, String detail) {
            this.state = state;
            this.face = face;
            this.progress = progress;
            this.message = message;
            this.discarded = discarded;
            this.detail = detail;
        }
    }

    private static final class Sample {
        final long time;
        final double[] values;
        Sample(long time, double[] values) { this.time = time; this.values = values.clone(); }
    }

    private final ArrayDeque<Sample> window = new ArrayDeque<>();
    private final EnumMap<Rejection, Integer> counts = new EnumMap<>(Rejection.class);
    private final long start;
    private long previousId, span, elapsed;
    private int received, accepted, afterCursorDeadline;
    private Face face = Face.UNKNOWN;
    private State state = State.WAITING;
    private String message = "Esperando fotogramas para medir el neutral.";
    private String motion = "";
    private Rejection lastRejected = Rejection.NONE;
    private double[] spread, baseline;

    NeutralCalibration(long start) { this.start = start; }
    boolean active() { return state != State.READY && state != State.TIMEOUT; }
    double[] baseline() { return baseline; }

    void record(long id, long capture, long now, double[] raw, Rejection rejection) {
        if (!active()) return;
        received++;
        elapsed = now - start;
        if (id <= 0) rejection = Rejection.NO_TIMESTAMP;
        else if (id <= previousId) rejection = Rejection.DUPLICATE;
        else previousId = id;

        if (rejection == Rejection.NO_FACE) face = Face.MISSING;
        else if (rejection != Rejection.STALE && rejection != Rejection.DUPLICATE
                && rejection != Rejection.NO_TIMESTAMP) face = Face.DETECTED;
        if (rejection == Rejection.NONE && !valid(raw)) rejection = Rejection.NONFINITE;
        trim(capture);
        if (rejection != Rejection.NONE) {
            counts.merge(rejection, 1, Integer::sum);
            lastRejected = rejection;
            message = rejection.message + ".";
        } else {
            if (!window.isEmpty() && capture - window.getLast().time > MAX_FACE_GAP_MS) window.clear();
            window.addLast(new Sample(capture, raw));
            accepted++;
            if (!FrameTiming.fresh(capture, now)) afterCursorDeadline++;
            measure();
        }
        if (active() && elapsed >= TIMEOUT_MS) {
            String cause = !motion.isEmpty() ? "Sigue variando: " + motion + "."
                    : accepted < GOAL || window.size() < GOAL ? "Muestras válidas recientes insuficientes."
                    : "No hubo un intervalo válido continuo de un segundo.";
            if (lastRejected != Rejection.NONE) cause += " Último descarte: " + lastRejected.message + ".";
            state = State.TIMEOUT;
            message = "Tiempo agotado. " + cause + " Pulsa Reintentar neutral.";
        }
    }

    private static boolean valid(double[] raw) {
        if (raw == null || raw.length != 10 || raw[9] < 25) return false;
        for (double value : raw) if (!Double.isFinite(value)) return false;
        return true;
    }

    private void trim(long time) {
        while (!window.isEmpty() && time - window.getFirst().time > WINDOW_MS) window.removeFirst();
        span = window.isEmpty() ? 0 : window.getLast().time - window.getFirst().time;
    }

    private void measure() {
        // Keep the most recent interval that still has enough samples and temporal coverage.
        while (window.size() > GOAL) {
            Sample first = window.removeFirst();
            if (window.getLast().time - window.getFirst().time < MIN_SPAN_MS) {
                window.addFirst(first);
                break;
            }
        }
        span = window.getLast().time - window.getFirst().time;
        state = State.COLLECTING;
        motion = "";
        spread = null;
        if (window.size() < GOAL || span < MIN_SPAN_MS) {
            message = window.size() < GOAL ? "Reuniendo muestras válidas del rostro."
                    : "Comprobando estabilidad durante un segundo.";
            return;
        }

        double[] median = new double[10];
        spread = new double[10];
        for (int feature = 0; feature < median.length; feature++) {
            double[] axis = new double[window.size()];
            int index = 0;
            for (Sample sample : window) axis[index++] = sample.values[feature];
            Arrays.sort(axis);
            median[feature] = percentile(axis, .5);
            spread[feature] = percentile(axis, .9) - percentile(axis, .1);
        }
        String[] names = {"giro lateral", "inclinación arriba/abajo", "inclinación lateral",
                "desplazamiento horizontal", "desplazamiento vertical", "variación de distancia"};
        double[] ratios = {spread[0] / ANGLE_LIMIT, spread[1] / ANGLE_LIMIT, spread[2] / ANGLE_LIMIT,
                spread[7] / median[9] / CENTER_LIMIT, spread[8] / median[9] / CENTER_LIMIT,
                spread[9] / median[9] / WIDTH_LIMIT};
        int largest = 0;
        for (int i = 1; i < ratios.length; i++) if (ratios[i] > ratios[largest]) largest = i;
        spread[7] /= median[9];
        spread[8] /= median[9];
        spread[9] /= median[9];
        if (ratios[largest] > 1) {
            state = State.UNSTABLE;
            motion = names[largest];
            message = "La medición varía por " + motion + ".";
            return;
        }
        baseline = median;
        state = State.READY;
        message = "Neutral calibrado";
    }

    private static double percentile(double[] sorted, double fraction) {
        double index = fraction * (sorted.length - 1);
        int low = (int)index, high = (int)Math.ceil(index);
        return sorted[low] + (index - low) * (sorted[high] - sorted[low]);
    }

    Feedback feedback() {
        String detail = String.format(Locale.ROOT,
                "Neutral · recibidos %d · válidos %d · descartados %d · ventana %d en %.2f s · %.1f FPS · válidos tras límite del cursor: %d",
                received, accepted, received - accepted, window.size(), span / 1000.0,
                elapsed > 0 ? received * 1000.0 / elapsed : 0, afterCursorDeadline);
        if (!counts.isEmpty()) detail += " · " + counts;
        if (spread != null) detail += String.format(Locale.ROOT,
                "%nAmplitud P10–P90 yaw/pitch/roll %.2f° / %.2f° / %.2f° (lím. %.1f°); "
                + "centro X/Y %.1f%% / %.1f%% (lím. %.0f%%); anchura facial %.1f%% (lím. %.0f%%, relativos a separación ocular).",
                spread[0], spread[1], spread[2], ANGLE_LIMIT, spread[7]*100, spread[8]*100, CENTER_LIMIT*100,
                spread[9]*100, WIDTH_LIMIT*100);
        return new Feedback(state, face, Math.min(window.size(), GOAL), message, lastRejected.message, detail);
    }
}
