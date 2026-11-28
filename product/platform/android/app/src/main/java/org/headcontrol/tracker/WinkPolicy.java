package org.headcontrol.tracker;

import java.util.ArrayList;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Locale;

final class WinkPolicy {
    enum Event { NONE, BEGIN, TAP, HOLD, RELEASE, CANCEL }
    private final float[] open = {Float.NaN, Float.NaN}, closed = {Float.NaN, Float.NaN};
    private final ArrayList<Float> left = new ArrayList<>(), right = new ArrayList<>();
    private int stage = -1;
    private int eye;
    private long start, onset = -1, openSince = -1, lastOpen = -1, duration;
    private boolean holding;
    private String gesture = "Esperando apertura estable";
    private boolean preview, previewConfirmed;
    private float[] lastEyes;
    private long lastSample = -1, rateStart = -1;
    private int rateCount;
    private double eyeHz;
    private final ArrayDeque<String> transitions = new ArrayDeque<>();
    private String lastReject = "—", reopening = "—";

    String gestureTiming() { return gesture + " · " + duration + " ms"; }
    String status = "Calibra los ojos para activar el guiño";

    void select(int eye) {
        if (this.eye != eye) {
            previewConfirmed = false;
            lastReject = reopening = "—";
            transitions.clear();
        }
        this.eye = eye;
        reset();
    }

    void setPreview(boolean enabled) {
        if (preview == enabled) return;
        preview = enabled;
        reset();
    }

    boolean preview() { return preview; }
    boolean previewConfirmed() { return previewConfirmed; }
    boolean ready() {
        return Float.isFinite(closed[0]) && Float.isFinite(closed[1])
                && closed[0] - open[0] >= .15f && closed[1] - open[1] >= .15f;
    }
    void calibrate(int stage, long now) {
        previewConfirmed = false;
        lastReject = reopening = "—";
        transitions.clear();
        if (stage == 0) {
            open[0] = open[1] = closed[0] = closed[1] = Float.NaN;
        } else {
            closed[stage - 1] = Float.NaN;
        }
        this.stage = stage;
        start = now;
        left.clear();
        right.clear();
        reset();
        status = "Mantén la posición ocular durante 2 s";
    }
    boolean calibrating() { return stage >= 0; }
    void cancelCalibration() {
        if (stage >= 0) status = "Calibración ocular interrumpida";
        stage = -1;
        reset();
    }
    private static float median(ArrayList<Float> samples) {
        Collections.sort(samples);
        return samples.get(samples.size() / 2);
    }
    void sample(float[] eyes, long now) {
        lastEyes = eyes;
        lastSample = now;
        if (eyes != null) {
            if (rateStart < 0 || now - rateStart > 1500) {
                rateStart = now;
                rateCount = 0;
            } else if (now - rateStart >= 1000) {
                eyeHz = rateCount * 1000.0 / (now - rateStart);
                rateStart = now;
                rateCount = 0;
            }
            rateCount++;
        }
        if (stage < 0) {
            if (preview) {
                var event = evaluate(eyes, now);
                if (event == Event.TAP) previewConfirmed = true;
            }
            return;
        }
        if (eyes != null && now - start >= 400 && now - start < 2000) {
            left.add(eyes[0]);
            right.add(eyes[1]);
        }
        if (now - start < 2000) return;
        if (left.size() < 10) {
            status = "Muestras oculares insuficientes. Repite esta etapa.";
        } else {
            float a = median(left), b = median(right);
            if (stage == 0) {
                open[0] = a;
                open[1] = b;
                status = "Ojos abiertos registrados. Calibra ambos guiños.";
            } else {
                int chosen = stage - 1, other = 1 - chosen;
                float signal = chosen == 0 ? a : b, opposite = other == 0 ? a : b;
                if (!(signal - open[chosen] >= .15f)
                        || opposite - open[other] >= Math.max(.10f, (signal - open[chosen]) * .45f)) {
                    status = "Cierre insuficiente, bilateral o en el canal contrario. Repite esta etapa.";
                } else {
                    closed[chosen] = signal;
                    status = ready() ? "Calibración ocular lista" : "Guiño registrado. Falta el otro ojo.";
                }
            }
        }
        stage = -1;
    }
    void reset() {
        onset = openSince = lastOpen = -1;
        holding = false;
        duration = 0;
        gesture = "Esperando apertura estable";
    }

    private float threshold(int side, float fraction) {
        return open[side] + (closed[side] - open[side]) * fraction;
    }

    String diagnostic(float[] eyes, boolean enabled, boolean armed) {
        String name = eye == 0 ? "izquierdo" : "derecho";
        return String.format(Locale.ROOT,
                "MediaPipe RAW Left/Right: %.3f / %.3f\nAnatómico I/D: %.3f / %.3f · elegido %s\n"
                + "Umbrales I cierre/reapertura: %.3f / %.3f · D: %.3f / %.3f\n"
                + "Máximo del otro ojo I/D: %.3f / %.3f\n"
                + "Guiño %s · control %s · %s · %d ms",
                eyes == null ? Float.NaN : eyes[1], eyes == null ? Float.NaN : eyes[0],
                eyes == null ? Float.NaN : eyes[0], eyes == null ? Float.NaN : eyes[1], name,
                threshold(0, .65f), threshold(0, .30f), threshold(1, .65f), threshold(1, .30f),
                threshold(0, .35f), threshold(1, .35f),
                enabled ? "habilitado" : "deshabilitado", armed ? "activo" : "pausado", gesture, duration);
    }

    String eyeReport(long now, boolean enabled) {
        float[] eyes = lastSample >= 0 && now - lastSample < 450 ? lastEyes : null;
        return diagnostic(eyes, enabled, enabled) + String.format(Locale.ROOT,
                "\nOjo izquierdo %s · derecho %s · muestras recientes %.1f Hz"
                + "\nCierre unilateral detectado %s · confirmación al reabrir: %s"
                + "\nÚltimo rechazo: %s\nGuiños efectivos %s · prueba B %s\nTransiciones: %s",
                eyeState(eyes, 0), eyeState(eyes, 1), eyes == null ? 0 : eyeHz,
                onset >= 0 ? "SÍ" : "NO", reopening, lastReject, enabled ? "SÍ" : "NO",
                previewConfirmed ? "VERIFICADA" : "PENDIENTE", String.join(" | ", transitions));
    }

    private String eyeState(float[] eyes, int side) {
        if (eyes == null) return "SIN SEÑAL";
        if (!ready()) return "SIN CALIBRAR";
        if (eyes[side] >= threshold(side, .65f)) return "CERRADO";
        if (eyes[side] <= threshold(side, .30f)) return "ABIERTO";
        return "EN TRANSICIÓN";
    }

    static float[] anatomical(float modelLeft, float modelRight) {
        return new float[]{modelRight, modelLeft};
    }

    boolean eyesClosed(float[] eyes) {
        if (eyes == null) return false;
        if (!ready()) return eyes[0] > .55f || eyes[1] > .55f;
        return (eyes[0] - open[0]) / (closed[0] - open[0]) > .35f
                || (eyes[1] - open[1]) / (closed[1] - open[1]) > .35f;
    }
    Event update(float[] eyes, long now) {
        // The dry run consumes the same camera samples in sample(), without exposing any event to Android.
        return preview ? Event.NONE : evaluate(eyes, now);
    }

    private Event evaluate(float[] eyes, long now) {
        Event result = classify(eyes, now);
        boolean rejected = gesture.startsWith("Rechazado:") || result == Event.CANCEL;
        if (result == Event.TAP || result == Event.RELEASE) reopening = gesture + " · " + duration + " ms";
        if (result == Event.BEGIN) reopening = "Esperando reapertura";
        if ((result != Event.NONE && result != Event.CANCEL) || (rejected && !gesture.equals(lastReject))) {
            if (transitions.size() == 6) transitions.removeFirst();
            transitions.addLast(now + " ms " + (rejected ? gesture : result + " (" + duration + " ms)"));
        }
        if (rejected) lastReject = gesture;
        return result;
    }

    private Event classify(float[] eyes, long now) {
        if (!ready() || eyes == null) {
            reset();
            gesture = eyes == null ? "Sin señales oculares" : "Falta calibración";
            return Event.CANCEL;
        }
        int other = 1 - eye;
        float chosen = (eyes[eye] - open[eye]) / (closed[eye] - open[eye]);
        float opposite = (eyes[other] - open[other]) / (closed[other] - open[other]);
        if (opposite > .35f) {
            reset();
            gesture = chosen > .35f ? "Rechazado: cierre bilateral" : "Rechazado: ojo contrario";
            return Event.CANCEL;
        }
        if (onset >= 0) {
            duration = now - onset;
            if (chosen <= .30f) {
                long elapsed = duration;
                Event result = holding ? Event.RELEASE : elapsed >= 190 && elapsed < 500
                        ? Event.TAP : Event.NONE;
                reset();
                duration = elapsed;
                openSince = lastOpen = now;
                gesture = result == Event.RELEASE ? "Reapertura confirmada · soltar"
                        : result == Event.TAP ? "Guiño confirmado · toque" : elapsed < 190
                        ? "Rechazado: cierre medido menor de 190 ms" : "Rechazado: reapertura tardía sin cierre sostenido confirmado";
                return result;
            }
            if (!holding && duration >= 500 && chosen >= .55f) {
                holding = true;
                gesture = "Cierre sostenido · pulsación larga";
                return Event.HOLD;
            }
        } else if (chosen >= .65f && chosen - opposite >= .40f && openSince >= 0
                && lastOpen - openSince >= 180 && now - lastOpen <= 250) {
            onset = now;
            openSince = lastOpen = -1;
            duration = 0;
            gesture = "Cierre unilateral · esperando reapertura";
            return Event.BEGIN;
        } else if (chosen <= .30f) {
            if (lastOpen < 0 || now - lastOpen > 250) openSince = now;
            lastOpen = now;
            if (duration == 0) gesture = now - openSince >= 180 ? "Ojos abiertos · listo" : "Estabilizando apertura";
        } else {
            // A normal closing ramp crosses intermediate scores before the close threshold.
            if (lastOpen < 0 || now - lastOpen > 250) openSince = lastOpen = -1;
            gesture = chosen < .65f ? "Cierre parcial" : chosen - opposite < .40f
                    ? "Rechazado: contraste insuficiente entre ojos"
                    : "Rechazado: apertura previa ausente, breve o caducada";
        }
        return Event.NONE;
    }
}
