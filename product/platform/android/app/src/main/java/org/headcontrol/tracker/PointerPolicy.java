package org.headcontrol.tracker;

/** Decisiones temporales y geométricas compartidas con las pruebas de escritorio. */
final class PointerPolicy {
    static final long FRESH_MS = 450, FACE_LOSS_MS = 180, REARM_MS = 400, DWELL_MS = 900;
    enum Mode { TAP, HOLD, UP, DOWN, POINTER }
    enum Sample { VALID, HOLD, FACE_LOST, STALE }
    private long validSince = -1, lastValid = -1, missingSince = -1;

    void reset() { validSince = lastValid = missingSince = -1; }

    Sample observe(long captured, long now, boolean valid) {
        if (now - captured >= FRESH_MS) {
            reset();
            return Sample.STALE;
        }
        if (valid) {
            if (validSince < 0 || captured - lastValid >= FRESH_MS) validSince = captured;
            lastValid = captured;
            missingSince = -1;
            return Sample.VALID;
        }
        validSince = -1;
        if (missingSince < 0) missingSince = now;
        return now - missingSince >= FACE_LOSS_MS ? Sample.FACE_LOST : Sample.HOLD;
    }

    boolean canArm(long now) {
        return validSince >= 0 && now - validSince >= REARM_MS && now - lastValid < FRESH_MS;
    }

    static int rotation(int sensor, int display) { return (sensor + display) % 360; }

    static int[] fit(int width, int height, int canvasWidth, int canvasHeight) {
        double factor = Math.min((double) canvasWidth / width, (double) canvasHeight / height);
        int w = (int) Math.round(width * factor), h = (int) Math.round(height * factor);
        return new int[]{(canvasWidth - w)/2, (canvasHeight - h)/2, w, h};
    }

    static double project(double normalized, int min, int max) {
        return min + Math.max(0., Math.min(1., normalized)) * (max - min - 1);
    }

    static float[] swipe(Mode mode, float y, int top, int bottom, float density) {
        float distance = Math.min((bottom - top) * .24f, 360 * density);
        float start = mode == Mode.UP ? Math.min(y, bottom - distance - 1) : Math.max(y, top + distance);
        return new float[]{start, mode == Mode.UP ? start + distance : start - distance};
    }
}
