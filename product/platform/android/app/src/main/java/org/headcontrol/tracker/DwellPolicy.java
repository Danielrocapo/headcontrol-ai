package org.headcontrol.tracker;

/** Fires once for the same target and region until the pointer leaves. */
final class DwellPolicy {
    private Object target, fired;
    private float x, y, firedX, firedY;
    private long since;
    float progress;

    boolean update(Object next, float px, float py, long now, float radius) {
        return update(next, px, py, now, radius, PointerPolicy.DWELL_MS);
    }

    boolean update(Object next, float px, float py, long now, float radius, long holdMs) {
        if (fired != null && Math.hypot(px - firedX, py - firedY) > radius) fired = null;
        if (next == null || fired != null) {
            reset();
            return false;
        }
        if (target != next || Math.hypot(px - x, py - y) > radius) {
            target = next;
            x = px;
            y = py;
            since = now;
        }
        progress = Math.min(1, (float)(now - since) / holdMs);
        if (progress < 1) return false;
        fired = target;
        firedX = px;
        firedY = py;
        reset();
        return true;
    }

    void reset() {
        target = null;
        progress = 0;
    }
}
