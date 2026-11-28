package org.headcontrol.tracker;

final class PointerResponse {
    static double position(double current, double target, long elapsedMs, boolean smooth) {
        if (!smooth || Math.abs(target - current) <= 1) return target;
        return current + (target - current) * -Math.expm1(-Math.max(1, elapsedMs) / 55.0);
    }
}
