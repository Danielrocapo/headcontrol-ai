package org.headcontrol.tracker;

final class ActivationRequest {
    private static final long TIMEOUT_MS = 3000;
    private long deadline = -1;

    void request(long now) { deadline = now + TIMEOUT_MS; }
    void cancel() { deadline = -1; }
    boolean pending() { return deadline >= 0; }

    boolean expire(long now) {
        if (!pending() || now < deadline) return false;
        cancel();
        return true;
    }

    boolean consume(long now, long lastUsable, PointerPolicy gate) {
        if (expire(now) || !pending() || !FrameTiming.live(lastUsable, now) || !gate.canArm(now)) return false;
        cancel();
        return true;
    }
}
