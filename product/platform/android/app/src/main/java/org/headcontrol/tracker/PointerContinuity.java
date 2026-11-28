package org.headcontrol.tracker;

final class PointerContinuity {
    enum Display { HIDDEN, FROZEN, PREVIEW, ACTIVE }
    private long capture;
    private boolean valid, armed;

    void update(long capture, boolean armed) {
        this.capture = capture;
        this.armed = armed;
        valid = true;
    }

    void freeze() { valid = false; }
    void pauseActions() { armed = false; }
    void hide() { capture = 0; valid = armed = false; }

    Display display(long now) {
        if (capture == 0) return Display.HIDDEN;
        if (!valid || !FrameTiming.live(capture, now)) return Display.FROZEN;
        return armed ? Display.ACTIVE : Display.PREVIEW;
    }

    boolean canAct(long now) { return display(now) == Display.ACTIVE; }
}
