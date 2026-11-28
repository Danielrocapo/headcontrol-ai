package org.headcontrol.tracker;

final class DragPolicy {
    enum Step { HOLD, MOVE, END }

    private DragPolicy() {}

    static Step step(long now, long holdUntil, long timedRelease, boolean stop, boolean reopened) {
        if (stop || timedRelease > 0 && now >= timedRelease || reopened && now >= holdUntil) return Step.END;
        if (now < holdUntil || timedRelease > 0) return Step.HOLD;
        return Step.MOVE;
    }
}
