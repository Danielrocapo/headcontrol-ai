package org.headcontrol.tracker;

final class FrameTiming {
    static final long MAX_AGE_MS = 250;

    static boolean fresh(long capture, long now) { return now >= capture && now - capture < MAX_AGE_MS; }

    // Keep the camera ingress limit and the existing face-loss/rearm age bound.
    // Their clocks share elapsedRealtime when the camera supplies a comparable sensor timestamp.
    static boolean live(long capture, long now) {
        return capture > 0 && now >= capture && now - capture < PointerPolicy.FRESH_MS;
    }

    static boolean usable(long capture, long acquired, long now) {
        return fresh(capture, acquired) && fresh(acquired, now) && live(capture, now);
    }

    /** Only the most recent result waits for the UI thread, with one runnable at most. */
    static final class Latest<T> {
        private T pending;

        synchronized boolean offer(T item) {
            boolean schedule = pending == null;
            pending = item;
            return schedule;
        }

        synchronized T take() {
            T result = pending;
            pending = null;
            return result;
        }
    }
}
