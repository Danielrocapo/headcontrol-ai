package org.headcontrol.tracker;

import java.util.function.Consumer;

/** One replaceable frame waiting behind the single inference worker. */
final class LatestCapture<T> {
    private T pending;
    private boolean closed;

    synchronized boolean offer(T next, Consumer<T> recycle) {
        if (closed) {
            recycle.accept(next);
            return false;
        }
        boolean schedule = pending == null;
        if (!schedule) recycle.accept(pending);
        pending = next;
        return schedule;
    }

    synchronized T take() {
        T result = pending;
        pending = null;
        return result;
    }

    synchronized void close(Consumer<T> recycle) {
        closed = true;
        if (pending != null) recycle.accept(take());
    }
}
