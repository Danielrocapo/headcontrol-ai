package org.headcontrol.tracker;

/** Tracks the diagnostic target independently of the gesture callback. */
final class GestureProbe {
    enum Callback { WAITING, COMPLETED, CANCELLED, ERROR }

    private boolean sent, targetHit;
    private Boolean accepted;
    private Callback callback = Callback.WAITING;
    private String cause = "";
    private float x, y;

    void send(float x, float y) {
        sent = true;
        this.x = x;
        this.y = y;
    }

    void accepted(boolean value) { accepted = value; }
    void complete(boolean value) {
        if (callback != Callback.ERROR) callback = value ? Callback.COMPLETED : Callback.CANCELLED;
    }
    void error(String cause) { this.cause = cause; callback = Callback.ERROR; }

    void hit() {
        if (sent && callback != Callback.CANCELLED && callback != Callback.ERROR) targetHit = true;
    }

    boolean passed() { return sent && Boolean.TRUE.equals(accepted) && callback == Callback.COMPLETED && targetHit; }
    boolean inFlight() { return sent && callback == Callback.WAITING; }

    String report() {
        String outcome = callback == Callback.WAITING ? "PENDIENTE"
                : callback == Callback.COMPLETED ? "COMPLETADO" : callback == Callback.CANCELLED ? "CANCELADO" : "ERROR";
        return "Solicitud enviada " + (sent ? "SÍ" : "NO") + " · dispatchGesture " + (accepted == null ? "—" : accepted)
                + "\nCallback " + outcome + " · destino recibió el toque " + (targetHit ? "SÍ" : "NO")
                + "\nCoordenadas " + (sent ? "(" + Math.round(x) + ", " + Math.round(y) + ")" : "—")
                + (cause.isEmpty() ? "" : "\n" + cause);
    }
}
