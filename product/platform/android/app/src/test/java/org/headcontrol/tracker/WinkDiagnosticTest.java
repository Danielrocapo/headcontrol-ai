package org.headcontrol.tracker;

public final class WinkDiagnosticTest {
    private static final float[] OPEN = {.05f,.10f}, LEFT = {.85f,.13f}, RIGHT = {.09f,.68f}, BOTH = {.85f,.70f};

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    private static void calibrate(WinkPolicy policy, int step, float[] data) {
        policy.calibrate(step, 0);
        for (long t = 400; t < 2000; t += 100) policy.sample(data, t);
        policy.sample(data, 2000);
    }

    public static void main(String[] args) {
        var a = new GestureProbe();
        require(!a.passed() && a.report().contains("Solicitud enviada NO"), "A inactive initially");
        a.send(321, 567);
        a.accepted(true);
        a.complete(true);
        require(!a.passed(), "Android callback alone does not pass A");
        a.hit();
        require(a.passed() && a.report().contains("321, 567"), "callback and target verify A");
        var cancelled = new GestureProbe();
        cancelled.send(1, 2);
        cancelled.accepted(true);
        cancelled.hit();
        cancelled.complete(false);
        require(!cancelled.passed() && cancelled.report().contains("CANCELADO"), "cancellation is not success");
        var rejected = new GestureProbe();
        rejected.send(1, 2);
        rejected.accepted(false);
        rejected.error("Android rechazó dispatchGesture");
        rejected.complete(true);
        require(!rejected.passed() && rejected.report().contains("ERROR"), "late callback cannot overwrite error");

        var b = new WinkPolicy();
        calibrate(b, 0, OPEN);
        calibrate(b, 1, LEFT);
        calibrate(b, 2, RIGHT);
        b.setPreview(true);
        b.sample(OPEN, 3000);
        b.sample(OPEN, 3180);
        b.sample(BOTH, 3230);
        require(!b.previewConfirmed() && b.eyeReport(3230, false).contains("cierre bilateral"), "bilateral rejection");
        b.sample(OPEN, 3300);
        b.sample(OPEN, 3480);
        b.sample(LEFT, 3520);
        require(b.eyeReport(3520, false).contains("detectado SÍ"), "simulated wink in preview");
        require(b.update(LEFT, 3550) == WinkPolicy.Event.NONE, "preview never produces a dispatchable event");
        b.sample(OPEN, 3750);
        require(b.previewConfirmed() && b.eyeReport(3750, false).contains("Guiño confirmado"), "dry run verifies on reopen");
        b.setPreview(false);
        require(b.previewConfirmed(), "C can be enabled after leaving B");
        b.select(1);
        require(!b.previewConfirmed(), "new eye requires its own proof");
        require(WinkPolicy.anatomical(.52f,.14f)[1] == .52f, "anatomical mirror mapping");
        System.out.println("WinkDiagnosticTest OK");
    }
}
