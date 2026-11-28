package org.headcontrol.tracker;

import static org.headcontrol.tracker.WinkPolicy.Event.*;
import static org.headcontrol.tracker.DragPolicy.Step.END;
import static org.headcontrol.tracker.DragPolicy.Step.MOVE;

public final class GlobalControlTest {
    private static final float[] OPEN = {.05f, .10f}, LEFT = {.85f, .13f}, RIGHT = {.09f, .68f};
    private static final float[] BOTH = {.85f, .70f};

    private static void expect(Object value, Object expected) {
        if (!expected.equals(value)) throw new AssertionError("expected " + expected + ", got " + value);
    }

    private static void calibrate(WinkPolicy p, int step, float[] values) {
        p.calibrate(step, 0);
        for (long t = 400; t < 2000; t += 100) p.sample(values, t);
        p.sample(values, 2000);
    }

    public static void main(String[] args) {
        WinkPolicy wink = new WinkPolicy();
        calibrate(wink, 0, OPEN);
        calibrate(wink, 1, BOTH);
        calibrate(wink, 2, RIGHT);
        expect(wink.ready(), false);
        calibrate(wink, 1, LEFT);
        expect(wink.ready(), true);
        expect(wink.eyesClosed(BOTH), true);
        expect(wink.eyesClosed(OPEN), false);
        expect(wink.update(OPEN, 0), NONE);
        expect(wink.update(OPEN, 180), NONE);
        expect(wink.update(new float[]{.46f, .12f}, 215), NONE);
        expect(wink.update(LEFT, 250), BEGIN);
        expect(wink.update(OPEN, 350), NONE);
        expect(wink.update(OPEN, 530), NONE);
        expect(wink.update(new float[]{.42f, .12f}, 560), NONE);
        expect(wink.update(LEFT, 600), BEGIN);
        expect(wink.update(OPEN, 850), TAP);
        expect(wink.update(OPEN, 1030), NONE);
        expect(wink.update(LEFT, 1060), BEGIN);
        expect(wink.update(BOTH, 1090), CANCEL);
        expect(wink.update(OPEN, 1250), NONE);
        expect(wink.update(OPEN, 1430), NONE);
        expect(wink.update(LEFT, 1470), BEGIN);
        expect(wink.update(null, 1500), CANCEL);
        expect(wink.update(OPEN, 1800), NONE);
        expect(wink.update(LEFT, 2000), NONE); // Most recent open interval was not confirmed.
        wink.select(1);
        expect(WinkPolicy.anatomical(.70f, .10f)[1], .70f);
        expect(wink.update(OPEN, 2000), NONE);
        expect(wink.update(LEFT, 2200), CANCEL);
        expect(wink.update(OPEN, 2250), NONE);
        expect(wink.update(OPEN, 2430), NONE);
        expect(wink.update(new float[]{.09f, .40f}, 2450), NONE);
        expect(wink.update(RIGHT, 2470), BEGIN);
        expect(wink.update(RIGHT, 2970), HOLD);
        expect(wink.update(RIGHT, 3100), NONE);
        expect(wink.update(OPEN, 3200), RELEASE);
        expect(wink.update(OPEN, 3220), NONE);

        DwellPolicy dwell = new DwellPolicy();
        Object button = new Object(), app = new Object();
        expect(dwell.update(button, 5, 5, 0, 10), false);
        expect(dwell.update(button, 5, 5, 899, 10), false);
        expect(dwell.update(button, 5, 5, 900, 10), true);
        dwell.reset(); // Pausar/Reanudar does not rearm the same location.
        expect(dwell.update(button, 5, 5, 2000, 10), false);
        expect(dwell.update(null, 5, 5, 2500, 10), false);
        expect(dwell.update(app, 5, 5, 3500, 10), false);
        expect(dwell.update(app, 5, 40, 3600, 10), false);
        expect(dwell.update(null, 5, 40, 4490, 10), false);
        expect(dwell.update(app, 5, 40, 4500, 10), false);
        expect(dwell.update(app, 5, 40, 5400, 10), true);
        expect(dwell.update(app, 5, 40, 7500, 10), false);

        expect(DragPolicy.step(0, 0, 0, false, false), MOVE);
        expect(DragPolicy.step(50, 500, 0, false, false), DragPolicy.Step.HOLD);
        expect(DragPolicy.step(50, 500, 0, false, true), DragPolicy.Step.HOLD);
        expect(DragPolicy.step(500, 500, 0, false, true), END);
        expect(DragPolicy.step(500, 500, 0, false, false), MOVE);
        expect(DragPolicy.step(51, 500, 0, true, false), END);
        expect(DragPolicy.step(120, 0, 500, false, false), DragPolicy.Step.HOLD);
        expect(DragPolicy.step(500, 0, 500, false, false), END);
        System.out.println("GlobalControlTest OK");
    }
}
