package org.headcontrol.tracker;

import java.util.Arrays;
import java.util.Locale;

public final class StablePointerTest {
    private static final StablePointer.Profile BALANCED = StablePointer.Profile.BALANCED;
    private static final StablePointer.Pose NEUTRAL = new StablePointer.Pose(0,0,0,4,3);

    private static void require(boolean ok, String detail) {
        if (!ok) throw new AssertionError(detail);
    }

    private static void step(StablePointer p, long now, StablePointer.Pose pose, double x, double y) {
        p.update(now, pose, x, y, p.x, p.y, 1, BALANCED, false, false, false);
    }

    private static StablePointer anchoredAt(double x, double y) {
        var p = new StablePointer();
        for (int i = 0; i <= 8; i++) step(p, i*33, NEUTRAL, x, y);
        require(p.state == StablePointer.State.ANCHORED, "neutral anchors");
        return p;
    }

    private static double p90(double[] values) {
        Arrays.sort(values);
        return values[(int)Math.ceil(values.length * .9)-1];
    }

    public static void main(String[] args) {
        var p = new StablePointer();
        var dwell = new DwellPolicy();
        double[] raw = new double[60], stabilized = new double[60];
        double previousRaw = 0, previousOut = 0, anchor = 0;
        int taps = 0;
        for (int i = 0; i < 75; i++) {
            long t = i * 33;
            double input = 400 + 2.4*Math.sin(i*1.7);
            var face = new StablePointer.Pose(.09*Math.cos(i), .07*Math.sin(i), 0, 4+.002*Math.sin(i), 3);
            step(p, t, face, input, 500);
            if (i == 14) anchor = p.x;
            if (i >= 15) {
                raw[i-15] = Math.abs(input - previousRaw);
                stabilized[i-15] = Math.abs(p.x - previousOut);
                require(p.x == anchor && p.state == StablePointer.State.ANCHORED, "anchor holds exact coordinates");
            }
            if (dwell.update(dwell, (float)p.x, (float)p.y, t, 18, 800)) taps++;
            previousRaw = input;
            previousOut = p.x;
        }
        require(taps == 1, "800 ms dwell completes once even when anchored");
        double rawP90 = p90(raw), stabilizedP90 = p90(stabilized);
        require(rawP90 > 3 && stabilizedP90 == 0, "stationary jitter P90");

        var outlier = anchoredAt(400,500);
        step(outlier, 297, new StablePointer.Pose(4,0,0,4,3), 510, 570);
        step(outlier, 330, NEUTRAL, 400,500);
        for (int i = 11; i < 80; i++) step(outlier, i*33, NEUTRAL, 400 + .8*i, 500);
        require(outlier.x == 400 && outlier.y == 500, "isolated jump and RAW-only slow drift never release");

        var precise = anchoredAt(400,500);
        var translated = new StablePointer.Pose(0,0,0,4.024,3);
        long firstMotion = 330, released = -1, firstVisible = -1;
        for (int i = 0; i < 8; i++) {
            long t = firstMotion + 67*i;
            step(precise, t, translated, 406,500);
            if (precise.state == StablePointer.State.RELEASING && released < 0) {
                released = t;
                require(precise.x == 400, "no jump in the release frame");
            }
            if (precise.x > 400 && firstVisible < 0) firstVisible = t;
        }
        require(released == 397 && firstVisible == 464, "release at 15 FPS");
        require(Math.abs(precise.x - 406) <= .1, "small movement reaches intended target");

        var fast = anchoredAt(400,500);
        var turned = new StablePointer.Pose(3,1,0,4,3);
        step(fast, 297, turned, 550,590);
        step(fast, 330, turned, 610,625);
        step(fast, 363, turned, 660,650);
        require(fast.state == StablePointer.State.RELEASING && fast.x == 400, "sustained fast motion releases");
        step(fast, 396, turned, 680,670);
        require(fast.x > 400 && Math.hypot(fast.x-400,fast.y-500) <= 21, "continuous bounded start");

        var edge = anchoredAt(2,2);
        for (int i = 0; i < 14; i++)
            edge.update(330+i*33, turned, 0, 0, edge.x, edge.y, 1, BALANCED, false, true, true);
        require(edge.x == 0 && edge.y == 0, "corner reach survives anchoring");
        var recovered = anchoredAt(302,425);
        recovered.suspend();
        recovered.suspend();
        step(recovered, 800, NEUTRAL, 360,425);
        require(recovered.state == StablePointer.State.RELEASING && recovered.x == 302, "recovery keeps the frozen position");
        step(recovered, 833, NEUTRAL, 360,425);
        require(recovered.x > 302 && recovered.x < 360, "recovery eases toward the new prediction");
        edge.reset();
        require(edge.state == StablePointer.State.MOVING, "loss clears the anchor");
        edge.update(1000, NEUTRAL, 100,80, edge.x,edge.y,1,StablePointer.Profile.OFF,false,false,false);
        require(edge.x == 100 && edge.y == 80, "off is direct");

        System.out.printf(Locale.ROOT,
                "P90 stationary %.2f -> %.2f dp; precision release %d ms; first movement %d ms; final error %.2f dp%n",
                rawP90, stabilizedP90, released-firstMotion, firstVisible-firstMotion, Math.abs(precise.x-406));
    }
}
