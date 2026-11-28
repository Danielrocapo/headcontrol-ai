package org.headcontrol.tracker;

import java.io.StringWriter;
import java.util.Arrays;
import java.util.Locale;

public final class PrecisePointerTest {
    private static final StablePointer.Pose NEUTRAL = new StablePointer.Pose(0,0,0,4,3);
    private static final StablePointer.Pose MOVED = new StablePointer.Pose(1.4,0,0,4.024,3);
    private static final StablePointer.Pose SMALL = new StablePointer.Pose(.2,0,0,4.021,3);
    private static final long[] IRREGULAR = {58, 79, 61, 67, 73};

    private static void require(boolean ok, String why) {
        if (!ok) throw new AssertionError(why);
    }

    private static void step(StablePointer pointer, StablePointer.Profile mode, long ms,
                             StablePointer.Pose pose, double target, boolean edge, boolean smooth) {
        pointer.update(ms, pose, target, 400, pointer.x, pointer.y, 1, mode, smooth, edge, false);
    }

    private static StablePointer anchor(StablePointer.Profile mode, double x) {
        StablePointer pointer = new StablePointer();
        for (int i=0; i<12; i++) step(pointer,mode,30*i,NEUTRAL,x,false,false);
        require(pointer.state == StablePointer.State.ANCHORED, "must anchor first");
        return pointer;
    }

    private static double percentile(double[] values) {
        Arrays.sort(values);
        return values[(int)Math.ceil(values.length*.9)-1];
    }

    private static final class Motion {
        final long firstVisible;
        final double error;
        Motion(long firstVisible, double error) { this.firstVisible = firstVisible; this.error = error; }
    }

    private static Motion correction(StablePointer.Profile mode, double correction) {
        var pointer = anchor(mode, 400);
        long t=390, first = -1;
        for (int i=0; i<12; i++) {
            step(pointer,mode,t,SMALL,400+correction,false,false);
            if (first < 0 && Math.abs(pointer.x-400) > .5) first = t-390;
            t += IRREGULAR[i%IRREGULAR.length];
        }
        return new Motion(first, Math.abs(pointer.x-400-correction));
    }

    private static final class Stop {
        final double jitter, error;
        final long movingMs;
        Stop(double jitter, double error, long movingMs) {
            this.jitter = jitter;
            this.error = error;
            this.movingMs = movingMs;
        }
    }

    private static Stop settle(StablePointer.Profile mode) {
        var pointer = anchor(mode, 400);
        long t=390, lastMovement = -1;
        for (int i=0; i<4; i++) {
            step(pointer,mode,t,MOVED,500,false,false);
            t += IRREGULAR[i];
        }
        long stopped = t;
        double[] jumps = new double[30];
        for (int i=0; i<45; i++) {
            double previous = pointer.x;
            step(pointer,mode,t,MOVED,500 + (i%2==0? 2.4 : -2.4),false,false);
            if (Math.abs(pointer.x-previous) > .08) lastMovement = t - stopped;
            if (i>=15) jumps[i-15] = Math.abs(pointer.x-previous);
            t += IRREGULAR[i%IRREGULAR.length];
        }
        return new Stop(percentile(jumps), Math.abs(pointer.x-500), lastMovement);
    }

    private static long rapid(StablePointer.Profile mode) {
        var p = anchor(mode, 400);
        long t = 390;
        for (int i = 0; i < 15; i++) {
            var face = i < 6 ? new StablePointer.Pose(.6*(i+1),0,0,4,3) : new StablePointer.Pose(3.6,0,0,4,3);
            double target = 400 + 28*Math.min(i+1, 6);
            step(p, mode, t, face, target, false, false);
            if (p.x >= 399 + 168) return t-390;
            t += IRREGULAR[i%IRREGULAR.length];
        }
        return Long.MAX_VALUE;
    }

    public static void main(String[] args) throws Exception {
        var balanced = StablePointer.Profile.BALANCED;
        var precise = StablePointer.Profile.PRECISE;
        Stop a = settle(balanced), b = settle(precise);
        Motion smallA = correction(balanced,3.8), smallB = correction(precise,3.8);
        Motion normalA = correction(balanced,6), normalB = correction(precise,6);
        long fastA = rapid(balanced), fastB = rapid(precise);
        System.out.printf(Locale.ROOT, "Post-stop P90 %.2f -> %.2f dp; final error %.2f -> %.2f dp; last motion %d -> %d ms%n",
                a.jitter,b.jitter,a.error,b.error,a.movingMs,b.movingMs);
        System.out.printf(Locale.ROOT, "3.8dp response %d -> %d ms, error %.2f -> %.2f dp; 6dp response %d -> %d ms; rapid to within 1dp %d -> %d ms%n",
                smallA.firstVisible,smallB.firstVisible,smallA.error,smallB.error,
                normalA.firstVisible,normalB.firstVisible,fastA,fastB);
        require(a.jitter > 1 && b.jitter == 0 && b.movingMs < a.movingMs && b.error <= a.error,
                "candidate resolves measured release oscillation without losing accuracy");
        require(smallA.firstVisible < 0 && smallB.firstVisible >= 0 && smallB.error <= .25, "tiny correction resolves");
        require(normalA.firstVisible >= 0 && normalB.firstVisible <= normalA.firstVisible
                && normalB.error <= Math.max(.25,normalA.error), "preserves normal precision");
        require(fastA < Long.MAX_VALUE && fastB <= fastA + 10, "rapid movement must not gain a frame of lag");

        var edge = anchor(precise,6);
        for (int i=0; i<15; i++) step(edge,precise,390+62*i,SMALL,0,true,false);
        require(edge.x==0, "endpoint reaches exact edge");
        var spikes = anchor(precise,400);
        for (int i=0; i<12; i++) step(spikes,precise,390+50*i,i%2==0? SMALL : NEUTRAL,400+(i%2==0? 4 : -4),false,false);
        require(spikes.x==400 && spikes.state==StablePointer.State.ANCHORED, "oscillating evidence cannot unlock repeatedly");

        var trace = new PointerTrace();
        var raw = StablePointer.Pose.from(new double[]{1,2,3,4,5,6,7,8,9,10}, 10);
        trace.add(new PointerTrace.Frame(100,120,precise,true,StablePointer.State.ANCHORED,
                StablePointer.State.RELEASING,raw,.35f,.4f,501.5,701.5));
        trace.drawn(100,133);
        var csv = new StringWriter();
        PointerTrace.write(trace.snapshot(), csv);
        require(csv.toString().contains("100,120,133,20,33,PRECISE,one_euro_and_anchor,ANCHORED,RELEASING,1.0,2.0,3.0")
                && csv.toString().contains("0.35,0.4,501.5,701.5"), "trace joins raw features and actual draw");
        System.out.println("PrecisePointerTest OK");
    }
}
