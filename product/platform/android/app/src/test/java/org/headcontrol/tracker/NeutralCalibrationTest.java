package org.headcontrol.tracker;

import java.util.Arrays;

import static org.headcontrol.tracker.NeutralCalibration.Rejection.*;
import static org.headcontrol.tracker.NeutralCalibration.State.*;

public final class NeutralCalibrationTest {
    private static double[] pose(double angle, double centerX) {
        return new double[]{angle, .3, -.2, .01, -.03, .2, .4, centerX, 358, 130};
    }

    private static void require(boolean condition, String label) {
        if (!condition) throw new AssertionError(label);
    }

    private static void add(NeutralCalibration c, long time, double[] raw) {
        c.record(time * 1_000_000L, time, time + 300, raw, NONE);
    }

    public static void main(String[] args) {
        var steady = new NeutralCalibration(0);
        require(steady.feedback().state == WAITING, "initial UI");
        int oldAccepted = 0;
        for (int i = 0; i < 12; i++) {
            long time = 100 + i*220 + i%3*18;
            if (time >= 500 && time <= 2500 && FrameTiming.fresh(time, time + 300)) oldAccepted++;
            double[] raw = pose(i == 4 ? 43 : .25*Math.sin(i), 640 + i*.19);
            add(steady, time, raw);
            require(steady.feedback().progress == i + 1, "UI progress counts valid frames only");
        }
        require(oldAccepted < NeutralCalibration.GOAL, "0.4 cursor deadline reproduces frame starvation");
        require(steady.feedback().state == READY && steady.feedback().message.equals("Neutral calibrado"), "ready after slow inference");
        require(steady.feedback().detail.contains("límite del cursor: 12"), "latency diagnostic");
        require(Math.abs(steady.baseline()[0]) < 1 && steady.baseline()[7] > 640, "outlier and small gradual drift");

        var noisy = new NeutralCalibration(0);
        for (int i = 1; i <= 6; i++) add(noisy, i*110, pose(.2, 640));
        noisy.record(660_000_000L, 660, 700, pose(0, 640), NONE);
        require(noisy.feedback().progress == 6 && noisy.feedback().discarded.contains("repetido"), "duplicate rejection");
        noisy.record(700_000_000L, 700, 740, null, NO_FACE);
        require(noisy.feedback().face == NeutralCalibration.Face.MISSING && noisy.feedback().progress == 6, "short face loss preserves progress");
        noisy.record(730_000_000L, 730, 980, null, STALE);
        var invalid = pose(0, 640);
        invalid[5] = Double.NaN;
        noisy.record(750_000_000L, 750, 780, invalid, NONE);
        require(noisy.feedback().discarded.contains("no finitas") && noisy.feedback().progress == 6, "invalid features");
        for (int i = 8; i <= 13; i++) add(noisy, i*110, pose(.1, 641));
        require(noisy.feedback().state == READY && noisy.feedback().detail.contains("descartados 4"), "face recovery and accounting");

        var moving = new NeutralCalibration(0);
        for (int i = 1; moving.active(); i++) add(moving, i*120L, pose(i%2 == 0 ? 18 : -18, 640));
        require(moving.feedback().state == TIMEOUT && moving.baseline() == null, "motion blocks calibration");
        require(moving.feedback().message.contains("giro lateral") && moving.feedback().message.contains("Reintentar"), "specific timeout feedback");
        var recovering = new NeutralCalibration(0);
        for (int i = 1; i <= 14; i++) add(recovering, i*120, pose(0, i%2 == 0 ? 660 : 620));
        require(recovering.feedback().state == UNSTABLE && recovering.feedback().message.contains("horizontal"), "translation diagnosis");
        for (int i = 15; i <= 28; i++) add(recovering, i*120, pose(0, 640));
        require(recovering.feedback().state == READY, "sliding window recovers without retry");

        var noFace = new NeutralCalibration(0);
        for (int i = 1; i <= 150; i++) noFace.record(i, i*100L, i*100L, null, NO_FACE);
        require(noFace.feedback().state == TIMEOUT && noFace.feedback().progress == 0
                && noFace.feedback().message.contains("no detectó rostro"), "poor detectability timeout");
        require(new NeutralCalibration(20_000).feedback().progress == 0, "retry starts a new window");

        int rotation = PointerPolicy.rotation(270, 0);
        require(rotation == 270 && Arrays.equals(PointerPolicy.fit(720, 1280, 1280, 720), new int[]{437,0,405,720}),
                "Tecno portrait rotation and frame dimensions");
        System.out.println("NeutralCalibrationTest OK");
    }
}
