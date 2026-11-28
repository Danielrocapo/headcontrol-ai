package org.headcontrol.tracker;

import java.util.Arrays;

/** Se ejecuta con javac y java. No abre cámara ni requiere un dispositivo. */
public final class PointerPolicyTest {
    private static void check(boolean ok) {
        if (!ok) throw new AssertionError("Falló una condición de PointerPolicy.");
    }

    public static void main(String[] args) {
        PointerPolicy gate = new PointerPolicy();
        check(!gate.canArm(1000));
        check(gate.observe(1000, 1025, true) == PointerPolicy.Sample.VALID);
        check(!gate.canArm(1200));
        check(gate.observe(1430, 1440, true) == PointerPolicy.Sample.VALID && gate.canArm(1440));
        check(gate.observe(1450, 1490, false) == PointerPolicy.Sample.HOLD && !gate.canArm(1490));
        check(gate.observe(1500, 1540, false) == PointerPolicy.Sample.HOLD);
        check(gate.observe(1610, 1670, false) == PointerPolicy.Sample.FACE_LOST);
        check(gate.observe(1750, 1760, true) == PointerPolicy.Sample.VALID && !gate.canArm(1760));
        check(gate.observe(2160, 2180, true) == PointerPolicy.Sample.VALID && gate.canArm(2180));
        check(gate.observe(2160, 2610, true) == PointerPolicy.Sample.STALE && !gate.canArm(2610));
        gate.reset();
        check(!gate.canArm(2700));

        check(PointerPolicy.Mode.values()[0] == PointerPolicy.Mode.TAP);
        check(PointerPolicy.DWELL_MS == 900);
        check(PointerPolicy.rotation(270, 0) == 270 && PointerPolicy.rotation(270, 90) == 0);
        check(Arrays.equals(PointerPolicy.fit(720, 1280, 1280, 720), new int[]{437, 0, 405, 720}));
        check(Arrays.equals(PointerPolicy.fit(1280, 720, 1280, 720), new int[]{0, 0, 1280, 720}));
        check(PointerPolicy.project(-.2, 24, 1104) == 24);
        check(PointerPolicy.project(1.2, 24, 1104) == 1103);
        check(PointerPolicy.project(.5, 32, 232) == 131.5);
        float[] up = PointerPolicy.swipe(PointerPolicy.Mode.UP, 1090f, 24, 1104, 2f);
        float[] down = PointerPolicy.swipe(PointerPolicy.Mode.DOWN, 25f, 24, 1104, 2f);
        check(up[0] >= 24 && up[1] > up[0] && up[1] <= 1103);
        check(down[0] <= 1103 && down[1] < down[0] && down[1] >= 24);
        System.out.println("Apertura, caducidad, pérdida transitoria/sostenida, rearme, retrato y bordes verificados.");
    }
}
