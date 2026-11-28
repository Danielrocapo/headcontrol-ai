package org.headcontrol.tracker;

import java.util.ArrayList;
import java.util.Arrays;

public final class CameraPerformanceTest {
    private static void require(boolean ok, String detail) {
        if (!ok) throw new AssertionError(detail);
    }

    public static void main(String[] args) {
        var latest = new LatestCapture<Integer>();
        var recycled = new ArrayList<Integer>();
        require(latest.offer(1, recycled::add), "only one drain is scheduled");
        require(!latest.offer(2, recycled::add) && !latest.offer(3, recycled::add), "new frames replace pending work");
        int processing = latest.take();
        require(processing == 3 && recycled.equals(Arrays.asList(1, 2)), "inference starts on latest");
        require(latest.offer(4, recycled::add) && !latest.offer(5, recycled::add), "latest pending while inference runs");
        latest.close(recycled::add);
        require(latest.take() == null && !latest.offer(6, recycled::add)
                && recycled.equals(Arrays.asList(1, 2, 4, 5, 6)), "shutdown recycles only buffers not in use");

        require(Arrays.equals(CameraGeometry.crop(640, 480), new int[]{0,60,640,360}), "16:9 central crop");
        int[] oldFit = PointerPolicy.fit(720,1280,1280,720);
        int[] newFit = PointerPolicy.fit(360,640,640,360);
        require(Math.abs(oldFit[0] - 2*newFit[0]) <= 1 && Math.abs(oldFit[2] - 2*newFit[2]) <= 1
                && 2*newFit[3] == oldFit[3], "portrait letterbox matches virtual 1280x720 to within rounding");
        require(PointerPolicy.rotation(270,0) == 270, "TECNO portrait rotation");

        var cursor = new PointerContinuity();
        require(cursor.display(1000) == PointerContinuity.Display.HIDDEN, "no coordinate yet");
        cursor.update(1000, false);
        require(cursor.display(1300) == PointerContinuity.Display.PREVIEW && !cursor.canAct(1300), "preview is passive");
        cursor.update(1100, true);
        require(cursor.canAct(1400), "valid and armed can act");
        cursor.freeze();
        require(cursor.display(1401) == PointerContinuity.Display.FROZEN && !cursor.canAct(1401), "one reject freezes without hiding");
        cursor.update(1400, true);
        require(cursor.canAct(1500), "fresh frame recovers");
        require(cursor.display(1850) == PointerContinuity.Display.FROZEN && !cursor.canAct(1850), "expiry is visible and inert");
        cursor.update(1800, true);
        cursor.pauseActions();
        require(cursor.display(1950) == PointerContinuity.Display.PREVIEW && !cursor.canAct(1950), "manual pause is passive");
        cursor.hide();
        require(cursor.display(2000) == PointerContinuity.Display.HIDDEN, "stop hides marker");
        System.out.println("CameraPerformanceTest OK");
    }
}
