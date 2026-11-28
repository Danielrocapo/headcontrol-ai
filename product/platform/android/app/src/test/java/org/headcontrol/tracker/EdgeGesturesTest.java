package org.headcontrol.tracker;

import java.util.Arrays;
import static org.headcontrol.tracker.EdgeGestures.Zone.*;

public final class EdgeGesturesTest {
    private static void require(boolean ok, String error) {
        if (!ok) throw new AssertionError(error);
    }

    public static void main(String[] args) {
        var layout = new EdgeGestures.ZoneLayout(15, 45, 375, 730, 1);
        for (var zone : EdgeGestures.Zone.values()) {
            float[] rect = layout.bounds[zone.ordinal()];
            require(rect[0] >= 15 && rect[1] >= 45 && rect[2] <= 375 && rect[3] <= 730, "safe bounds " + zone);
            require(layout.hit((rect[0]+rect[2])/2, (rect[1]+rect[3])/2) == zone, "hit " + zone);
        }
        require(layout.hit(195,400) == null, "center passes through for normal touch");
        float[] up = EdgeGestures.swipe(UP,15,45,375,730), down = EdgeGestures.swipe(DOWN,15,45,375,730);
        float[] left = EdgeGestures.swipe(LEFT,15,45,375,730), right = EdgeGestures.swipe(RIGHT,15,45,375,730);
        require(up[3] > up[1] && down[3] < down[1] && left[2] > left[0] && right[2] < right[0],
                "finger moves opposite the content destination for all four directions");

        var repeat = new EdgeGestures.Repeat();
        require(!repeat.update(UP,100,false,350,550), "entry waits");
        require(!repeat.update(null,240,false,350,550), "quick crossing does not scroll");
        require(!repeat.update(UP,300,false,350,550), "reenter resets wait");
        require(!repeat.update(UP,649,false,350,550), "not before 350 ms");
        require(repeat.update(UP,650,false,350,550), "first scroll");
        require(!repeat.update(UP,1000,false,350,550), "repeat interval");
        require(!repeat.update(UP,1250,true,350,550), "wait pending dispatch callback");
        require(repeat.update(UP,1251,false,350,550), "repeat once callback clears busy");
        require(!repeat.update(null,2000,false,350,550), "exit stops");
        require(!repeat.update(DOWN,2010,false,350,550), "direction change starts fresh");
        repeat.reset();
        require(!repeat.update(DOWN,2500,false,350,550), "pause or face loss clears accumulated dwell");

        var dwell = new DwellPolicy();
        Object app = new Object();
        require(!dwell.update(app,10,11,100,18,800), "app dwell starts");
        require(!dwell.update(app,10,11,899,18,800), "wait 800 ms");
        require(dwell.update(app,10,11,900,18,800), "single tap");
        require(!dwell.update(app,10,11,1900,18,800), "held location does not repeat");
        require(!dwell.update(null,200,100,1950,18,800), "a control zone does not emit normal tap");
        require(!dwell.update(app,200,100,2000,18,800) && dwell.update(app,200,100,2800,18,800), "tap rearms after moving");

        var drag = new EdgeGestures.Selection();
        require(drag.confirm(10,20) == null, "idle cannot drag");
        drag.start();
        require(drag.confirm(10,20) == null && drag.phase == EdgeGestures.Selection.Phase.DESTINATION,
                "origin is stored without dispatch");
        require(Arrays.equals(drag.confirm(80,90), new float[]{10,20,80,90}), "two independent points");
        drag.start();
        drag.confirm(30,40);
        drag.cancel();
        require(drag.confirm(20,60) == null && drag.phase == EdgeGestures.Selection.Phase.IDLE, "cancel clears destination");
        System.out.println("EdgeGesturesTest OK");
    }
}
