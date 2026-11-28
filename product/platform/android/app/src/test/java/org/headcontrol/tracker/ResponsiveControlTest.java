package org.headcontrol.tracker;

import java.nio.ByteBuffer;
import java.util.Arrays;

public final class ResponsiveControlTest {
    private static void require(boolean condition) {
        if (!condition) throw new AssertionError();
    }

    public static void main(String[] args) {
        require(FrameTiming.fresh(100, 349));
        require(!FrameTiming.fresh(100, 350));
        require(!FrameTiming.fresh(150, 149));
        var latest = new FrameTiming.Latest<Long>();
        require(latest.offer(100L));
        require(!latest.offer(200L));
        require(!latest.offer(300L));
        require(latest.take() == 300L);
        require(latest.offer(400L));
        require(latest.take() == 400L);

        double edge = PointerPolicy.project(1.05, 0, 2400);
        require(edge == 2399);
        require(PointerResponse.position(0, edge, 33, false) == edge);
        require(PointerResponse.position(edge, 0, 33, false) == 0);
        double smooth = 0;
        for (int i = 0; i < 14; i++) smooth = PointerResponse.position(smooth, edge, 33, true);
        require(smooth == edge);
        double slow = 300.0;
        for (int i = 0; i < 10; i++) {
            double target = 301 + i * 2;
            double next = PointerResponse.position(slow, target, 33, true);
            require(next > slow && next <= target);
            slow = next;
        }
        // Android YUV planes may have row padding and share interleaved chroma storage.
        byte[] nv21 = new byte[24];
        byte[] luma = {1,2,3,4,0,0,5,6,7,8,0,0,9,10,11,12,0,0,13,14,15,16};
        byte[] chroma = {21,31,22,32,0,0,23,33,24,34};
        var u = ByteBuffer.wrap(chroma);
        u.position(1);
        YuvPacking.copy(ByteBuffer.wrap(luma), 6, 1, nv21, 0, 1, 4, 4);
        YuvPacking.copy(ByteBuffer.wrap(chroma), 6, 2, nv21, 16, 2, 2, 2);
        YuvPacking.copy(u, 6, 2, nv21, 17, 2, 2, 2);
        require(Arrays.equals(nv21, new byte[]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
                21,31,22,32,23,33,24,34}));
        System.out.println("ResponsiveControlTest OK");
    }
}
