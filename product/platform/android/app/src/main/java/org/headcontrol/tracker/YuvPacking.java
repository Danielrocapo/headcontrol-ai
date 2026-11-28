package org.headcontrol.tracker;

import java.nio.ByteBuffer;

final class YuvPacking {
    static void copy(ByteBuffer input, int rowStride, int pixelStride, byte[] output, int offset,
                     int step, int width, int height) {
        int base = input.position();
        for (int y = 0; y < height; y++) {
            int row = base + y * rowStride;
            if (step == 1 && pixelStride == 1) {
                input.position(row);
                input.get(output, offset, width);
                offset += width;
            } else for (int x = 0; x < width; x++) {
                output[offset] = input.get(row + x * pixelStride);
                offset += step;
            }
        }
    }
}
