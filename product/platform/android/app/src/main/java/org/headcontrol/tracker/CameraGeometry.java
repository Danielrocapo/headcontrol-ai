package org.headcontrol.tracker;

final class CameraGeometry {
    static int[] crop(int width, int height) {
        int w = Math.min(width, height * 16 / 9);
        int h = Math.min(height, width * 9 / 16);
        return new int[]{(width - w) / 2, (height - h) / 2, w, h};
    }
}
