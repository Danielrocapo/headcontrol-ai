package org.headcontrol.tracker;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;

/** Half-resolution image with the original 16:9 letterbox and mirrored portrait convention. */
final class TrackingImage implements AutoCloseable {
    static final int WIDTH = 640, HEIGHT = 360;
    private final ByteBuffer pixels = ByteBuffer.allocateDirect(WIDTH * HEIGHT * 4);
    private final Mat yuv, rgba, crop, frame, area;
    private final Mat oriented = new Mat();
    private final int rotation;
    private final Size fitSize;

    TrackingImage(int width, int height, int rotation) {
        this.rotation = rotation;
        yuv = new Mat(height * 3 / 2, width, CvType.CV_8UC1);
        rgba = new Mat(height, width, CvType.CV_8UC4);
        int[] fov = CameraGeometry.crop(width, height);
        crop = rgba.submat(new Rect(fov[0], fov[1], fov[2], fov[3]));
        boolean vertical = rotation % 180 != 0;
        int[] fit = PointerPolicy.fit(vertical ? fov[3] : fov[2], vertical ? fov[2] : fov[3], WIDTH, HEIGHT);
        frame = new Mat(HEIGHT, WIDTH, CvType.CV_8UC4, pixels);
        frame.setTo(new Scalar(0, 0, 0, 255));
        area = frame.submat(new Rect(fit[0], fit[1], fit[2], fit[3]));
        fitSize = new Size(fit[2], fit[3]);
    }

    ByteBuffer prepare(byte[] nv21) {
        yuv.put(0, 0, nv21);
        Imgproc.cvtColor(yuv, rgba, Imgproc.COLOR_YUV2RGBA_NV21);
        if (rotation % 180 == 0) {
            Core.flip(crop, oriented, rotation == 0 ? 1 : 0);
        } else {
            Core.transpose(crop, oriented);
            if (rotation == 270) Core.flip(oriented, oriented, -1);
        }
        Imgproc.resize(oriented, area, fitSize);
        pixels.rewind();
        return pixels;
    }

    @Override public void close() {
        for (Mat mat : new Mat[]{area, frame, oriented, crop, rgba, yuv}) mat.release();
    }
}
