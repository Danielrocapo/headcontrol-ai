package org.headcontrol.tracker;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.ImageFormat;
import android.hardware.camera2.*;
import android.media.Image;
import android.media.ImageReader;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Size;
import android.util.Range;
import android.util.Log;

import java.util.Collections;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.function.Consumer;

/** Camera2 source. Sensor frames wait only in ImageReader, which discards all but the latest. */
final class CameraPreview implements AutoCloseable {
    interface FrameListener {
        void onFrame(byte[] nv21, long frameId, long captureMs, long acquiredMs, long packedMs, boolean sensorClock) throws Exception;
    }

    final int width, height, orientation;
    private final ImageReader reader;
    private final String id;
    private final CameraManager manager;
    private final boolean sensorClock;
    private CameraDevice camera;
    private CameraCaptureSession session;
    private volatile boolean closed;
    private final Range<Integer> fpsRange;
    private final long minDuration;
    private final LatestCapture<Frame> latest = new LatestCapture<>();
    private final ArrayBlockingQueue<byte[]> buffers = new ArrayBlockingQueue<>(3);
    private HandlerThread acquisitionThread;
    private Handler acquisition;
    private volatile String cameraStats = "Cámara: esperando mediciones";

    private static final class Frame {
        final byte[] bytes;
        final long id, captured, acquired, packed;
        final boolean sensor;
        Frame(byte[] bytes, long id, long captured, long acquired, long packed, boolean sensor) {
            this.bytes = bytes;
            this.id = id;
            this.captured = captured;
            this.acquired = acquired;
            this.packed = packed;
            this.sensor = sensor;
        }
    }

    String stats() { return cameraStats; }

    CameraPreview(Context context) throws CameraAccessException {
        manager = context.getSystemService(CameraManager.class);
        String front = null;
        CameraCharacteristics info = null;
        for (String candidate : manager.getCameraIdList()) {
            CameraCharacteristics details = manager.getCameraCharacteristics(candidate);
            if (Integer.valueOf(CameraCharacteristics.LENS_FACING_FRONT).equals(details.get(CameraCharacteristics.LENS_FACING))) {
                front = candidate;
                info = details;
                break;
            }
        }
        if (info == null) throw new IllegalStateException("front_camera_missing");
        id = front;
        orientation = info.get(CameraCharacteristics.SENSOR_ORIENTATION);
        sensorClock = Integer.valueOf(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME)
                .equals(info.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE));
        var map = info.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        Size[] sizes = map == null ? null : map.getOutputSizes(ImageFormat.YUV_420_888);
        Size best = null;
        int bestCost = Integer.MAX_VALUE;
        if (sizes != null) for (Size size : sizes) {
            int cost = Math.abs(size.getWidth() - 640) + Math.abs(size.getHeight() - 480);
            if (cost < bestCost) { best = size; bestCost = cost; }
        }
        if (best == null) throw new IllegalStateException("camera_yuv_resolution_missing");
        width = best.getWidth();
        height = best.getHeight();
        minDuration = map.getOutputMinFrameDuration(ImageFormat.YUV_420_888, best);
        int maxFps = minDuration > 0 ? (int)Math.round(1_000_000_000.0 / minDuration) : 30;
        int target = Math.min(30, maxFps);
        Range<Integer> chosen = null;
        int rangeCost = Integer.MAX_VALUE;
        Range<Integer>[] supported = info.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
        if (supported != null) for (Range<Integer> range : supported) {
            if (range.getUpper() > maxFps) continue;
            int cost = Math.abs(range.getUpper() - target) * 10 + Math.abs(range.getLower() - 15);
            if (cost < rangeCost) { chosen = range; rangeCost = cost; }
        }
        fpsRange = chosen;
        reader = ImageReader.newInstance(width, height, ImageFormat.YUV_420_888, 3);
    }

    @SuppressLint("MissingPermission")
    @SuppressWarnings("deprecation")
    void start(Handler worker, FrameListener listener, Consumer<Throwable> onError) throws CameraAccessException {
        for (int i = 0; i < 3; i++) buffers.add(new byte[width * height * 3 / 2]);
        acquisitionThread = new HandlerThread("HeadControlAcquisition");
        acquisitionThread.start();
        acquisition = new Handler(acquisitionThread.getLooper());
        Runnable drain = () -> {
            Frame frame = latest.take();
            if (frame == null) return;
            try {
                if (!closed) listener.onFrame(frame.bytes, frame.id, frame.captured, frame.acquired, frame.packed, frame.sensor);
            } catch (Exception | LinkageError error) {
                if (!closed) onError.accept(error);
            } finally {
                recycle(frame);
            }
        };
        reader.setOnImageAvailableListener(source -> acquire(source, worker, drain, onError), acquisition);
        Log.i("HeadControl", "camera_profile=" + width + "x" + height + " AE_FPS=" + fpsRange + " min_ns=" + minDuration);
        manager.openCamera(id, new CameraDevice.StateCallback() {
            @Override public void onOpened(CameraDevice device) {
                if (closed) { device.close(); return; }
                camera = device;
                try {
                    device.createCaptureSession(Collections.singletonList(reader.getSurface()),
                            new CameraCaptureSession.StateCallback() {
                                @Override public void onConfigured(CameraCaptureSession configured) {
                                    if (closed) { configured.close(); return; }
                                    session = configured;
                                    try {
                                        CaptureRequest.Builder request = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                                        request.addTarget(reader.getSurface());
                                        if (fpsRange != null) request.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fpsRange);
                                        configured.setRepeatingRequest(request.build(), null, acquisition);
                                    } catch (CameraAccessException | RuntimeException error) { onError.accept(error); }
                                }
                                @Override public void onConfigureFailed(CameraCaptureSession configured) {
                                    configured.close();
                                    if (!closed) onError.accept(new IllegalStateException("camera_session_failed"));
                                }
                            }, acquisition);
                } catch (CameraAccessException | RuntimeException error) { onError.accept(error); }
            }
            @Override public void onDisconnected(CameraDevice device) {
                device.close();
                if (!closed) onError.accept(new IllegalStateException("camera_disconnected"));
            }
            @Override public void onError(CameraDevice device, int code) {
                device.close();
                if (!closed) onError.accept(new IllegalStateException("camera_error_" + code));
            }
        }, acquisition);
    }

    private long statsStart;
    private int acquiredCount, replacedCount;

    private void acquire(ImageReader source, Handler worker, Runnable drain, Consumer<Throwable> onError) {
        if (closed) return;
        byte[] bytes = null;
        long acquired, id, captured;
        boolean sensor;
        try (Image image = source.acquireLatestImage()) {
            if (image == null) return;
            acquired = SystemClock.elapsedRealtime();
            id = image.getTimestamp();
            sensor = sensorClock && id > 0;
            captured = sensor ? id / 1_000_000L : acquired;
            if (FrameTiming.fresh(captured, acquired)) {
                bytes = buffers.poll();
                if (bytes == null) throw new IllegalStateException("camera_buffer_pool_exhausted");
                Image.Plane[] p = image.getPlanes();
                YuvPacking.copy(p[0].getBuffer(), p[0].getRowStride(), p[0].getPixelStride(), bytes,
                        0, 1, width, height);
                YuvPacking.copy(p[2].getBuffer(), p[2].getRowStride(), p[2].getPixelStride(), bytes,
                        width * height, 2, width / 2, height / 2);
                YuvPacking.copy(p[1].getBuffer(), p[1].getRowStride(), p[1].getPixelStride(), bytes,
                        width * height + 1, 2, width / 2, height / 2);
            }
        } catch (Exception error) {
            if (bytes != null) buffers.offer(bytes);
            if (!closed) onError.accept(error);
            return;
        }
        Frame frame = new Frame(bytes, id, captured, acquired, SystemClock.elapsedRealtime(), sensor);
        if (latest.offer(frame, this::recycle)) worker.post(drain);
        else replacedCount++;
        acquiredCount++;
        if (statsStart == 0) statsStart = acquired;
        if (acquired - statsStart >= 2000) {
            cameraStats = String.format(java.util.Locale.ROOT, "Cámara %.1f FPS · reemplazados %d · YUV %d ms · perfil %dx%d · AE %s",
                    acquiredCount * 1000.0 / (acquired - statsStart), replacedCount, frame.packed - acquired, width, height, fpsRange);
            acquiredCount = replacedCount = 0;
            statsStart = acquired;
        }
    }

    private void recycle(Frame frame) {
        if (frame.bytes != null) buffers.offer(frame.bytes);
    }

    @Override public void close() {
        closed = true;
        latest.close(this::recycle);
        if (acquisition == null) {
            releaseCamera();
            return;
        }
        acquisition.post(() -> {
            try { releaseCamera(); } finally { acquisitionThread.quitSafely(); }
        });
        try {
            acquisitionThread.join(1000);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    private void releaseCamera() {
        reader.setOnImageAvailableListener(null, null);
        if (session != null) session.close();
        if (camera != null) camera.close();
        reader.close();
    }
}
