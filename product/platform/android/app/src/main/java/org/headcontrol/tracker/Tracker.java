package org.headcontrol.tracker;

import android.content.Context;
import com.google.mediapipe.framework.image.ByteBufferImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.List;
import org.json.JSONArray;
import org.json.JSONObject;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

final class Tracker implements AutoCloseable {
    static final String[] FEATURES = {
        "yaw_rel_deg", "pitch_rel_deg", "roll_rel_deg", "nose_x_rel", "nose_y_rel",
        "chin_x_rel", "chin_y_rel", "face_center_dx", "face_center_dy"
    };
    final FaceLandmarker face;
    final OrtEnvironment env = OrtEnvironment.getEnvironment();
    final OrtSession session;
    final double[] mean = new double[9], std = new double[9];

    Tracker(Context context) throws Exception {
        JSONObject json = new JSONObject(new String(asset(context, "normalizer.json"), StandardCharsets.UTF_8));
        JSONArray names = json.getJSONArray("features");
        for (int i = 0; i < 9; i++) {
            if (!FEATURES[i].equals(names.getString(i))) throw new IllegalArgumentException("Orden de entradas");
            mean[i] = json.getJSONArray("mean").getDouble(i);
            std[i] = json.getJSONArray("std").getDouble(i);
            if (!Double.isFinite(mean[i]) || !(std[i] > 0) || !Double.isFinite(std[i]))
                throw new IllegalArgumentException("Normalizador inválido");
        }
        face = FaceLandmarker.createFromOptions(context, FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_landmarker.task").build())
                .setRunningMode(RunningMode.VIDEO).setNumFaces(1)
                .setMinFaceDetectionConfidence(.5f).setMinFacePresenceConfidence(.5f)
                .setMinTrackingConfidence(.5f).setOutputFacialTransformationMatrixes(true)
                .setOutputFaceBlendshapes(true).build());
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(1);
            session = env.createSession(asset(context, "tracker_d.onnx"), options);
        } catch (Exception failure) {
            face.close();
            throw failure;
        }
    }

    private static byte[] asset(Context context, String name) throws Exception {
        try (InputStream input = context.getAssets().open(name); ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] block = new byte[8192];
            for (int n; (n = input.read(block)) != -1;) output.write(block, 0, n);
            return output.toByteArray();
        }
    }

    String lastMeasurement = "pending";
    NeutralCalibration.Rejection geometryStatus = NeutralCalibration.Rejection.NONE;
    float[] eyes;

    double[] measure(ByteBuffer pixels, int width, int height, long millis) {
        FaceLandmarkerResult result;
        try (MPImage image = new ByteBufferImageBuilder(pixels, width, height, MPImage.IMAGE_FORMAT_RGBA).build()) {
            result = face.detectForVideo(image, millis);
        }
        eyes = null;
        if (result.faceBlendshapes().isPresent() && !result.faceBlendshapes().get().isEmpty()) {
            float left = Float.NaN, right = Float.NaN;
            // Camera frames are mirrored before MediaPipe. Convert back to anatomical sides.
            for (var item : result.faceBlendshapes().get().get(0)) {
                if ("eyeBlinkLeft".equals(item.categoryName())) left = item.score();
                if ("eyeBlinkRight".equals(item.categoryName())) right = item.score();
            }
            if (Float.isFinite(left) && Float.isFinite(right)) eyes = WinkPolicy.anatomical(left, right);
        }
        if (result.faceLandmarks().isEmpty()) {
            lastMeasurement = "face_missing";
            return reject(NeutralCalibration.Rejection.NO_FACE);
        }
        if (!result.facialTransformationMatrixes().isPresent()
                || result.facialTransformationMatrixes().get().isEmpty()) {
            lastMeasurement = "pose_matrix_missing";
            return reject(NeutralCalibration.Rejection.NO_POSE);
        }
        geometryStatus = NeutralCalibration.Rejection.NONE;
        double[] values = geometry(result.faceLandmarks().get(0), result.facialTransformationMatrixes().get().get(0));
        lastMeasurement = values == null ? "landmarks_or_pose_rejected" : "ok";
        return values;
    }

    private double[] reject(NeutralCalibration.Rejection reason) {
        geometryStatus = reason;
        return null;
    }

    private double[] geometry(List<NormalizedLandmark> landmarks, float[] matrix) {
        if (landmarks.size() <= 263) return reject(NeutralCalibration.Rejection.LANDMARKS);
        if (matrix.length != 16 || matrix[14] >= 0 ||
                Math.abs(matrix[3]) > 1e-5 || Math.abs(matrix[7]) > 1e-5 ||
                Math.abs(matrix[11]) > 1e-5 || Math.abs(matrix[15] - 1) > 1e-5) return reject(NeutralCalibration.Rejection.MATRIX);
        for (float element : matrix) if (!Float.isFinite(element)) return reject(NeutralCalibration.Rejection.MATRIX);
        NormalizedLandmark a = landmarks.get(33), b = landmarks.get(263);
        if (a.x() > b.x()) { NormalizedLandmark swap = a; a = b; b = swap; }
        double x = (b.x() - (double)a.x()) * 1280., y = (b.y() - (double)a.y()) * 720.;
        double span = Math.hypot(x, y);
        if (!(span >= 25) || x <= 0) return reject(NeutralCalibration.Rejection.EYE_SPAN);
        double cx = (a.x() + (double)b.x()) * 640., cy = (a.y() + (double)b.y()) * 360.;
        Mat m = new Mat(3, 3, CvType.CV_64F), u = new Mat(), s = new Mat(), vt = new Mat();
        Mat rotation = new Mat(), r = new Mat(), q = new Mat(), empty = new Mat();
        try {
            for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
                m.put(i, j, matrix[j * 4 + i]);
            Core.SVDecomp(m, s, u, vt);
            double min = s.get(2, 0)[0];
            if (min <= 1e-6 || s.get(0, 0)[0] / min > 1.05) return reject(NeutralCalibration.Rejection.GEOMETRY);
            Core.gemm(u, vt, 1, empty, 0, rotation);
            if (Core.determinant(rotation) <= 0) return reject(NeutralCalibration.Rejection.GEOMETRY);
            for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
                if ((i == 0) != (j == 0)) rotation.put(i, j, -rotation.get(i,j)[0]);
            double[] angles = Calib3d.RQDecomp3x3(rotation, r, q);
            double[] values = {angles[1], angles[0], angles[2], 0, 0, 0, 0, cx, cy, span};
            for (int index = 0; index < 2; index++) {
                NormalizedLandmark p = landmarks.get(index == 0 ? 1 : 152);
                double dx = (p.x() * 1280. - cx) / span, dy = (p.y() * 720. - cy) / span;
                values[3 + index * 2] = (dx * x + dy * y) / span;
                values[4 + index * 2] = (-dx * y + dy * x) / span;
            }
            for (double value : values) if (!Double.isFinite(value)) return reject(NeutralCalibration.Rejection.NONFINITE);
            return values;
        } finally {
            for (Mat mat : new Mat[]{m,u,s,vt,rotation,r,q,empty}) mat.release();
        }
    }

    float[] normalize(double[] raw, double[] base) {
        float[] input = new float[9];
        for (int i = 0; i < 9; i++) {
            double delta = raw[i] - base[i];
            if (i < 3) delta -= 360 * Math.floor((delta + 180) / 360);
            if (i >= 7) delta /= base[9];
            input[i] = (float)((delta - mean[i]) / std[i]);
        }
        return input;
    }

    float[] infer(float[] input) throws Exception {
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), new long[]{1, 9});
             OrtSession.Result result = session.run(Collections.singletonMap("features", tensor))) {
            float[] xy = ((float[][])result.get(0).getValue())[0];
            return Float.isFinite(xy[0]) && Float.isFinite(xy[1]) ? xy : null;
        }
    }

    @Override public void close() throws Exception {
        try { session.close(); } finally { face.close(); }
    }
}
