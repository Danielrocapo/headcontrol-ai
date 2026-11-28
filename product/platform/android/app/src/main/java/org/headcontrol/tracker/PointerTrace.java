package org.headcontrol.tracker;

import java.io.IOException;
import java.io.Writer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

/** Opt-in chronological ring buffer. Export detaches the frames from the UI before disk I/O. */
final class PointerTrace {
    static final int LIMIT = 1800;
    private final ArrayDeque<Frame> frames = new ArrayDeque<>();

    static final class Frame {
        final long capture, published;
        final StablePointer.Profile profile;
        final boolean smooth;
        final StablePointer.State previous, state;
        final StablePointer.Pose pose;
        final float rawX, rawY;
        final double visibleX, visibleY;
        long drawn;

        Frame(long capture, long published, StablePointer.Profile profile, boolean smooth,
                StablePointer.State previous, StablePointer.State state, StablePointer.Pose pose,
                float rawX, float rawY, double visibleX, double visibleY) {
            this.capture = capture;
            this.published = published;
            this.profile = profile;
            this.smooth = smooth;
            this.previous = previous;
            this.state = state;
            this.pose = pose;
            this.rawX = rawX;
            this.rawY = rawY;
            this.visibleX = visibleX;
            this.visibleY = visibleY;
        }
    }

    void add(Frame frame) {
        if (frames.size() == LIMIT) frames.removeFirst();
        frames.addLast(frame);
    }

    void drawn(long capture, long now) {
        Frame frame = frames.peekLast();
        if (frame != null && frame.capture == capture && frame.drawn == 0) frame.drawn = now;
    }

    int size() { return frames.size(); }
    List<Frame> snapshot() { return new ArrayList<>(frames); }

    static void write(List<Frame> frames, Writer out) throws IOException {
        out.write("capture_ms,published_ms,drawn_ms,age_at_publish_ms,age_at_draw_ms,profile,filter,previous_state,state,");
        out.write("yaw_deg,pitch_deg,roll_deg,nose_x,nose_y,chin_x,chin_y,face_center_x,face_center_y,eye_span,");
        out.write("onnx_x,onnx_y,overlay_x_px,overlay_y_px\n");
        for (Frame f : frames) {
            StringBuilder line = new StringBuilder(210);
            line.append(f.capture).append(',').append(f.published).append(',');
            if (f.drawn > 0) line.append(f.drawn);
            line.append(',').append(f.published-f.capture).append(',');
            if (f.drawn > 0) line.append(f.drawn-f.capture);
            line.append(',').append(f.profile).append(',')
                    .append(f.profile.adaptive() ? "one_euro_and_anchor"
                            : f.smooth ? "smooth" : "direct")
                    .append(',').append(f.previous).append(',').append(f.state).append(',');
            double[] pose = f.pose == null ? null : f.pose.raw;
            for (int i = 0; i < 10; i++) {
                if (pose != null) line.append(pose[i]);
                line.append(',');
            }
            line.append(f.rawX).append(',').append(f.rawY).append(',').append(f.visibleX).append(',')
                    .append(f.visibleY).append('\n');
            out.write(line.toString());
        }
    }
}
