package org.headcontrol.tracker;

import java.util.ArrayDeque;
import java.util.Arrays;

final class StablePointer {
    static final String PREF = "stable_pointer_profile";
    static final int DEFAULT = 1;
    private static final long MAX_GAP_MS = 240, RELEASE_MS = 45;

    enum Profile {
        OFF(0, 0, 0, 0), BALANCED(200, 3, 4.5, 11), FIRM(240, 3.6, 5.5, 14),
        PRECISE(150, 3, 3.2, 11), CONSENSUS(150, 3, 3, 11);
        final long settleMs;
        final double quietDp, precisionDp, exitDp;
        Profile(long settleMs, double quietDp, double precisionDp, double exitDp) {
            this.settleMs = settleMs;
            this.quietDp = quietDp;
            this.precisionDp = precisionDp;
            this.exitDp = exitDp;
        }
        boolean adaptive() { return this == PRECISE || this == CONSENSUS; }
        static Profile at(int index) {
            return index == 0 ? OFF : index == 2 ? FIRM : index == 3 ? PRECISE : index == 4 ? CONSENSUS : BALANCED;
        }
    }

    enum State { MOVING, SETTLING, ANCHORED, RELEASING }

    static final class Pose {
        final double yaw, pitch, roll, cx, cy;
        final double[] raw;
        Pose(double yaw, double pitch, double roll, double cx, double cy) {
            this(yaw, pitch, roll, cx, cy, null);
        }
        private Pose(double yaw, double pitch, double roll, double cx, double cy, double[] raw) {
            this.yaw = yaw;
            this.pitch = pitch;
            this.roll = roll;
            this.cx = cx;
            this.cy = cy;
            this.raw = raw;
        }
        static Pose from(double[] face, double neutralSpan) {
            return new Pose(face[0], face[1], face[2], face[7] / neutralSpan, face[8] / neutralSpan, face);
        }
        double distance(Pose other) {
            double angles = Math.hypot(Math.hypot(delta(yaw, other.yaw), delta(pitch, other.pitch)),
                    delta(roll, other.roll));
            return Math.max(angles, 25 * Math.hypot(cx - other.cx, cy - other.cy));
        }
        double alignment(Pose first, Pose origin) {
            double ay = delta(yaw, origin.yaw), ap = delta(pitch, origin.pitch), ar = delta(roll, origin.roll);
            double ax = 25*(cx-origin.cx), az = 25*(cy-origin.cy);
            double by = delta(first.yaw, origin.yaw), bp = delta(first.pitch, origin.pitch), br = delta(first.roll, origin.roll);
            double bx = 25*(first.cx-origin.cx), bz = 25*(first.cy-origin.cy);
            double length = Math.sqrt((ay*ay+ap*ap+ar*ar+ax*ax+az*az)*(by*by+bp*bp+br*br+bx*bx+bz*bz));
            return length == 0 ? 0 : (ay*by+ap*bp+ar*br+ax*bx+az*bz)/length;
        }
        private static double delta(double a, double b) { return Math.IEEEremainder(a - b, 360); }
    }

    private static final class Sample {
        final long time;
        final double x, y;
        final Pose pose;
        Sample(long time, double x, double y, Pose pose) {
            this.time = time;
            this.x = x;
            this.y = y;
            this.pose = pose;
        }
    }

    private final ArrayDeque<Sample> window = new ArrayDeque<>();
    private final OneEuro euro = new OneEuro();
    double minimumCutoff = OneEuro.DEFAULT_CUTOFF / 10.0, beta = OneEuro.DEFAULT_BETA / 100.0;
    private Profile profile;
    State state = State.MOVING;
    double x, y;
    private long last = -1, releasingSince, releaseCandidate = -1;
    private double rawX, rawY, directionX, directionY;
    private Pose anchorPose, candidatePose;
    private long weakSince = -1;
    private int weakSamples;
    private Pose weakPose;
    private boolean guardedRelease;
    private boolean resumeFromAnchor;

    void reset() {
        window.clear();
        euro.reset();
        last = releaseCandidate = -1;
        state = State.MOVING;
        resumeFromAnchor = false;
        guardedRelease = false;
    }

    void suspend() {
        boolean resume = resumeFromAnchor || state == State.ANCHORED || state == State.RELEASING;
        reset();
        resumeFromAnchor = resume;
    }

    void update(long now, Pose pose, double tx, double ty, double currentX, double currentY,
                double density, Profile mode, boolean smooth, boolean edgeX, boolean edgeY) {
        if (mode != profile || last >= 0 && now <= last) reset();
        else if (last >= 0 && now - last > MAX_GAP_MS) suspend();
        profile = mode;
        long dt = last < 0 ? 0 : now - last;
        last = now;
        x = currentX;
        y = currentY;
        if (mode == Profile.OFF || pose == null) {
            window.clear();
            state = State.MOVING;
            x = PointerResponse.position(x, tx, dt, smooth && dt > 0);
            y = PointerResponse.position(y, ty, dt, smooth && dt > 0);
            return;
        }

        if (resumeFromAnchor) {
            resumeFromAnchor = false;
            releasingSince = now;
            state = State.RELEASING;
            return;
        }

        if (state == State.ANCHORED) {
            boolean edgePending = edgeX && tx != x || edgeY && ty != y;
            if (intentional(now, pose, tx, ty, density, edgePending)) {
                releasingSince = now;
                state = State.RELEASING;
                window.clear();
                if (mode.adaptive()) {
                    double distance = Math.hypot(tx-x, ty-y);
                    double start = Math.min(distance*.46, (guardedRelease ? 4 : 7)*density);
                    if (distance > 0) {
                        x += (tx-x)*start/distance;
                        y += (ty-y)*start/distance;
                    }
                }
            }
            return;
        }
        if (state == State.RELEASING) {
            double distance = Math.hypot(tx - x, ty - y);
            long sinceRelease = now - releasingSince;
            double maxSpeed = sinceRelease < 100 ? (guardedRelease ? .38 : .62) : 1.1;
            double step = Math.min(distance * -Math.expm1(-dt / 38.0),
                    density * dt * maxSpeed);
            if (distance <= density) {
                x = tx;
                y = ty;
                state = State.MOVING;
                euro.reset();
            } else {
                x += (tx - x) * step / distance;
                y += (ty - y) * step / distance;
            }
            if (mode.adaptive()) considerAnchor(now, pose, tx, ty, density, edgeX, edgeY, state == State.RELEASING);
            return;
        }

        if (mode.adaptive()) {
            euro.step(now, tx, ty, density, minimumCutoff, beta);
            x = edgeX && Math.abs(euro.x-tx) <= density ? tx : euro.x;
            y = edgeY && Math.abs(euro.y-ty) <= density ? ty : euro.y;
        } else {
            x = PointerResponse.position(x, tx, dt, smooth && dt > 0);
            y = PointerResponse.position(y, ty, dt, smooth && dt > 0);
        }
        considerAnchor(now, pose, tx, ty, density, edgeX, edgeY, false);
    }

    private void considerAnchor(long now, Pose pose, double tx, double ty, double density,
                                boolean edgeX, boolean edgeY, boolean releasing) {
        Profile mode = profile;
        Sample newest = new Sample(now, tx, ty, pose);
        window.addLast(newest);
        while (window.size() > 3 && now - window.getFirst().time > mode.settleMs + 75) window.removeFirst();
        if (!releasing) state = State.SETTLING;
        if (window.size() < 3 || now - window.getFirst().time < mode.settleMs) return;

        int size = window.size();
        Sample[] recent = window.toArray(new Sample[0]);
        double[] xs = new double[size], ys = new double[size], spreads = new double[size], faces = new double[size];
        for (int i = 0; i < size; i++) {
            xs[i] = recent[i].x;
            ys[i] = recent[i].y;
        }
        Arrays.sort(xs);
        Arrays.sort(ys);
        double mx = xs[size/2], my = ys[size/2];
        if (mode.adaptive()) {
            // Center the central half of the samples so alternating noise cannot bias the latch.
            int lower = size/4, upper = size-1-lower;
            mx = (xs[lower] + xs[upper])/2;
            my = (ys[lower] + ys[upper])/2;
        }
        Sample reference = recent[size/2];
        for (int i = 0; i < size; i++) {
            spreads[i] = Math.hypot(recent[i].x - mx, recent[i].y - my) / density;
            faces[i] = recent[i].pose.distance(reference.pose);
        }
        Arrays.sort(spreads);
        Arrays.sort(faces);
        int quantile = (int)Math.floor((size - 1)*.8);
        boolean moving = spreads[quantile] > mode.quietDp || faces[quantile] > .35
                || Math.hypot(tx - mx, ty - my) > mode.quietDp * density
                || pose.distance(reference.pose) > .35 || Math.hypot(x - mx, y - my) > mode.quietDp * density
                || edgeX && x != tx || edgeY && y != ty;
        if (moving) {
            if (!releasing) state = State.MOVING;
            return;
        }
        rawX = mx;
        rawY = my;
        anchorPose = reference.pose;
        releaseCandidate = -1;
        if (mode.adaptive()) {
            // Finish the last sub-dp correction once, then keep the position exactly fixed.
            double dx = (edgeX ? tx : mx) - x, dy = (edgeY ? ty : my) - y;
            double distance = Math.hypot(dx, dy);
            if (distance > 0) {
                double fraction = Math.min(1, .75*density/distance);
                x += dx*fraction;
                y += dy*fraction;
            }
        }
        state = State.ANCHORED;
        euro.reset();
    }

    private boolean intentional(long now, Pose pose, double tx, double ty, double density, boolean edgePending) {
        double dx = tx - rawX, dy = ty - rawY;
        double distance = Math.hypot(dx, dy) / density;
        double head = pose.distance(anchorPose);
        boolean far = distance >= profile.exitDp && head >= .36;
        boolean precise = distance >= profile.precisionDp && head >= (profile.adaptive() ? .37 : .42);
        boolean edge = edgePending && Math.hypot(dx, dy) >= .5 && head >= .42;
        if (!far && !precise && !edge) {
            releaseCandidate = -1;
            return false;
        }
        double alignment = dx * directionX + dy * directionY;
        double minimumAlignment = profile.adaptive() ? .65*Math.hypot(dx,dy)*Math.hypot(directionX,directionY) : 0;
        boolean consensus = profile == Profile.CONSENSUS;
        // Large screen displacement with weak facial support takes a separate, longer path.
        // Fine corrections and reaching a clipped edge keep the short confirmation.
        boolean weak = consensus && far && !edge && head < .85;
        boolean disagree = consensus && releaseCandidate >= 0 && pose.alignment(candidatePose, anchorPose) < .65;
        if (releaseCandidate < 0 || alignment <= minimumAlignment || disagree) {
            releaseCandidate = now;
            directionX = dx;
            directionY = dy;
            candidatePose = pose;
            weakSince = weak ? now : -1;
            weakSamples = weak ? 1 : 0;
            weakPose = weak ? pose : null;
            return false;
        }
        long elapsed = now - releaseCandidate;
        if (weak) {
            if (weakSince < 0) {
                weakSince = now;
                weakPose = pose;
            }
            weakSamples++;
            guardedRelease = true;
            return weakSamples >= 4 && now - weakSince >= 160 && pose.distance(weakPose) >= .16;
        }
        guardedRelease = consensus && far && !edge && weakSince >= 0;
        weakSince = -1;
        weakSamples = 0;
        // A small readjustment may finish in place, but a slowly drifting RAW estimate alone cannot release.
        return elapsed >= RELEASE_MS && (far || edge || pose.distance(candidatePose) >= .10 || elapsed >= 65);
    }
}
