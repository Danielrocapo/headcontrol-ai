package org.headcontrol.tracker;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public final class PointerTraceReplay {
    private static final class Row {
        final long capture;
        final long drawn;
        final String previous, state;
        final double[] face = new double[10];
        final float x, y;
        final double shownX, shownY;
        Row(String line) {
            String[] c = line.split(",", -1);
            capture = Long.parseLong(c[0]);
            drawn = c[2].isEmpty() ? 0 : Long.parseLong(c[2]);
            previous = c[7];
            state = c[8];
            for (int i = 0; i < 10; i++) face[i] = Double.parseDouble(c[9+i]);
            x = Float.parseFloat(c[19]);
            y = Float.parseFloat(c[20]);
            shownX = Double.parseDouble(c[21]);
            shownY = Double.parseDouble(c[22]);
        }
    }

    private static final class Release {
        int frame;
        long onset, firstMovement = -1;
        final double x, y;
        double max;
        boolean quick;
        String inputs;
        Release(int frame, long onset, double x, double y) {
            this.frame = frame;
            this.onset = onset;
            this.x = x;
            this.y = y;
        }
    }

    private static final class Result {
        final List<Release> events = new ArrayList<>();
        int exact, anchored, releasing;
        double[] x, y;
    }

    private static java.lang.reflect.Field field(String name) throws Exception {
        var field = StablePointer.class.getDeclaredField(name);
        field.setAccessible(true);
        return field;
    }

    private static Result replay(Row[] rows, int start, double density, int[] safe, double eyeSpan,
                                 StablePointer.Profile mode) throws Exception {
        var p = new StablePointer();
        p.minimumCutoff = 2.0;
        var anchorPose = field("anchorPose");
        var releaseCandidate = field("releaseCandidate");
        var rawX = field("rawX");
        var rawY = field("rawY");
        var result = new Result();
        result.x = new double[rows.length];
        result.y = new double[rows.length];
        Release active = null;
        for (int i = start; i < rows.length; i++) {
            Row row = rows[i];
            var pose = StablePointer.Pose.from(row.face, eyeSpan);
            double x = PointerPolicy.project(row.x, safe[0], safe[2]);
            double y = PointerPolicy.project(row.y, safe[1], safe[3]);
            var previous = p.state;
            double currentX = p.x, currentY = p.y;
            p.update(row.capture, pose, x, y, currentX, currentY, density, mode, false,
                    row.x <= 0 || row.x >= 1, row.y <= 0 || row.y >= 1);
            result.x[i] = p.x;
            result.y[i] = p.y;
            if (p.state.name().equals(row.state)) result.exact++;
            if (p.state == StablePointer.State.ANCHORED) result.anchored++;
            if (p.state == StablePointer.State.RELEASING) result.releasing++;
            if (previous == StablePointer.State.ANCHORED && p.state == StablePointer.State.RELEASING) {
                long onset = releaseCandidate.getLong(p);
                double dx = (x - rawX.getDouble(p))/density, dy = (y - rawY.getDouble(p))/density;
                var origin = (StablePointer.Pose)anchorPose.get(p);
                double angle = Math.hypot(Math.hypot(Math.IEEEremainder(pose.yaw-origin.yaw, 360),
                        Math.IEEEremainder(pose.pitch-origin.pitch, 360)), Math.IEEEremainder(pose.roll-origin.roll, 360));
                double center = 25*Math.hypot(pose.cx-origin.cx,pose.cy-origin.cy);
                active = new Release(i, onset, currentX, currentY);
                double distance = Math.hypot(dx,dy), head = Math.max(angle, center);
                boolean edge = ((row.x <= 0 || row.x >= 1) && x != currentX
                        || (row.y <= 0 || row.y >= 1) && y != currentY) && distance*density >= .5 && head >= .42;
                String regime = distance >= mode.exitDp && head >= .36 ? (edge ? "far+edge" : "far")
                        : edge ? "edge" : "fine";
                active.inputs = String.format(Locale.ROOT, "%s raw dp=(%.1f,%.1f) ang=%.3f centerEq=%.3f confirm=%dms",
                        regime,dx,dy,angle,center,row.capture-onset);
                result.events.add(active);
            }
            if (active != null) {
                double displacement = Math.hypot(p.x-active.x,p.y-active.y);
                active.max = Math.max(active.max, displacement);
                if (displacement > .5*density && active.firstMovement < 0) active.firstMovement = row.capture;
                if (p.state != StablePointer.State.RELEASING) {
                    active.quick = p.state == StablePointer.State.ANCHORED && row.capture-rows[active.frame].capture <= 500;
                    active = null;
                }
            }
        }
        return result;
    }

    private static double median(double[] values) {
        Arrays.sort(values);
        return values.length == 0 ? Double.NaN : (values[(values.length-1)/2] + values[values.length/2])/2;
    }

    private static void print(String label, Row[] rows, Result result, int start, boolean details) {
        double delay = median(result.events.stream().filter(e -> e.firstMovement >= 0)
                .mapToDouble(e -> e.firstMovement-e.onset).toArray());
        System.out.printf(Locale.ROOT, "%s match=%d/%d release=%d short=%d anchored=%d releasing=%d confirm_to_visible_median_ms=%.1f%n",
                label, result.exact, rows.length-start, result.events.size(), result.events.stream().filter(e->e.quick).count(),
                result.anchored, result.releasing, delay);
        if (details) for (Release event : result.events) {
            System.out.printf(Locale.ROOT, "  %.3fs%s peak=%.1fpx %s%n",
                    (rows[event.frame].capture-rows[0].capture)/1000.0, event.quick ? " short" : "", event.max, event.inputs);
        }
    }

    private static void compareTiming(Row[] rows, Result reference, Result candidate) {
        var diffs = new ArrayList<Double>();
        var available = new ArrayList<>(candidate.events);
        for (Release old : reference.events) {
            if (old.firstMovement < 0) continue;
            Release closest = null;
            long gap = 301;
            for (Release recent : available) {
                if (recent.firstMovement < 0) continue;
                long time = Math.abs(rows[recent.frame].capture - rows[old.frame].capture);
                if (time < gap) {
                    closest = recent;
                    gap = time;
                }
            }
            if (closest != null) {
                available.remove(closest);
                diffs.add((double)(closest.firstMovement - old.firstMovement));
            }
        }
        double[] values = diffs.stream().mapToDouble(v -> v).toArray();
        if (values.length == 0) {
            System.out.println("matched=0");
            return;
        }
        System.out.printf(Locale.ROOT, "matched=%d first_visible_delta_median=%.1fms p90=%.0fms max=%.0fms%n",
                values.length, median(values), values[(int)Math.ceil(values.length*.9)-1], values[values.length-1]);
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 6) throw new IllegalArgumentException("CSV left top right bottom density [neutral-eye-span]");
        Row[] rows;
        try (var lines = Files.lines(Path.of(args[0]))) {
            rows = lines.skip(1).filter(line -> !line.isBlank()).map(Row::new).toArray(Row[]::new);
        }
        int[] safe = new int[4];
        for (int i = 0; i < 4; i++) safe[i] = Integer.parseInt(args[i+1]);
        double density = Double.parseDouble(args[5]);
        double medianSpan = median(Arrays.stream(rows).mapToDouble(row -> row.face[9]).toArray());
        double eyeSpan = args.length > 6 ? Double.parseDouble(args[6]) : medianSpan;
        int start = 0;
        // The trace starts anchored with earlier state missing. Start where the original filter has just converged.
        while (start < rows.length && !(rows[start].previous.equals("RELEASING") && rows[start].state.equals("SETTLING"))) start++;
        if (start == rows.length) throw new IllegalArgumentException("No filter convergence point in trace");
        double draw = median(Arrays.stream(rows).filter(row -> row.drawn > 0).mapToDouble(row -> row.drawn-row.capture).toArray());
        System.out.printf(Locale.ROOT, "rows=%d prefix=%d eyeSpan=%.3f session_median_span=%.3f capture_to_draw_median_ms=%.1f%n",
                rows.length, start, eyeSpan, medianSpan, draw);
        var reference = replay(rows, start, density, safe, eyeSpan, StablePointer.Profile.PRECISE);
        print("reference", rows, reference, start, true);
        var candidate = replay(rows, start, density, safe, eyeSpan, StablePointer.Profile.CONSENSUS);
        print("candidate", rows, candidate, start, true);
        compareTiming(rows, reference, candidate);
    }
}
