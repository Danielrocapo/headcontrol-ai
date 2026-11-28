package org.headcontrol.tracker;

/** Filters screen positions. Derivative is measured in dp/s with actual capture intervals. */
final class OneEuro {
    static final String CUTOFF = "one_euro_cutoff_tenths", BETA = "one_euro_beta_hundredths";
    static final int DEFAULT_CUTOFF = 30, DEFAULT_BETA = 22;
    private long last = -1;
    private double rawX, rawY, vx, vy;
    double x, y;

    void reset() { last = -1; }

    void step(long time, double tx, double ty, double density, double minCutoff, double beta) {
        if (last < 0 || time <= last) {
            x = rawX = tx;
            y = rawY = ty;
            vx = vy = 0;
            last = time;
            return;
        }
        double dt = (time-last)/1000.0;
        last = time;
        double a = gain(4, dt);
        vx += a*((tx-rawX)/(density*dt) - vx);
        vy += a*((ty-rawY)/(density*dt) - vy);
        rawX = tx;
        rawY = ty;
        double positionGain = gain(minCutoff + beta*Math.hypot(vx,vy), dt);
        x += positionGain*(tx-x);
        y += positionGain*(ty-y);
    }

    private static double gain(double cutoff, double dt) {
        double scaled = 2*Math.PI*cutoff*dt;
        return scaled/(1+scaled);
    }
}
