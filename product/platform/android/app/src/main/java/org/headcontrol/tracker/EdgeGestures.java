package org.headcontrol.tracker;

final class EdgeGestures {
    static final String SETTINGS = "edge_controls";
    static final String TAP = "tap_ms", ARROW = "arrow_ms", REPEAT = "repeat_ms";
    static final int DEFAULT_TAP = 800, DEFAULT_ARROW = 350, DEFAULT_REPEAT = 550;

    enum Zone {
        UP("↑"), DOWN("↓"), LEFT("←"), RIGHT("→"),
        BACK("Atrás"), DRAG("Arrastrar"), HOME("Inicio"), PAUSE("Pausa");

        final String label;
        Zone(String label) { this.label = label; }
        boolean arrow() { return ordinal() <= RIGHT.ordinal(); }
    }

    static final class ZoneLayout {
        final float[][] bounds = new float[Zone.values().length][4];

        ZoneLayout(float left, float top, float right, float bottom, float density) {
            float w = right - left, h = bottom - top;
            float cx = (left + right) / 2, cy = (top + bottom) / 2;
            float edgeW = Math.min(44 * density, w * .18f), edgeH = Math.min(40 * density, h * .12f);
            float inset = 4 * density;
            place(Zone.UP, cx - edgeW / 2, top + inset, edgeW, edgeH);
            place(Zone.DOWN, cx - edgeW / 2, bottom - inset - edgeH, edgeW, edgeH);
            place(Zone.LEFT, left + inset, cy - edgeH / 2, edgeW, edgeH);
            place(Zone.RIGHT, right - inset - edgeW, cy - edgeH / 2, edgeW, edgeH);
            float gap = 5 * density, pill = Math.min(73 * density, (w - 8 * density - 3 * gap) / 4);
            float y = bottom - inset - 2 * edgeH - gap;
            Zone[] bottomZones = {Zone.BACK, Zone.DRAG, Zone.HOME, Zone.PAUSE};
            for (int i = 0; i < bottomZones.length; i++)
                place(bottomZones[i], cx - (4 * pill + 3 * gap) / 2 + i * (pill + gap), y, pill, edgeH);
        }

        private void place(Zone zone, float x, float y, float w, float h) {
            bounds[zone.ordinal()] = new float[]{x, y, x + w, y + h};
        }

        Zone hit(float x, float y) {
            for (Zone zone : Zone.values()) {
                float[] b = bounds[zone.ordinal()];
                if (x >= b[0] && x < b[2] && y >= b[1] && y < b[3]) return zone;
            }
            return null;
        }
    }

    static float[] swipe(Zone zone, float left, float top, float right, float bottom) {
        float cx = (left + right) / 2, cy = (top + bottom) / 2;
        float x = (right - left) * .18f, y = (bottom - top) * .18f;
        switch (zone) {
            case UP: return new float[]{cx, cy - y, cx, cy + y};
            case DOWN: return new float[]{cx, cy + y, cx, cy - y};
            case LEFT: return new float[]{cx - x, cy, cx + x, cy};
            case RIGHT: return new float[]{cx + x, cy, cx - x, cy};
            default: throw new IllegalArgumentException("Swipe requires an arrow");
        }
    }

    static final class Repeat {
        private Zone current;
        private long entered, last;
        float progress;

        boolean update(Zone zone, long now, boolean busy, long firstMs, long repeatMs) {
            if (zone != current) {
                current = zone;
                entered = now;
                last = -1;
            }
            if (zone == null) {
                progress = 0;
                return false;
            }
            long elapsed = now - (last < 0 ? entered : last);
            long limit = last < 0 ? firstMs : repeatMs;
            progress = Math.min(1f, (float) elapsed / limit);
            if (busy || elapsed < limit) return false;
            last = now;
            return true;
        }

        void reset() {
            current = null;
            progress = 0;
        }
    }

    static final class Selection {
        enum Phase { IDLE, ORIGIN, DESTINATION }
        Phase phase = Phase.IDLE;
        float x, y;

        void start() { phase = Phase.ORIGIN; }
        void cancel() { phase = Phase.IDLE; }
        float[] confirm(float px, float py) {
            if (phase == Phase.ORIGIN) {
                x = px;
                y = py;
                phase = Phase.DESTINATION;
                return null;
            }
            if (phase != Phase.DESTINATION) return null;
            phase = Phase.IDLE;
            return new float[]{x, y, px, py};
        }
    }
}
