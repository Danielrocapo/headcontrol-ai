package org.headcontrol.tracker;

import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

final class PipelineTrace {
    enum Stage {
        CAPTURA, ADQUISICION, MEDIAPIPE_INICIO, MEDIAPIPE_FIN, RAW_VALIDO, NORMALIZADOR,
        ONNX_INICIO, ONNX_FIN, ONNX_VALIDO, BUZON, REEMPLAZADO, SERVICIO, PUNTERO
    }

    private final EnumMap<Stage, Long> counts = new EnumMap<>(Stage.class);
    private final EnumMap<Stage, Long> times = new EnumMap<>(Stage.class);
    private final Map<String, Long> rejects = new LinkedHashMap<>();
    private String lastReject = "—", prediction = "—", clock = "—";
    private long rejectedAt;

    synchronized void mark(Stage stage, long now) {
        counts.merge(stage, 1L, Long::sum);
        times.put(stage, now);
    }

    synchronized void frame(long captured, long acquired, boolean sensor) {
        mark(Stage.CAPTURA, captured);
        mark(Stage.ADQUISICION, acquired);
        clock = sensor ? "sensor comparable" : "adquisición (edad mínima)";
    }

    synchronized void prediction(float[] xy, long captured, long finished) {
        mark(Stage.ONNX_VALIDO, finished);
        prediction = String.format(Locale.ROOT, "(%.4f, %.4f) · captura %d ms · ONNX %d ms",
                xy[0], xy[1], captured, finished);
    }

    synchronized void reject(String reason, long time) {
        rejects.merge(reason, 1L, Long::sum);
        lastReject = reason;
        rejectedAt = time;
    }

    synchronized boolean runningInference() {
        return count(Stage.ONNX_INICIO) > count(Stage.ONNX_FIN);
    }

    private long count(Stage stage) { return counts.getOrDefault(stage, 0L); }

    synchronized String summary() {
        StringBuilder stages = new StringBuilder("Etapas: conteo @ último ms monotónico · ").append(clock);
        for (Stage stage : Stage.values()) {
            stages.append('\n').append(stage).append(": ").append(count(stage)).append(" @ ")
                    .append(times.containsKey(stage) ? times.get(stage) : "—");
        }
        return stages.append("\nÚltima predicción ONNX válida: ").append(prediction)
                .append("\nÚltimo rechazo: ").append(lastReject).append(" @ ").append(rejectedAt)
                .append(" · ").append(rejects).toString();
    }
}
