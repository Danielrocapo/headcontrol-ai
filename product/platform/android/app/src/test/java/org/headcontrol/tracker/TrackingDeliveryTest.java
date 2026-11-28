package org.headcontrol.tracker;

public final class TrackingDeliveryTest {
    private static void require(boolean ok, String detail) {
        if (!ok) throw new AssertionError(detail);
    }

    public static void main(String[] args) {
        var gate = new PointerPolicy();
        var activation = new ActivationRequest();
        var trace = new PipelineTrace();
        activation.request(1000);
        require(!activation.consume(1020, 0, gate), "no usable frame before the request");
        long last = 0;
        boolean activated = false;
        // Reproduces 6.4 FPS with 100 ms camera delivery plus 157 ms local work.
        for (int i = 0; i < 4; i++) {
            long capture = 1050 + i * 156, acquired = capture + 100, delivered = capture + 257;
            require(!FrameTiming.fresh(capture, delivered), "0.4.1 used a shared 250 ms bound");
            require(FrameTiming.usable(capture, acquired, delivered), "two bounded stages accept the result");
            require(gate.observe(capture, delivered, true) == PointerPolicy.Sample.VALID, "fresh prediction enters gate");
            last = capture;
            if (activation.consume(delivered, last, gate)) activated = true;
        }
        require(activated && !activation.pending(), "one explicit request activates once");
        require(FrameTiming.live(last, last + 449) && !FrameTiming.live(last, last + 450), "absolute ceiling");
        require(!FrameTiming.usable(1000, 1250, 1251), "old at camera ingress");
        require(!FrameTiming.usable(1000, 1100, 1350), "backlogged local processing/UI");
        require(!FrameTiming.usable(1000, 1230, 1450), "both stages cannot exceed overall cap");
        require(!FrameTiming.usable(1101, 1100, 1300), "future camera time is rejected");
        require(FrameTiming.usable(1000, 1000, 1249), "acquisition fallback uses the local bound");

        activation.request(3000);
        gate.reset();
        require(!activation.consume(3020, last, gate) && activation.pending(), "loss denies activation");
        require(activation.expire(6000) && !activation.pending(), "request expires while waiting");
        activation.request(7000);
        activation.cancel();
        require(!activation.pending(), "explicit cancel clears request");

        trace.frame(1000, 1100, true);
        trace.mark(PipelineTrace.Stage.ONNX_INICIO, 1200);
        require(trace.runningInference(), "stage running before completion");
        trace.mark(PipelineTrace.Stage.ONNX_FIN, 1210);
        trace.prediction(new float[]{.25f,.75f}, 1000, 1210);
        trace.reject("entrega tardía", 1450);
        require(!trace.runningInference() && trace.summary().contains("ONNX_VALIDO: 1 @ 1210")
                && trace.summary().contains("(0.2500, 0.7500)") && trace.summary().contains("entrega tardía=1"),
                "retains last ONNX prediction even when rejected before delivery");
        System.out.println("TrackingDeliveryTest OK");
    }
}
