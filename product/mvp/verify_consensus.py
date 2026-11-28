"""Comparación de fijaciones a 30 FPS, con el actuador Windows y la cámara sustituidos."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QEvent, QPoint
from PySide6.QtWidgets import QApplication

import tracker_windows as desktop
from absolute_smoother import ConsensusFixation, StableFixation
from verify_accessibility import FakeSink


def simulate(app, consensus):
    window = desktop.Window(consensus=consensus)
    window.timer.stop()
    sink = FakeSink()
    base = np.array([780, 420])
    sink.cursor[:] = base
    window.scale = np.array([1919, 1079])
    window.configure_absolute()
    window.sink = sink
    window.access_bar.cursor_point = lambda: QPoint(*map(int, sink.cursor))

    def send(x, y):
        sink.cursor[:] = (x, y)
        return 1

    window.api = SimpleNamespace(GetAsyncKeyState=lambda _: 0, SetCursorPos=send)
    frame = 0

    def step(point, pose=0.):
        nonlocal frame
        frame += 1
        stamp = frame / 30
        window.vision.publish_sample(stamp, np.asarray(point) / window.scale, ("blendshapes", (.03, .03)),
                                     "valid", np.array([pose, 0., 0., 0., 0.]))
        with patch.object(desktop, "monotonic", return_value=stamp):
            window.tick()
        return sink.cursor.copy()

    def start():
        window.pause()
        sink.cursor[:] = base
        step(base)
        with patch.object(desktop, "monotonic", return_value=frame/30):
            window.toggle()
        for _ in range(26):
            step(base)
        assert window.fixation.anchor is not None

    def trial(goal, pose, duration):
        start()
        first, first_step, peak = None, 0., 0.
        for i in range(duration):
            shown = step(goal, pose)
            distance = np.linalg.norm(shown - base)
            if first is None and distance >= 1:
                first, first_step = (i+1)*1000/30, distance
            peak = max(peak, distance)
        return first, first_step, peak, float(np.linalg.norm(sink.cursor - goal))

    try:
        assert type(window.fixation) is (ConsensusFixation if consensus else StableFixation)
        start()
        window.change_desktop_clicks(True)
        rng = np.random.default_rng(486)
        jitter = [step(base + rng.normal(0, .9, 2), .03) for _ in range(70)]
        assert sink.events == ["left"]
        assert np.max(np.linalg.norm(np.diff(jitter, axis=0), axis=1)) == 0
        window.change_desktop_clicks(False)

        start()
        assert np.array_equal(step(base + [105, -80], 1.5), base)
        assert np.array_equal(step(base, 0.), base)
        # A weak screen excursion lasting 133 ms after coming to rest.
        transient = trial(base + [30, 0], .45, 4)
        for _ in range(15):
            step(base)
        rebound = np.linalg.norm(sink.cursor - base)
        weak = trial(base + [30, 0], .45, 7)
        fine_3 = trial(base + [3, 0], .46, 25)
        fine_6 = trial(base + [0, 6], .57, 25)
        rapid = trial(base + [120, 65], 2.7, 40)
        start()
        for _ in range(100):
            step([0, 420], -7.)
        assert sink.cursor[0] == 0
        assert window.fixation.anchor is not None
        window.actions.dragging = True
        step([20, 420], -6.)
        assert window.fixation.anchor is None and not window.fixation_active()
        window.actions.dragging = False
        window.fixation_method.setCurrentIndex(int(not consensus))
        assert not window.armed and type(window.fixation) is (StableFixation if consensus else ConsensusFixation)
        return transient, rebound, weak, fine_3, fine_6, rapid
    finally:
        window.close()
        window.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def main():
    app = QApplication([])
    original = simulate(app, False)
    candidate = simulate(app, True)
    for label, old, new in zip(("excursion 133ms", "return error", "weak excursion", "fine 3", "fine 6", "rapid"),
                               original, candidate):
        print(f"{label}: {old} -> {new}")
    assert original[2][2] > 0 and candidate[2][2] == 0
    assert candidate[3][3] == candidate[4][3] == 0
    assert candidate[3][0] <= 135 and candidate[4][0] <= 135
    assert candidate[5][0] <= original[5][0] + 34
    assert candidate[5][1] <= original[5][1]
    print("Verified: exact anchor, dwell, physical pixel corrections, edges, drag and mode switch.")


if __name__ == "__main__":
    main()
