"""Simula anclaje absoluto, permanencia y pausa con las salidas de Windows interceptadas."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QEvent, QPoint
from PySide6.QtWidgets import QApplication

import tracker_windows as desktop
from absolute_smoother import StableFixation
from verify_accessibility import FakeSink


def main():
    boundary = StableFixation()
    point = np.array([780, 420])
    scale = np.array([1919, 1079])
    for i in range(6):
        boundary.step(i/30, point/scale, None, point, scale, True)
        assert boundary.anchor is None
    boundary.step(.20, point/scale, None, point, scale, True)
    np.testing.assert_array_equal(boundary.anchor, point)
    app = QApplication([])
    window = desktop.Window()
    window.timer.stop()
    sink = FakeSink()
    sink.cursor[:] = (780, 420)
    window.scale = np.array([1919, 1079])
    window.configure_absolute()
    window.sink = sink
    window.access_bar.cursor_point = lambda: QPoint(int(sink.cursor[0]), int(sink.cursor[1]))
    positions = []

    def send(x, y):
        positions.append((x, y))
        sink.cursor[:] = (x, y)
        return 1

    window.api = SimpleNamespace(GetAsyncKeyState=lambda _: 0, SetCursorPos=send)
    frame = 0

    def step(raw, angle=0., eyes=(.03, .03), lost=False):
        nonlocal frame
        frame += 1
        stamp = frame / 30
        point = None if lost else np.asarray(raw) / window.scale
        ocular = None if lost else ("blendshapes", eyes)
        window.vision.publish_sample(stamp, point, ocular, "lost" if lost else "valid",
                                     None if lost else np.array([angle, 0., 0., 0., 0.]))
        with patch.object(desktop, "monotonic", return_value=stamp):
            window.tick()

    def repeat(raw, angle, count):
        for _ in range(count):
            step(raw, angle)

    def unlock(raw, angle):
        fixed = sink.cursor.copy()
        for i in range(1, 8):
            step(raw, angle)
            if window.fixation.anchor is None:
                assert np.linalg.norm(sink.cursor-fixed) <= 9
                return i * 1000/30
            np.testing.assert_array_equal(sink.cursor, fixed)
        raise AssertionError("El movimiento intencional no liberó el ancla.")

    try:
        assert window.fixation_active() and not window.fixation_drag.isChecked()
        step([780, 420])
        with patch.object(desktop, "monotonic", return_value=frame/30):
            window.toggle()
        repeat([780, 420], 0., 24)
        for x in range(800, 981, 20):
            step([x, 500], (x-780) / 80)
        target = np.array([980, 500])
        repeat(target, 2.5, 30)
        assert window.fixation.state == "ANCLADO"
        anchor = sink.cursor.copy()
        assert np.linalg.norm(anchor-target) <= 10
        window.change_desktop_clicks(True)
        starts_at = len(positions)
        rng = np.random.default_rng(720)
        for i in range(120):
            raw = target + [i*.33, 0] + rng.normal(0, 1.1, 2)
            if i == 44:
                raw += [175, -130]
            step(raw, 2.5)
        np.testing.assert_array_equal(positions[starts_at:], np.tile(anchor, (120, 1)))
        assert sink.events == ["left"]
        blink_start = len(positions)
        for _ in range(4):
            step([1600, 50], 10., eyes=(.9, .93))
        assert len(positions) == blink_start
        repeat(target, 2.5, 33)
        np.testing.assert_array_equal(sink.cursor, anchor)
        assert sink.events == ["left"] and window.fixation.state == "ANCLADO"

        window.change_desktop_clicks(False)
        rapid_ms = unlock([1050, 550], 4.5)
        assert rapid_ms <= 170
        repeat([1050, 550], 4.5, 45)
        assert window.fixation.state == "ANCLADO"
        fine_start = sink.cursor.copy()
        fine_ms = unlock([1059, 550], 5.0)
        assert fine_ms <= 170
        repeat([1059, 550], 5.0, 25)
        assert sink.cursor[0] >= fine_start[0]+4

        repeat([2, 540], -9, 180)
        assert window.fixation.state == "ANCLADO" and 0 < sink.cursor[0] <= 6
        unlock([0, 540], -9.5)
        repeat([0, 540], -9.5, 45)
        assert sink.cursor[0] == 0

        for goal, angle in (([0, 0], -10), ([1919, 0], 10), ([1919, 1079], 15),
                            ([0, 1079], -15), ([0, 580], -8), ([900, 1079], 8)):
            repeat(goal, angle, 180)
            border = (np.array(goal) == 0) | (np.array(goal) == window.scale)
            np.testing.assert_array_equal(sink.cursor[border], np.asarray(goal)[border])
            assert window.fixation.state == "ANCLADO"

        window.actions.dragging = True
        step([900, 1079], 8)
        assert window.fixation.anchor is None and not window.fixation_active()
        window.actions.dragging = False
        repeat([900, 1079], 8, 25)
        step([900, 1079], lost=True)
        assert not window.armed and window.fixation.anchor is None
        window.absolute_strength.setCurrentText("Suave anterior")
        assert not window.fixation_active()
        print(f"Anclaje: 0 px durante 120 muestras, un clic de permanencia, salida rápida {rapid_ms:.0f} ms, "
              f"reajuste de 9 px en {fine_ms:.0f} ms, 6/6 destinos de borde.")
    finally:
        window.close()
        window.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


if __name__ == "__main__":
    main()
