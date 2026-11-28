"""Simula los controles y la permanencia sin cámara ni eventos de Windows."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication

import tracker_windows as desktop
from calibrate_user import Tracker, atomic_json, profile_matrix, read_json
from features9 import VERSION, neutral_reference
from coverage2d import ANCHORS, CoverageMap
import reach
from validate_cursor import SCENARIOS, target_at
from verify_tracker9 import synthetic_row


def verify_bounds():
    for width, height in ((1920, 1080), (2560, 1440), (1536, 864)):
        scale = np.array([width - 1, height - 1])
        for x, y in ((0, 0), (0, 1), (1, 0), (1, 1), (.25, .75)):
            control = desktop.Safety(.12)
            location = scale // 2
            assert control.step(np.array([x, y]), location, 0., True, scale) is None
            for step in range(1, 250):
                out = control.step(np.array([x, y]), location, step / 30, True, scale)
                if out is not None:
                    location = out
            np.testing.assert_array_equal(location, np.rint(np.array([x, y]) * scale))
            assert not control.trace.limited
            assert not control.trace.clipped.any()
        control = desktop.Safety(0.)
        location = scale // 2
        beyond = np.array([-.25, 1.3])
        assert control.step(beyond, location, 0., True, scale) is None
        out = control.step(beyond, location, .41, True, scale)
        assert control.trace.limited and control.trace.clipped.tolist() == [True, True]
        np.testing.assert_array_equal(control.trace.target_px, [0, height - 1])
        assert np.linalg.norm(out - location) <= 41
        assert control.step(beyond, np.array([-1, 0]), .44, True, scale) is None
        assert control.trace is None


def verify_panel(window):
    assert window.absolute_strength.currentText() == "Suave"
    window.absolute_strength.setCurrentText("Desactivado")
    window.face_rearm = False  # Escenario independiente del rearme tras pérdida.
    class Win32:
        sent = None

        def GetAsyncKeyState(self, key):
            return 0

        def GetSystemMetrics(self, metric):
            return (1920, 1080)[metric]

        def GetDpiForSystem(self):
            return 120

        def GetClipCursor(self, pointer):
            pointer._obj.left, pointer._obj.top, pointer._obj.right, pointer._obj.bottom = 20, 10, 1920, 1080
            return 1

        def SetCursorPos(self, x, y):
            self.sent = np.array([x, y])
            return 1

    api = Win32()
    window.api = api
    window.scale = np.array([1919, 1079])
    window.sink.position = lambda: np.maximum([20, 10], api.sent) if api.sent is not None else np.array([21, 11])
    window.change_smoothing(0)
    window.diagnostic_toggle.setChecked(True)
    window.vision.latest = (1., np.array([-.02, -.03]))
    with patch.object(desktop, "monotonic", return_value=1.):
        window.toggle()
        window.tick()
    assert api.sent is None
    window.vision.latest = (1.42, np.array([-.02, -.03]))
    with patch.object(desktop, "monotonic", return_value=1.42):
        window.tick()
    np.testing.assert_array_equal(api.sent, [0, 0])
    text = window.diagnostic.toPlainText()
    for expected in ("DPI 125%", "fuera de [0, 1]: X, Y", "clipping: X, Y", "Limitador: inactivo",
                     "Enviado a Windows: X 0, Y 0 px", "Cursor real: X 20, Y 10 px",
                     "fuera del rectángulo permitido por Windows"):
        assert expected in text


def verify_reach(window, identity, camera):
    original = {"izquierda": (.07883, .5), "derecha": (.91, .5), "neutral": (.49, .51),
                "arriba": (.49, .06538), "abajo": (.49, .87)}
    mapping = reach.ReachMap.from_points(original)
    center = np.asarray(original["neutral"])
    np.testing.assert_allclose(mapping.apply(center), [.5, .5])
    for x in (0, 1):
        for y in (0, 1):
            raw_corner = mapping.knots[[0, 1], [2*x, 2*y]]
            expected = np.array([x, y])
            np.testing.assert_allclose(mapping.apply(raw_corner), expected)
            for fraction in np.linspace(0, 1, 11):
                np.testing.assert_allclose(mapping.apply(center + fraction * (raw_corner - center)),
                                           .5 + fraction * (expected - .5), atol=1e-12)
    samples = center + np.linspace(-.02, .02, 21)[:, None] * np.array([1., -1.])
    corrected = np.array([mapping.apply(p) for p in samples])
    assert np.all(np.linalg.norm(np.diff(corrected, axis=0), axis=1) <=
                  reach.MAX_GAIN * np.linalg.norm(np.diff(samples, axis=0), axis=1) + 1e-12)
    for malformed in ([[.4, .5, .9], [.1, .5, .9]], [[.7, .5, .9], [.1, .5, .9]]):
        try:
            reach.ReachMap(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("Se aceptó un alcance inválido.")
    with TemporaryDirectory() as directory, patch.object(reach, "PROFILES", Path(directory)):
        profile = Path(directory) / "profile.json"
        saved_neutral = dict(raw=[179., 2., 3., 0., 0., 0., 0.], center_px=[620., 330.], interocular_px=100.)
        atomic_json(profile, dict(schema="tracker-reach-v1", feature_contract=VERSION, model=identity,
                                  screen_size=[1920, 1080], camera=camera, knots=mapping.knots.tolist(),
                                  neutral_reference=saved_neutral))
        reach.set_active(profile)
        loaded = reach.active_reach(identity, (1920, 1080), camera)
        np.testing.assert_allclose(loaded.knots, mapping.knots)
        window.pause()
        window.vision.reach = loaded
        current = dict(raw=[-179., 2., 3., 0., 0., 0., 0.], center_px=[624., 328.], interocular_px=105.)
        window.vision.reach_neutral = (current, np.array([.52, .49]))
        angle, delta, span = reach.compare_neutral(saved_neutral, current)
        np.testing.assert_array_equal(angle, [2, 0, 0])
        np.testing.assert_array_equal(delta, [4, -2])
        np.testing.assert_allclose(span, 5.)
        checks = (
            (mapping.knots[:, 0], "izquierda, arriba"),
            ([mapping.knots[0, 0], .3], "izquierda"),
            ([.3, mapping.knots[1, 0]], "arriba"),
            ([.3, .3], "ninguno"),
        )
        for point, sides in checks:
            assert f"Bordes que pide esa salida: {sides}" in window.reach_diagnostics(np.asarray(point))
        lines = "\n".join(window.reach_diagnostics(center))
        for label in ("Perfil cargado y compatible", "máximo 2×", "Entrada actual ALCANCE",
                      "Salida ALCANCE", "Neutral RAW de la sesión", "yaw +2.00°"):
            assert label in lines
        window.mode.model().item(2).setEnabled(True)
        window.mode.setCurrentIndex(2)
        window.sink.position = lambda: np.array([8, 8])
        window.smoothing.setValue(0)
        window.vision.latest = (10., mapping.knots[:, 0])
        with patch.object(desktop, "monotonic", return_value=10.):
            window.toggle()
            window.tick()
        window.vision.latest = (10.42, mapping.knots[:, 0])
        with patch.object(desktop, "monotonic", return_value=10.42):
            window.tick()
        np.testing.assert_array_equal(window.api.sent, [0, 0])
        assert "Último envío · Alcance" in window.diagnostic.toPlainText()
        assert "Bordes que pide esa salida: izquierda, arriba" in window.diagnostic.toPlainText()
        window.reset_reach()
        assert window.mode.currentIndex() == 0 and not window.armed
        assert reach.active_reach(identity, (1920, 1080), camera) is None and profile.is_file()


def verify_coverage_mode(window):
    anchors = {key: .08 + .84 * np.asarray(xy) for key, xy in ANCHORS.items()}
    window.vision.coverage = CoverageMap(anchors, (1920, 1080))
    window.mode.model().item(3).setEnabled(True)
    window.mode.setCurrentIndex(3)
    window.sink.position = lambda: np.array([8, 8])
    window.vision.latest = (20., anchors["superior izquierda"])
    with patch.object(desktop, "monotonic", return_value=20.):
        window.toggle()
        window.tick()
    window.vision.latest = (20.42, anchors["superior izquierda"])
    with patch.object(desktop, "monotonic", return_value=20.42):
        window.tick()
    assert window.mode.currentText() == "Cobertura 2D"
    np.testing.assert_array_equal(window.api.sent, [0, 0])
    assert "Último envío · Cobertura 2D" in window.diagnostic.toPlainText()
    window.mode.setCurrentIndex(0)
    assert not window.armed


def verify_relative_mode(window):
    class FakeWindows:
        cursor = np.array([900, 500])
        keys = set()
        sends = 0

        def GetAsyncKeyState(self, key):
            return 0x8000 if key in self.keys else 0

        def SetCursorPos(self, x, y):
            self.cursor = np.array([x, y])
            self.sends += 1
            return 1

    api = FakeWindows()
    window.api = api
    window.sink.position = lambda: api.cursor
    window.sink.click = lambda _: (_ for _ in ()).throw(AssertionError("Clic inesperado"))
    window.diagnostic_toggle.setChecked(False)
    window.control.setCurrentIndex(1)
    assert not window.armed and not window.mode.isEnabled() and not window.clicks.isChecked()
    neutral = np.array([.48, .51])
    movement = neutral + [.1, -.1]
    window.vision.neutral_prediction = neutral

    def tick(stamp, raw):
        window.vision.latest = (stamp, raw)
        with patch.object(desktop, "monotonic", return_value=stamp):
            window.tick()

    with patch.object(desktop, "monotonic", return_value=10.):
        window.vision.latest = (10., movement)
        window.toggle()
    tick(10., movement)
    assert not api.sends
    tick(10.03, neutral)
    tick(10.06, movement)
    assert api.sends == 1 and np.all((api.cursor - [900, 500]) * [1, -1] > 0)
    tick(10.06, movement)
    assert api.sends == 1
    api.keys.add(0x77)
    tick(10.09, movement)
    assert not window.armed and api.sends == 1
    api.keys.clear()
    tick(10.12, neutral)

    with patch.object(desktop, "monotonic", return_value=10.12):
        window.toggle()
    assert window.armed
    window.vision.losses += 1
    tick(10.15, neutral)
    assert not window.armed and api.sends == 1

    with patch.object(desktop, "monotonic", return_value=10.15):
        window.begin_recenter()
    assert not window.arm.isEnabled()
    updated = neutral + [.03, -.04]
    for frame in range(1, 56):
        tick(10.15 + frame/30, updated)
    assert not window.armed and window.arm.isEnabled() and api.sends == 1
    np.testing.assert_array_equal(window.current_neutral(), updated)
    with patch.object(desktop, "monotonic", return_value=12.):
        window.begin_recenter()
    api.keys.add(0x1B)
    tick(12.03, neutral)
    np.testing.assert_array_equal(window.current_neutral(), updated)
    assert window.recentering is None and not window.armed
    window.control.setCurrentIndex(0)
    assert window.mode.isEnabled()


def verify_model_switch(window, original, profile, index, runtime):
    candidate = Tracker(runtime)
    assert candidate.identity["onnx_sha256"] != original.identity["onnx_sha256"]
    assert candidate.identity["normalizer_sha256"] == original.identity["normalizer_sha256"]
    try:
        profile_matrix(desktop.PROFILE, candidate.identity, tuple(profile["screen_size"]), profile["camera"])
    except ValueError as exc:
        assert "otro modelo" in str(exc)
    else:
        raise AssertionError("El candidato aceptó el perfil afín del original.")
    row = synthetic_row(dx=25, dy=-15, yaw=8)
    neutral = neutral_reference([synthetic_row()] * 16)
    assert np.linalg.norm(original.predict(row, neutral) - candidate.predict(row, neutral)) > .001

    window.clicks.setChecked(True)
    window.vision.latest = (10., np.array([.4, .6]))
    window.relative_neutral = np.array([.5, .5])
    window.vision.neutral_prediction = np.array([.5, .5])
    window.armed = True
    window.start_camera.setEnabled(False)
    with patch.object(window.vision, "requestInterruption") as stop, patch.object(window.vision, "start") as start:
        window.model.setCurrentIndex(index)
        stop.assert_called_once()
        assert window.model_switch_pending and not window.armed and not window.clicks.isChecked()
        assert window.vision.latest is None and window.relative_neutral is None
        assert window.mode.currentIndex() == 0 and not window.mode.model().item(1).isEnabled()
        # Una lectura en vuelo del modelo previo no puede rearmar el cursor.
        window.vision.latest = (10., np.array([.4, .6]))
        with patch.object(desktop, "monotonic", return_value=10.):
            window.toggle()
        assert not window.armed
        window.start()
        start.assert_not_called()
        window.vision.latest = None
        window.finished()
        assert not window.model_switch_pending and window.current_neutral() is None
        assert "neutral nuevo" in window.status.text()
        window.start()
        start.assert_called_once()
        assert window.vision.runtime == runtime
        assert window.control.currentIndex() == 0 and window.mode.currentIndex() == 0
        assert not window.armed and not window.clicks.isChecked()
        window.finished()
        window.model.setCurrentIndex(0)
        window.start()
        assert start.call_count == 2 and window.vision.runtime == desktop.RUNTIME
        window.finished()


def replay(milliseconds):
    scale = np.array([1919, 1079])
    moving = next(s for s in SCENARIOS if s[0] == "movil")
    control = desktop.Safety(milliseconds / 1000)
    position = target_at(moving, 0, scale)
    errors, maximum = [], 0
    for tick in range(270):
        now = tick / 30
        point = target_at(moving, now - 1, scale)
        predicted = point + np.array([12 * (-1)**tick, 7 * (-1)**tick])
        output = control.step(predicted / scale, position, now, True, scale)
        if output is not None:
            maximum = max(maximum, np.linalg.norm(output - position))
            position = output
        if tick >= 30:
            errors.append(np.linalg.norm(position - point))
    assert maximum <= 41
    stationary = np.array([960, 540])
    position = stationary.copy()
    control.reset()
    jitter = []
    for tick in range(120):
        now = tick / 30
        prediction = stationary + np.array([12 * (-1)**tick, 7 * (-1)**tick])
        output = control.step(prediction / scale, position, now, True, scale)
        if output is not None:
            position = output
        if tick >= 60:
            jitter.append(np.linalg.norm(position - stationary))
    return np.mean(errors), np.percentile(errors, 90), maximum, np.percentile(jitter, 90)


def verify():
    raw = Tracker()
    camera = dict(source_sha256="synthetic", face_task_sha256="synthetic", camera_index=0,
                  camera_width=1280, camera_height=720, mirrored=True)
    profile = dict(schema=1, screen_size=[1920, 1080], feature_contract=VERSION,
                   model=raw.identity, camera=camera, enabled=True,
                   matrix=[[1., 0.], [0., 1.], [0., 0.]],
                   metrics={"original": {"mean_px": 2.}, "corrected": {"mean_px": 1.}})
    with TemporaryDirectory() as directory:
        path = Path(directory) / "profile.json"
        atomic_json(path, profile)
        with patch.object(desktop, "PROFILE", path):
            verify_integration(raw, profile)


def verify_integration(raw, profile):
    verify_bounds()
    assert profile_matrix(desktop.PROFILE, raw.identity, tuple(profile["screen_size"]), profile["camera"]) is not None
    row = synthetic_row()
    ref = neutral_reference([row] * 16)
    original = raw.predict(row, ref)
    displaced = raw.predict(synthetic_row(dx=36, dy=14, yaw=10), ref)
    assert np.isfinite(original).all() and np.linalg.norm(original - displaced) > .02
    app = QApplication(sys.argv[:1])
    window = desktop.Window()
    try:
        assert not window.vision.isRunning() and not window.armed
        assert window.model.currentIndex() == window.control.currentIndex() == window.mode.currentIndex() == 0
        assert not window.clicks.isChecked()
        window.timer.stop()
        with patch.object(desktop, "monotonic", return_value=1.0):
            window.vision.latest = (1., np.array([.5, .5]))
            window.toggle()
            assert window.armed
            window.vision.losses += 1
            window.tick()
            assert not window.armed
        window.vision.latest = None
        window.toggle()
        assert not window.armed
        verify_panel(window)
        verify_reach(window, raw.identity, profile["camera"])
        verify_coverage_mode(window)
        verify_relative_mode(window)
        assert window.model.count() == 3
        verify_model_switch(window, raw, profile, 1, desktop.GUIDED_RUNTIME)
        verify_model_switch(window, raw, profile, 2, desktop.GUIDED_V2_RUNTIME)
        guided = Tracker(desktop.GUIDED_RUNTIME)
        v2 = Tracker(desktop.GUIDED_V2_RUNTIME)
        assert guided.identity["onnx_sha256"] != v2.identity["onnx_sha256"]
        assert np.linalg.norm(guided.predict(row, ref) - v2.predict(row, ref)) > .001
    finally:
        window.close()
        window.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    dwell = desktop.Dwell()
    location = np.array([20., 100.])
    assert not dwell.update(location, 0)
    assert dwell.update(location, .95)
    assert not dwell.update(location, 2.)
    assert not dwell.update(location + [50, 0], 2.1)
    assert dwell.update(location + [50, 0], 3.1)
    dwell.reset()
    assert not dwell.update(location, 4.)
    for ms in (0,120,250):
        mean, p90, max_step, jitter = replay(ms)
        print(f"{ms} ms | medio {mean:.1f} px | P90 {p90:.1f} px | paso máximo {max_step:.1f} px | "
              f"P90 en reposo {jitter:.1f} px")


if __name__ == "__main__":
    verify()
