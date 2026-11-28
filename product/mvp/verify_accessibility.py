"""Pruebas Qt con cara, puntero y eventos Windows simulados."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtWidgets import QApplication

import tracker_windows as desktop
from win_sink import WindowsSink


class FakeSink:
    def __init__(self):
        self.cursor = np.array([80, 80])
        self.events = []
        self.fail_release = False

    def position(self):
        return tuple(self.cursor)

    def click(self, button):
        self.events.append(button)

    def double_click(self):
        self.events.append("double")

    def mouse_down(self):
        self.events.append("down")

    def mouse_up(self):
        if self.fail_release:
            raise OSError("fallo simulado")
        self.events.append("up")

    def wheel(self, direction):
        self.events.append(("wheel", direction))

    def keyboard(self):
        self.events.append("keyboard")


def verify_native_adapter():
    with patch.object(WindowsSink, "send_mouse") as send, patch("win_sink.pyautogui.failSafeCheck"):
        WindowsSink.double_click()
        send.assert_called_with((2, 4, 2, 4), release=4)
        WindowsSink.mouse_down()
        send.assert_called_with((2,), release=4)
        WindowsSink.mouse_up()
        send.assert_called_with((4,))
        WindowsSink.wheel(-1)
        send.assert_called_with((0x800,), data=(-120) & 0xffffffff)
    with patch("win_sink.startfile") as launch:
        WindowsSink.keyboard()
        assert launch.call_args.args[0].lower().endswith("\\system32\\osk.exe")
        assert launch.call_args.kwargs["show_cmd"] == 4


def verify():
    verify_native_adapter()
    app = QApplication([])
    window = desktop.Window()
    window.timer.stop()
    window.absolute_strength.setCurrentText("Desactivado")
    window.change_smoothing(0)
    bar, sink = window.access_bar, FakeSink()
    screen = app.primaryScreen().geometry()
    window.scale = np.array([screen.width()-1, screen.height()-1])
    window.sink = sink
    window.keyboard_dock.open = lambda launch: launch()
    keys = set()
    sends = []

    def move(x, y):
        sink.cursor[:] = (x, y)
        sends.append(tuple(sink.cursor))
        return 1

    window.api = SimpleNamespace(GetAsyncKeyState=lambda k: 0x8000 if k in keys else 0, SetCursorPos=move)
    bar.cursor_point = lambda: QPoint(int(sink.cursor[0]), int(sink.cursor[1]))
    bar.place()
    bar.show()
    app.processEvents()
    assert bar.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    assert window.dwell_ring.windowFlags() & Qt.WindowType.WindowTransparentForInput
    assert screen.contains(bar.geometry()) and bar.geometry().top() > screen.top() and bar.geometry().bottom() < screen.bottom()
    assert (window.dwell.seconds, window.dwell.radius) == (.9, 25)
    assert not window.winks.isChecked()
    now = 0.

    def frames(count=1, point=None, eyes=(.03, .03), lost=False):
        nonlocal now
        for _ in range(count):
            now += 1/30
            raw = np.asarray(point if point is not None else sink.cursor) / window.scale
            window.vision.publish_sample(now, None if lost else raw,
                                         None if eyes is None else ("blendshapes", eyes), "valid")
            with patch.object(desktop, "monotonic", return_value=now):
                window.tick()

    def button(name):
        widget = bar.buttons[name]
        p = widget.mapToGlobal(widget.rect().center())
        return p.x(), p.y()

    def hover(name):
        sink.cursor[:] = button(name)
        frames(31)

    def outside(x=80, y=80, count=31):
        sink.cursor[:] = (x, y)
        frames(count)

    try:
        # Rearme sin mover el puntero real mientras está pausado.
        at = button("pause")
        frames(35, point=at)
        assert not window.armed and not sends
        frames(2, point=(80, 80))
        frames(31, point=at)
        assert window.armed
        assert not sink.events
        hover("clicks")
        assert window.clicks.isChecked() and not sink.events
        frames(45)
        assert not sink.events  # La barra nunca hace clic en la app que tiene debajo.

        outside(count=12)
        assert window.dwell_ring.isVisible() and 0 < window.dwell_ring.value < 1
        assert bar.progress_text.text().endswith(" %")
        frames(20)
        assert sink.events == ["left"]
        assert not window.dwell_ring.isVisible() and not bar.progress_text.text()
        frames(3, eyes=(.9, .9))
        frames(50)
        assert sink.events == ["left"]
        sink.cursor[0] += 25
        frames(31)
        assert sink.events == ["left"]
        sink.cursor[0] += 1
        frames(31)
        assert sink.events == ["left", "left"]

        hover("right")
        outside()
        assert sink.events[-1] == "right" and window.actions.mode == "left"
        hover("double")
        outside()
        assert sink.events[-1] == "double" and window.actions.mode == "left"

        hover("drag")
        outside()
        assert sink.events[-1] == "down" and window.actions.dragging
        outside(x=180)
        assert sink.events[-1] == "up" and not window.actions.dragging
        hover("drag")
        outside()
        assert window.actions.dragging
        frames(lost=True, eyes=None)
        assert sink.events[-1] == "up" and not window.actions.dragging and not window.armed

        frames(2, point=(80, 80))
        frames(31, point=button("pause"))
        assert window.armed
        hover("scroll")
        outside(y=15)
        assert sink.events[-1] == ("wheel", 1)
        outside(y=screen.height()-20)
        assert sink.events[-1] == ("wheel", -1)
        n = len(sink.events)
        outside(y=screen.height()//2)
        assert len(sink.events) == n
        hover("keyboard")
        assert sink.events[-1] == "keyboard"
        hover("clicks")
        assert not window.clicks.isChecked() and window.armed
        n = len(sink.events)
        outside(count=45)
        assert len(sink.events) == n
        hover("left")
        outside(count=15)
        assert window.dwell.anchor is not None
        frames(40, eyes=(.9, .9))
        assert window.dwell.anchor is None and len(sink.events) == n
        frames(15)
        assert len(sink.events) == n
        hover("pause")
        assert not window.armed
        frames(40, point=button("pause"))
        assert not window.armed
        frames(2, point=(80, 80))
        frames(31, point=button("pause"))
        assert window.armed

        hover("drag")
        outside()
        assert window.actions.dragging
        keys.add(0x77)
        sink.fail_release = True
        frames()
        assert not window.armed and window.actions.dragging
        sink.fail_release = False
        frames()
        assert sink.events[-1] == "up" and not window.actions.dragging
        keys.clear()
        frames(2, point=(80, 80))
        frames(31, point=button("pause"))
        hover("drag")
        outside()
        assert window.actions.dragging
    finally:
        held = window.actions.dragging
        window.close()
        assert not window.actions.dragging and (not held or sink.events[-1] == "up")
        window.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


if __name__ == "__main__":
    verify()
    print("Barra, permanencia, clics, arrastre, scroll, teclado, pausas y liberación de seguridad verificados.")
