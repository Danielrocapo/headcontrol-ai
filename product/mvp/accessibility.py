"""Acciones de permanencia y barra flotante no activable."""

from time import monotonic

import numpy as np
from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QPushButton, QWidget


class Dwell:
    def __init__(self, seconds=.9, radius=25):
        self.seconds, self.radius = seconds, radius
        self.reset()

    def reset(self):
        self.anchor, self.start, self.latched = None, None, False

    def suspend(self):
        if self.latched:
            self.start = None
        else:
            self.reset()

    def update(self, position, now):
        if self.anchor is None or np.linalg.norm(position - self.anchor) > self.radius:
            self.anchor, self.start, self.latched = position.copy(), now, False
        if not self.latched and now - self.start >= self.seconds:
            self.latched = True
            return True
        return False

    def progress(self, now):
        return 0. if self.start is None else min(1., (now - self.start) / self.seconds)


class DesktopActions:
    def __init__(self):
        self.mode = "left"
        self.dragging = False
        self.zone, self.zone_since, self.last_scroll = 0, 0., None

    def stop_scroll(self):
        self.zone, self.last_scroll = 0, None

    def release(self, sink):
        self.stop_scroll()
        if self.dragging:
            sink.mouse_up()
            self.dragging = False

    def select(self, mode, sink):
        self.release(sink)
        self.mode = mode

    def activate(self, sink):
        if self.mode == "drag":
            if self.dragging:
                self.release(sink)
                self.mode = "left"
            else:
                self.dragging = True
                sink.mouse_down()
        elif self.mode == "double":
            sink.double_click()
            self.mode = "left"
        else:
            sink.click(self.mode)
            self.mode = "left"

    def scroll(self, zone, now, sink):
        if not zone:
            self.stop_scroll()
            return
        if zone != self.zone:
            self.zone, self.zone_since, self.last_scroll = zone, now, None
        if now - self.zone_since >= .18 and (self.last_scroll is None or now - self.last_scroll >= .13):
            sink.wheel(zone)
            self.last_scroll = now


NO_FOCUS = Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | (
    Qt.WindowType.WindowDoesNotAcceptFocus)


class AccessBar(QWidget):
    def __init__(self, on_action):
        super().__init__(None, NO_FOCUS)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowTitle("HeadControl AI · Accesibilidad")
        self.buttons = {}
        layout = QGridLayout(self)
        layout.setContentsMargins(7, 5, 7, 6)
        layout.setSpacing(4)
        self.hint = QLabel("Cursor pausado")
        layout.addWidget(self.hint, 0, 0, 1, 3)
        self.progress_text = QLabel("")
        self.progress_text.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.progress_text, 0, 3)
        for i, (key, label) in enumerate((
            ("left", "Izquierdo"), ("right", "Derecho"), ("double", "Doble clic"), ("drag", "Arrastrar"),
            ("scroll", "Scroll"), ("keyboard", "Teclado"), ("clicks", "Sin clics"), ("pause", "Reanudar"),
        )):
            button = QPushButton(label)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setMinimumSize(87, 29)
            button.clicked.connect(lambda _=False, action=key: on_action(action))
            self.buttons[key] = button
            layout.addWidget(button, 1 + i//4, i%4)
        self.setStyleSheet(
            "QWidget{background:#132538;color:#eef6ff;font-size:12px;}"
            "QPushButton{background:#203d55;border:1px solid #5886a3;border-radius:5px;padding:3px;}"
            "QPushButton[active=true]{background:#285a57;border:1px solid #83e0c0;}"
            "QPushButton[hovered=true]{border:2px solid #f1cb67;}"
        )

    @staticmethod
    def cursor_point():
        return QCursor.pos()

    @staticmethod
    def virtual_point(raw):
        screen = QApplication.primaryScreen().geometry()
        return QPoint(screen.x() + round(np.clip(raw[0], 0, 1) * (screen.width()-1)),
                      screen.y() + round(np.clip(raw[1], 0, 1) * (screen.height()-1)))

    @staticmethod
    def scroll_zone(point):
        screen = QApplication.primaryScreen().geometry()
        if not screen.contains(point):
            return 0
        relative = (point.y() - screen.y()) / screen.height()
        return 1 if relative <= .16 else -1 if relative >= .84 else 0

    def place(self, keyboard=False):
        self.adjustSize()
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.center().x() - self.width()//2,
                  screen.y() + round((.20 if keyboard else .73) * (screen.height() - self.height())))

    def contains(self, point):
        return self.isVisible() and self.rect().contains(self.mapFromGlobal(point))

    def target(self, point):
        if not self.contains(point):
            return None
        local = self.mapFromGlobal(point)
        return next((name for name, button in self.buttons.items() if button.geometry().contains(local)), None)

    def render(self, armed, enabled, actions, hovered):
        self.buttons["pause"].setText("Pausar" if armed else "Reanudar")
        self.buttons["clicks"].setText("Clics activos" if enabled else "Sin clics")
        self.buttons["drag"].setText("Soltar" if actions.dragging else "Arrastrar")
        mode = {"left": "Clic izquierdo", "right": "Próximo clic derecho", "double": "Próximo doble clic",
                "drag": "Sujeta o suelta con permanencia",
                "scroll": "Scroll · arriba / abajo de la pantalla"}[actions.mode]
        self.hint.setText((mode if enabled else "Sin clics sobre aplicaciones") if armed else
                          "Pausado · apunta a Reanudar con la cabeza")
        for name, button in self.buttons.items():
            props = {"hovered": hovered == name, "active": enabled and actions.mode == name}
            changed = False
            for key, value in props.items():
                if button.property(key) != value:
                    button.setProperty(key, value)
                    changed = True
            if changed:
                button.style().unpolish(button)
                button.style().polish(button)

    def show_progress(self, percent):
        self.progress_text.setText(f"{percent} %" if percent >= 0 else "")


class DwellRing(QWidget):
    progress_changed = Signal(int)

    def __init__(self, keep_on_top):
        super().__init__(None, NO_FOCUS | Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(36, 36)
        self.value = 0.
        self.keep_on_top = keep_on_top
        self.last_raise = float("-inf")
        self.last_percent = -1

    def track(self, point, progress):
        self.value = max(0., min(1., progress))
        percent = round(self.value * 100)
        if self.last_percent != percent:
            self.last_percent = percent
            self.progress_changed.emit(percent)
        self.move(point - QPoint(self.width()//2, self.height()//2))
        if not self.isVisible():
            self.show()
            self.last_raise = float("-inf")
        now = monotonic()
        if QApplication.platformName() == "windows" and now - self.last_raise >= .15:
            self.keep_on_top(int(self.winId()))
            self.last_raise = now
        self.update()

    def hideEvent(self, event):
        self.last_percent = -1
        self.progress_changed.emit(-1)
        super().hideEvent(event)

    def paintEvent(self, event):
        canvas = QPainter(self)
        canvas.setRenderHint(QPainter.RenderHint.Antialiasing)
        circle = QRectF(4, 4, 28, 28)
        canvas.setPen(QPen(QColor("#182b3d"), 5))
        canvas.drawEllipse(circle)
        canvas.setPen(QPen(QColor("#f4ca59"), 3))
        canvas.drawArc(circle, 90*16, round(-360*16*self.value))
