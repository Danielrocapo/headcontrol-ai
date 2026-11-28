"""Ubicación del teclado de Windows y orden nativo del indicador."""

import ctypes
from ctypes import wintypes as wt


SW_SHOWNOACTIVATE = 4
SWP_NOACTIVATE = 0x10
SWP_NOSIZE = 0x01
SWP_NOMOVE = 0x02
SWP_NOZORDER = 0x04
SWP_NOOWNERZORDER = 0x0200
HWND_TOPMOST = -1


class GuiThreadInfo(ctypes.Structure):
    _fields_ = [("size", wt.DWORD), ("flags", wt.DWORD), *((name, wt.HWND) for name in (
        "active", "focus", "capture", "menu", "move", "caret")), ("caret_rect", wt.RECT)]


class MonitorInfo(ctypes.Structure):
    _fields_ = [("size", wt.DWORD), ("monitor", wt.RECT), ("work", wt.RECT), ("flags", wt.DWORD)]


def user32_api():
    api = ctypes.WinDLL("user32", use_last_error=True)
    for name, args, result in (
        ("FindWindowW", (wt.LPCWSTR, wt.LPCWSTR), wt.HWND),
        ("GetForegroundWindow", (), wt.HWND),
        ("SetForegroundWindow", (wt.HWND,), wt.BOOL),
        ("GetAncestor", (wt.HWND, wt.UINT), wt.HWND),
        ("GetGUIThreadInfo", (wt.DWORD, ctypes.POINTER(GuiThreadInfo)), wt.BOOL),
        ("ClientToScreen", (wt.HWND, ctypes.POINTER(wt.POINT)), wt.BOOL),
        ("GetWindowRect", (wt.HWND, ctypes.POINTER(wt.RECT)), wt.BOOL),
        ("MonitorFromPoint", (wt.POINT, wt.DWORD), wt.HANDLE),
        ("GetMonitorInfoW", (wt.HANDLE, ctypes.POINTER(MonitorInfo)), wt.BOOL),
        ("IsWindowVisible", (wt.HWND,), wt.BOOL),
        ("IsIconic", (wt.HWND,), wt.BOOL),
        ("ShowWindow", (wt.HWND, ctypes.c_int), wt.BOOL),
        ("SetWindowPos", (wt.HWND, wt.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wt.UINT), wt.BOOL),
    ):
        fn = getattr(api, name)
        fn.argtypes, fn.restype = args, result
    return api


def rect_tuple(rect):
    return rect.left, rect.top, rect.right, rect.bottom


def keyboard_position(work, size, field):
    left, top, right, bottom = work
    width, height = size
    x = left + max(0, (right - left - width) // 2)
    y = max(top, bottom - height - 8)
    if field is not None:
        l, t, r, b = field
        if l < x + width and x < r and t < y + height and y < b:
            # Si el campo ocupa la parte inferior, conserva la mayor separación posible.
            above = t - height - 16
            if above >= top:
                y = above
    return x, y


class WindowsDesktop:
    def __init__(self):
        self.api = user32_api()

    def keyboard_handle(self):
        return self.api.FindWindowW("OSKMainClass", None)

    def foreground(self):
        return self.api.GetForegroundWindow()

    def work_area(self):
        screen = self.api.MonitorFromPoint(wt.POINT(0, 0), 1)
        info = MonitorInfo()
        info.size = ctypes.sizeof(info)
        return rect_tuple(info.work) if self.api.GetMonitorInfoW(screen, ctypes.byref(info)) else None

    def rect(self, hwnd):
        rect = wt.RECT()
        return rect_tuple(rect) if self.api.GetWindowRect(hwnd, ctypes.byref(rect)) else None

    def focused_field(self):
        info = GuiThreadInfo()
        info.size = ctypes.sizeof(info)
        if not self.api.GetGUIThreadInfo(0, ctypes.byref(info)):
            return None
        caret = info.caret_rect
        if info.caret and caret.bottom > caret.top:
            point = wt.POINT(caret.left, caret.top)
            if self.api.ClientToScreen(info.caret, ctypes.byref(point)):
                return point.x - 2, point.y - 8, point.x + 3, point.y + caret.bottom - caret.top + 8
        if info.focus:
            rect = self.rect(info.focus)
            if rect is not None and 0 < rect[3] - rect[1] <= 120:
                return rect
        return None

    def place_keyboard(self, hwnd, foreground, field):
        if not self.api.IsWindowVisible(hwnd) or self.api.IsIconic(hwnd):
            self.api.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
        rect, work = self.rect(hwnd), self.work_area()
        placed = False
        if rect is not None and work is not None:
            x, y = keyboard_position(work, (rect[2]-rect[0], rect[3]-rect[1]), field)
            placed = rect[:2] == (x, y) or bool(self.api.SetWindowPos(
                hwnd, None, x, y, 0, 0, SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOZORDER | SWP_NOOWNERZORDER))
            if placed:
                actual = self.rect(hwnd)
                placed = actual is not None and max(abs(actual[0]-x), abs(actual[1]-y)) <= 8
        current = self.foreground()
        if foreground and foreground != hwnd and (current == hwnd or self.api.GetAncestor(current, 3) == hwnd):
            self.api.SetForegroundWindow(foreground)
        return placed

    def keep_indicator_on_top(self, hwnd):
        self.api.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                              SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE | SWP_NOOWNERZORDER)


class KeyboardDock:
    def __init__(self, desktop, schedule, report):
        self.desktop, self.schedule, self.report = desktop, schedule, report
        self.pending = None

    def open(self, launch):
        if self.pending is not None:
            return
        focus, field = self.desktop.foreground(), self.desktop.focused_field()
        hwnd = self.desktop.keyboard_handle()
        if hwnd:
            self.finish(hwnd, focus, field)
            return
        token = object()
        self.pending = token
        try:
            launch()
        except OSError:
            self.pending = None
            raise

        def poll(remaining):
            if self.pending is not token:
                return
            hwnd = self.desktop.keyboard_handle()
            if hwnd:
                self.pending = None
                self.finish(hwnd, focus, field)
            elif remaining > 0:
                self.schedule(100, lambda: poll(remaining-1))
            else:
                self.pending = None
                self.report("Si el teclado se abrió, Windows no expuso su ventana. Puedes arrastrarlo manualmente.")

        poll(45)

    def finish(self, hwnd, focus, field):
        if not self.desktop.place_keyboard(hwnd, focus, field):
            self.report("Windows conservó la posición del teclado. Puedes arrastrarlo manualmente.")

    def cancel(self):
        self.pending = None
