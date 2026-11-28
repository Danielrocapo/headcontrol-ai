"""Adaptador Windows para cursor relativo y clices en la posición vigente."""

import ctypes
from ctypes import wintypes
from os import getenv, startfile
from pathlib import Path

import pyautogui
from desktop_windows import SW_SHOWNOACTIVATE


class MouseInput(ctypes.Structure):
    _fields_ = (("dx", wintypes.LONG), ("dy", wintypes.LONG), ("data", wintypes.DWORD),
                ("flags", wintypes.DWORD), ("time", wintypes.DWORD), ("extra", ctypes.c_size_t))


class Input(ctypes.Structure):
    _fields_ = (("type", wintypes.DWORD), ("mouse", MouseInput))


class WindowsSink:
    position = staticmethod(pyautogui.position)
    moveRel = staticmethod(pyautogui.moveRel)
    scroll = staticmethod(pyautogui.scroll)

    @staticmethod
    def click(button):
        pyautogui.failSafeCheck()
        down, up = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010)}[button]
        WindowsSink.send_mouse((down, up), release=up)

    @staticmethod
    def double_click():
        pyautogui.failSafeCheck()
        WindowsSink.send_mouse((0x0002, 0x0004, 0x0002, 0x0004), release=0x0004)

    @staticmethod
    def mouse_down():
        pyautogui.failSafeCheck()
        WindowsSink.send_mouse((0x0002,), release=0x0004)

    @staticmethod
    def mouse_up():
        WindowsSink.send_mouse((0x0004,))

    @staticmethod
    def wheel(direction):
        WindowsSink.send_mouse((0x0800,), data=(120 * direction) & 0xffffffff)

    @staticmethod
    def keyboard():
        startfile(str(Path(getenv("SystemRoot", r"C:\Windows")) / "System32" / "osk.exe"),
                  show_cmd=SW_SHOWNOACTIVATE)

    @staticmethod
    def send_mouse(flags, release=None, data=0):
        actions = (Input * len(flags))(*(Input(0, MouseInput(flags=flag, data=data)) for flag in flags))
        send = ctypes.WinDLL("user32", use_last_error=True).SendInput
        send.argtypes = (wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int)
        send.restype = wintypes.UINT
        count = send(len(actions), actions, ctypes.sizeof(Input))
        if release is not None and count % 2 and count != len(actions):
            release_event = Input(0, MouseInput(flags=release))
            send(1, ctypes.byref(release_event), ctypes.sizeof(Input))
        if count != len(actions):
            raise OSError(ctypes.get_last_error(), "Windows no aceptó la secuencia de clic.")
