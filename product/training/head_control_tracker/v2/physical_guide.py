"""Esquemas de movimiento para la práctica de PHYSICAL_CHECK."""

from functools import lru_cache
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BACKGROUND = (33, 39, 49)
OUTLINE = (143, 157, 178)
ACTIVE = (95, 225, 255)

INSTRUCTIONS = {
    "translation_x": "Desplaza toda tu cabeza hacia la izquierda o derecha, como si deslizaras la cabeza lateralmente. "
                     "Mantén la nariz apuntando hacia la cámara. No gires la cara.",
    "translation_y": "Desplaza la cabeza hacia arriba o abajo manteniendo la cara orientada hacia la cámara. "
                     "No levantes ni bajes el mentón deliberadamente.",
    "rotation_x": "Gira la cabeza hacia la izquierda o derecha, como cuando dices que no. "
                  "Intenta mantener el centro de tu rostro cerca de su posición inicial.",
    "rotation_y": "Levanta y baja suavemente el mentón, como cuando asientes. "
                  "Intenta mantener el centro de tu rostro cerca de su posición inicial.",
    "combined": "Ahora mueve la cabeza naturalmente combinando desplazamiento y giro. "
                "No intentes separar artificialmente los movimientos.",
    "neutral": "Mantén el rostro quieto, orientado hacia la cámara, en una postura cómoda.",
}
DIRECTIONS = {
    "TX1": "Izquierda", "TX2": "Derecha", "TY1": "Arriba", "TY2": "Abajo",
    "RX1": "Izquierda", "RX2": "Derecha", "RY1": "Arriba", "RY2": "Abajo",
    "C1": "Combinado libre", "C2": "Combinado hacia el otro lado",
}


@lru_cache(maxsize=20)
def text_strip(text):
    try:
        font = ImageFont.truetype("segoeui.ttf", 22)
    except OSError:
        font = ImageFont.load_default(size=22)
    image = Image.new("RGB", (1130, 76), (32, 27, 24))
    canvas = ImageDraw.Draw(image)
    for i, line in enumerate(textwrap.wrap(text, width=96)):
        canvas.text((0, 25 * i), line, font=font, fill=(218, 225, 236))
    return np.array(image)[:, :, ::-1]


def face(canvas, x, y, color, yaw=0, pitch=0):
    cv2.ellipse(canvas, (x, y), (26, 34), 0, 0, 360, color, 2, cv2.LINE_AA)
    for shift in (-10, 10):
        cv2.circle(canvas, (x + shift + yaw // 2, y - 7 + pitch), 2, color, -1, cv2.LINE_AA)
    nose = np.array([(x + yaw, y - 2 + pitch), (x + yaw * 2, y + 8 + pitch),
                     (x + yaw - 4, y + 8 + pitch)], dtype=np.int32)
    cv2.polylines(canvas, [nose], False, color, 2, cv2.LINE_AA)
    cv2.line(canvas, (x - 8 + yaw, y + 18 + pitch), (x + 8 + yaw, y + 18 + pitch), color, 2, cv2.LINE_AA)


def arrow(canvas, start, end, color):
    cv2.arrowedLine(canvas, start, end, color, 2, cv2.LINE_AA, tipLength=.24)


@lru_cache(maxsize=16)
def diagram(condition, step):
    canvas = np.full((236, 620, 3), BACKGROUND, dtype=np.uint8)
    side = -1 if step.endswith("1") else 1
    if condition == "translation_x":
        for direction in (-1, 0, 1):
            face(canvas, 310 + 124 * direction, 105, ACTIVE if direction == side else OUTLINE)
        for direction in (-1, 1):
            arrow(canvas, (310 + 35 * direction, 105), (310 + 87 * direction, 105),
                  ACTIVE if direction == side else OUTLINE)
        caption = "Cambia la posicion de toda la cabeza. La nariz sigue al frente."
    elif condition == "translation_y":
        for direction in (-1, 0, 1):
            # El perfil frontal permanece igual en las tres alturas.
            face(canvas, 310, 100 + 67 * direction, ACTIVE if direction == side else OUTLINE)
        for direction in (-1, 1):
            arrow(canvas, (377, 100 + 12 * direction), (377, 100 + 67 * direction),
                  ACTIVE if direction == side else OUTLINE)
        caption = "Sube o baja toda la cabeza. Conserva la orientacion."
    elif condition.startswith("rotation"):
        for divider in (248, 372):
            cv2.line(canvas, (divider, 47), (divider, 172), (58, 67, 82), 1, cv2.LINE_AA)
        for direction in (-1, 0, 1):
            x = 310 + 124 * direction
            color = ACTIVE if direction == side else OUTLINE
            cv2.drawMarker(canvas, (x, 105), OUTLINE, cv2.MARKER_CROSS, 74, 1, cv2.LINE_AA)
            face(canvas, x, 105, color, yaw=8 * direction if condition == "rotation_x" else 0,
                 pitch=9 * direction if condition == "rotation_y" else 0)
        x = 310 + 124 * side
        if condition == "rotation_x":
            cv2.ellipse(canvas, (x, 82), (37, 18), 0, 210, 330, ACTIVE, 2, cv2.LINE_AA)
            arrow(canvas, (x + 26 * side, 69), (x + 34 * side, 75), ACTIVE)
            caption = "Gira como al decir que no. La cruz marca el centro de referencia."
        else:
            cv2.ellipse(canvas, (x, 105), (42, 46), 0, -38, 38, ACTIVE, 2, cv2.LINE_AA)
            arrow(canvas, (x + 38, 105 + 21 * side), (x + 32, 105 + 30 * side), ACTIVE)
            caption = "Inclina como al asentir. La cruz marca el centro de referencia."
    elif condition == "combined":
        face(canvas, 310, 105, OUTLINE)
        face(canvas, 310 + 115 * side, 95, ACTIVE, yaw=8 * side)
        arrow(canvas, (310 + 36 * side, 102), (310 + 76 * side, 96), ACTIVE)
        caption = "Desplaza y gira la cabeza con un gesto natural."
    else:
        cv2.drawMarker(canvas, (310, 105), OUTLINE, cv2.MARKER_CROSS, 78, 1, cv2.LINE_AA)
        face(canvas, 310, 105, ACTIVE)
        caption = "Vuelve al centro con el cuello relajado."
    cv2.putText(canvas, caption, (16, 219), cv2.FONT_HERSHEY_SIMPLEX, .50, OUTLINE, 1, cv2.LINE_AA)
    return canvas
