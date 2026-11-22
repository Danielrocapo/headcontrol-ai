"""Descarga una sola vez el modelo oficial de MediaPipe Face Landmarker."""
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DESTINATION = Path(__file__).resolve().parent / "assets" / "face_landmarker.task"


def main():
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists() and DESTINATION.stat().st_size > 1_000_000:
        print("Modelo ya disponible:", DESTINATION)
        return
    print("Descargando modelo oficial de MediaPipe...")
    temporary = DESTINATION.with_suffix(".download")
    try:
        with urlopen(Request(URL, headers={"User-Agent": "HeadControlAI/2.0"}), timeout=40) as response:
            with temporary.open("wb") as file:
                while True:
                    block = response.read(1024 * 256)
                    if not block:
                        break
                    file.write(block)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError("La descarga es demasiado pequeña o está incompleta.")
        temporary.replace(DESTINATION)
        print("Listo. Modelo guardado en:", DESTINATION)
        print("Las siguientes ejecuciones no necesitan Internet.")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
