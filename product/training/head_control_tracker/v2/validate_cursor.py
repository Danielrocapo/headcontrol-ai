"""Prueba pareada RAW/CALIBRATED del puntero real, aislada de las sesiones TR."""

import argparse
from contextlib import ExitStack
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import monotonic, perf_counter
from uuid import uuid4

import cv2
import numpy as np

from calibrate_user import BLUE, GREEN, Tracker, atomic_json, calibrate, primary_size, profile_matrix
from calibration_math import Smoother, affine
from capture_tracker import Screen, StopCapture, WINDOW, YELLOW
from cursor_validation_metrics import measure, report
from features9 import Camera9

HERE = Path(__file__).resolve().parent
PROFILE = HERE / "calibration_profiles/20261126T205813Z_32c20684/profile.json"
SPEED_PX_S = 950.
RECOVERY_S = .4
MAX_STEP_PX = 40.
WARM_S = 1.5
SCENARIOS = (
    ("horizontal", 5., (.17,.42), (.83,.42)),
    ("vertical", 5., (.61,.14), (.61,.84)),
    ("diagonal_arriba_izquierda", 5., (.51,.53), (.18,.19)),
    ("diagonal_arriba_derecha", 5., (.51,.53), (.84,.20)),
    ("diagonal_abajo_izquierda", 5., (.51,.53), (.19,.82)),
    ("diagonal_abajo_derecha", 5., (.51,.53), (.81,.81)),
    ("reposo", 10., (.50,.50), (.50,.50)),
    ("movil", 8., (.50,.46), (.50,.46)),
    ("recuperacion", 8., (.58,.47), (.58,.47)),
)


def target_at(scenario, elapsed, scale):
    kind, duration, a, b = scenario
    f = np.clip(elapsed / duration,0,1)
    xy = (np.array([.50 + .29*np.sin(2*np.pi*f),.46 + .21*np.sin(4*np.pi*f)])
          if kind == "movil" else np.asarray(a) + f*(np.asarray(b)-a))
    return np.rint(xy*scale).astype(int)


@dataclass
class CursorStep:
    smoothed: np.ndarray
    clipped: np.ndarray
    target_px: np.ndarray
    distance_px: float
    limit_px: float
    limited: bool


class Safety:
    def __init__(self, tau):
        self.smooth = Smoother(tau)
        self.reset()

    def reset(self):
        self.smooth.reset()
        self.valid_since = self.last_sent = None
        self.reopened_at = None
        self.soften_first_send = False
        self.trace = None

    def resume_after_reopen(self, now):
        self.reset()
        self.reopened_at = now

    def soften_after_face_loss(self):
        self.soften_first_send = True

    def step(self, predicted, current, now, enabled, scale):
        if predicted is None or not enabled or np.any((current < 0) | (current > scale)):
            self.reset()
            return None
        if self.valid_since is None:
            self.valid_since = now
        if self.reopened_at is None and now - self.valid_since < RECOVERY_S:
            return None
        previous = self.last_sent
        self.last_sent = now
        if previous is None and self.soften_first_send:
            self.reopened_at = now
        self.smooth.value, self.smooth.time = (
            (np.asarray(current,dtype=float)/scale, now-1/30) if previous is None else
            (self.smooth.value,self.smooth.time))
        smoothed = self.smooth.step(predicted,now)
        bounded = np.clip(smoothed,0.,1.)
        value = bounded*scale
        delta = value-current
        cap = min(MAX_STEP_PX, SPEED_PX_S*(now-previous if previous is not None else 1/30))
        if self.reopened_at is not None:
            ramp = (now - self.reopened_at) / .12
            if ramp < 1.:
                cap *= .25 + .75 * max(0., ramp)
        length = np.linalg.norm(delta)
        limited = length > cap
        self.trace = CursorStep(smoothed.copy(), smoothed != bounded, value, float(length), cap, limited)
        if limited:
            delta *= cap/length
        return np.rint(current+delta).astype(int)


class WindowsCursor:
    def __init__(self):
        self.api = ctypes.windll.user32
        self.api.FindWindowW.argtypes = (wintypes.LPCWSTR,wintypes.LPCWSTR)
        self.api.FindWindowW.restype = wintypes.HWND
        self.api.GetForegroundWindow.restype = wintypes.HWND
        self.window = self.api.FindWindowW(None,WINDOW)
        if not self.window:
            raise RuntimeError("Windows no reconoce la ventana de validación.")

    def down(self,key):
        return bool(self.api.GetAsyncKeyState(key) & 0x8000)

    def focused(self):
        return self.api.GetForegroundWindow() == self.window

    def get(self):
        point = wintypes.POINT()
        if not self.api.GetCursorPos(ctypes.byref(point)):
            raise RuntimeError("Windows no devolvió la posición del puntero.")
        return np.array([point.x,point.y])

    def move(self,point):
        if not self.api.SetCursorPos(int(point[0]),int(point[1])):
            raise RuntimeError("Windows rechazó el movimiento del puntero.")


class Experiment(Screen):
    def __init__(self,camera,width,height,tracker,ref,matrix,tau):
        super().__init__(camera,width,height)
        self.scale=np.array([width-1,height-1])
        self.tracker,self.ref,self.matrix=tracker,ref,matrix
        self.safety=Safety(tau)
        self.cursor=WindowsCursor()
        self.armed=self.last_f8=self.preview=False

    def keys(self):
        if self.cursor.down(0x1B):
            raise StopCapture
        down=self.cursor.down(0x77)
        if down and not self.last_f8:
            self.armed=not self.armed and self.cursor.focused()
            self.safety.reset()
        self.last_f8=down
        if not self.cursor.focused():
            self.armed=False

    def show(self,title,target,raw,corrected,cursor,row,rgb,lines,fps=None,ms=None,stable=None):
        self.buffer[:]=(23,26,30)
        for p,color in ((raw,BLUE),(corrected,GREEN)):
            if p is not None and np.all((p>=0)&(p<=1)):
                cv2.circle(self.buffer,tuple(np.rint(p*self.scale).astype(int)),self.radius+4,color,2,cv2.LINE_AA)
        cv2.circle(self.buffer,tuple(target),self.radius,YELLOW,-1,cv2.LINE_AA)
        cv2.drawMarker(self.buffer,tuple(cursor),(242,242,242),cv2.MARKER_CROSS,18,1,cv2.LINE_AA)
        width, height = self.width,self.height
        left=target[0]>=width/2
        x=round(width*(.04 if left else .52))
        text=[
            title,
            f"Actuador {'ARMADO' if self.armed else 'PAUSADO'} | F8 pausa/arma | ESC sale | V camara",
            f"Deteccion: {row['status']} | Afin disponible | azul original, verde corregido",
            f"FPS: {f'{fps:.1f}' if fps is not None else '--'} | "
            f"Inferencia: {f'{ms:.2f} ms' if ms is not None else '--'} | Estabilidad: {stable or '--'}",*lines]
        for i,s in enumerate(text):
            cv2.putText(self.buffer,s,(x,round(height*.16)+i*max(19,round(height*.027))),
                        cv2.FONT_HERSHEY_SIMPLEX,min(.74,width/2300),(225,233,242),1,cv2.LINE_AA)
        if self.preview and rgb is not None:
            w=round(width*.23)
            h=round(w*9/16)
            img=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),(w,h))
            axes=np.array([w/row["camera_width"],h/row["camera_height"]])
            origin=np.rint(np.array(self.ref["center_px"])*axes).astype(int)
            cv2.drawMarker(img,tuple(origin),(240,240,240),cv2.MARKER_CROSS,16,1)
            if row["valid_detection"]:
                p=np.rint(np.array([row["face_center_x_px"],row["face_center_y_px"]])*axes).astype(int)
                cv2.line(img,tuple(origin),tuple(p),YELLOW,2)
                cv2.circle(img,tuple(p),4,YELLOW,-1)
            y=round(height*.53)
            self.buffer[y:y+h,x:x+w]=img
        cv2.imshow(WINDOW,self.buffer)
        key=cv2.waitKey(1)&255
        if key==27 or cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:
            raise StopCapture
        if cv2.getWindowImageRect(WINDOW)!=(0,0,width,height):
            raise RuntimeError("La ventana dejó de coincidir con el monitor principal.")
        if key in (ord("v"),ord("V")):
            self.preview=not self.preview
        return key

    def sample(self):
        now,row,rgb,_=self.camera.read()
        start=perf_counter()
        raw=self.tracker.predict(row,self.ref) if row["valid_detection"] else None
        elapsed=(perf_counter()-start)*1000 if raw is not None else None
        return now,row,rgb,raw,affine(raw,self.matrix) if raw is not None else None,elapsed

    def wait_trial(self,title,scenario,result=False,note=""):
        target=target_at(scenario,scenario[1] if result else 0,self.scale)
        while True:
            self.keys()
            _,row,rgb,raw,corrected,ms=self.sample()
            message=("ENTER confirma | R repite" if result else
                     "Apunta al amarillo. F8 arma el cursor y ESPACIO inicia.")
            key=self.show(title,target,raw,corrected,self.cursor.get(),row,rgb,(message,note),ms=ms)
            if result and key in (10,13,ord("r"),ord("R")):
                return key in (10,13)
            if not result and key==32 and self.armed:
                return True

    def measure_trial(self,title,scenario,mode,rows):
        self.safety.reset()
        start=monotonic()
        last=None
        recent=[]
        target=target_at(scenario,0,self.scale)
        while True:
            now,row,rgb,raw,corrected,ms=self.sample()
            self.keys()
            if not self.armed:
                return False
            elapsed=now-start-WARM_S
            if elapsed>=scenario[1]:
                return True
            if elapsed<0:
                text="Preparacion: mantente en el punto inicial."
            elif scenario[0]=="recuperacion":
                text=("CUBRE brevemente la camara" if 2.2<=elapsed<4.2 else
                      "DESCUBRE la camara y apunta al amarillo" if elapsed>=4.2 else "Sostente en el objetivo")
            else:
                text="Sigue el objetivo amarillo con un movimiento comodo."
            before=self.cursor.get()
            selected=raw if mode=="RAW" else corrected
            sent=self.safety.step(selected,before,now,self.armed,self.scale)
            if sent is not None:
                self.keys()
                if not self.armed:
                    return False
                self.cursor.move(sent)
            actual=self.cursor.get()
            if selected is None:
                recent.clear()
            else:
                recent.append((now,selected))
                recent=[(t,p) for t,p in recent if now-t<=.5]
            stable=None
            if len(recent)>=8:
                span=np.ptp(np.percentile([p for _,p in recent],(10,90),axis=0),axis=0)*self.scale
                stable="estable" if max(span)<=25 else "en movimiento"
            if elapsed>=0:
                rows.append(dict(phase="measure",elapsed_s=elapsed,target_x=int(target[0]),target_y=int(target[1]),
                                 raw_x=raw[0] if raw is not None else None,raw_y=raw[1] if raw is not None else None,
                                 corrected_x=corrected[0] if corrected is not None else None,
                                 corrected_y=corrected[1] if corrected is not None else None,
                                 cursor_x=int(actual[0]),cursor_y=int(actual[1]),detection=raw is not None,
                                 status=row["status"],actuated=sent is not None,inference_ms=ms,
                                 outside=int(selected is not None and np.any((selected<0)|(selected>1))),
                                 cursor_step_px=float(np.linalg.norm(actual-before)),stable=stable))
            target=target_at(scenario,max(0,monotonic()-start-WARM_S),self.scale)
            self.show(title,target,raw,corrected,actual,row,rgb,(text,),
                      fps=1/(now-last) if last is not None and now>last else None,ms=ms,stable=stable)
            last=now


def run(experiment, folder):
    rows,attempts=[],[]
    try:
        for i,scenario in enumerate(SCENARIOS):
            modes=("RAW","CALIBRATED") if i%2==0 else ("CALIBRATED","RAW")
            for j,mode in enumerate(modes):
                title=f"{i*2+j+1}/18 | {mode} | {scenario[0].upper()}"
                while True:
                    experiment.armed=False
                    experiment.wait_trial(title,scenario)
                    sample=[]
                    outcome="cancelled"
                    try:
                        complete=experiment.measure_trial(title,scenario,mode,sample)
                        experiment.armed=False
                        stats=measure(sample,scenario[0]) if sample else None
                        usable = stats is not None and (scenario[0]!="recuperacion" or stats["recovery_observed"])
                        note = (f"Error medio: {stats['mean_px']:.1f} px" if usable else
                                "Falta perdida y recuperacion verificable, o el cursor no se movio.")
                        outcome=("confirmed" if experiment.wait_trial(title,scenario,result=True,note=note)
                                 else "repeated") if complete else "paused"
                        if outcome=="confirmed" and not usable:
                            outcome="incomplete"
                    finally:
                        experiment.armed=False
                        info=dict(attempt=len(attempts)+1,mode=mode,task=scenario[0],outcome=outcome)
                        attempts.append(info)
                        rows.extend(dict(r,**info) for r in sample)
                        report(folder,rows,attempts)
                    if outcome=="confirmed":
                        break
    except StopCapture:
        pass
    finally:
        results=report(folder,rows,attempts)
        print(f"{len(results)}/18 pruebas confirmadas | {folder}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",type=Path,default=PROFILE)
    parser.add_argument("--smoothing-ms",type=float,default=120.)
    args=parser.parse_args()
    if not np.isfinite(args.smoothing_ms) or args.smoothing_ms<0:
        parser.error("El suavizado debe ser finito y no negativo.")
    tracker=Tracker()
    width,height=primary_size()
    with ExitStack() as stack:
        camera=Camera9(stack)
        stack.callback(cv2.destroyAllWindows)
        matrix=profile_matrix(args.profile,tracker.identity,(width,height),camera.metadata)
        if matrix is None:
            raise RuntimeError("Se requiere un perfil aprobado para la comparación pareada.")
        experiment=Experiment(camera,width,height,tracker,None,matrix,args.smoothing_ms/1000)
        # La calibración neutral usa el visor base y nunca mueve el puntero.
        try:
            experiment.ref=calibrate(experiment)
        except StopCapture:
            return
        stamp=datetime.now(timezone.utc)
        folder=HERE/"cursor_validation"/f"{stamp:%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
        folder.mkdir(parents=True)
        atomic_json(folder/"metadata.json",dict(date=stamp.isoformat(),model=tracker.identity,
                    profile_sha256=sha256(args.profile.read_bytes()).hexdigest(),screen=[width,height],
                    camera=camera.metadata,neutral=experiment.ref,scenarios=SCENARIOS,
                    smoothing_ms=args.smoothing_ms,speed_px_s=SPEED_PX_S,max_step_px=MAX_STEP_PX,
                    recovery_s=RECOVERY_S,warm_s=WARM_S))
        print(folder,flush=True)
        run(experiment,folder)


if __name__=="__main__":
    try:
        main()
    except (ValueError,RuntimeError,OSError,cv2.error) as exc:
        raise SystemExit(str(exc)) from exc
