"""Regression of the presentation layout and isolated settings, without camera or cursor."""
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import tracker_windows as desktop


def verify():
    app = QApplication([])
    with TemporaryDirectory() as folder:
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, folder)
        window = desktop.Window(consensus=True, presentation=True)
        window.timer.stop()
        home, settings, advanced = [window.sections.widget(i).widget() for i in range(3)]
        assert [window.sections.tabText(i) for i in range(3)] == ["Inicio", "Ajustes", "Avanzado"]
        assert home.isAncestorOf(window.start_camera) and home.isAncestorOf(window.arm)
        assert home.isAncestorOf(window.clicks) and advanced.isAncestorOf(window.winks)
        assert settings.isAncestorOf(window.fixation_method) and advanced.isAncestorOf(window.diagnostic)
        assert window.fixation_method.currentIndex() == 1 and window.home_state.text() == "Inactivo"
        with patch.object(window.vision, "start") as start, patch.object(window.access_bar, "show"), patch.object(window.access_bar, "place"):
            window.start_camera.click()
            start.assert_called_once()
            assert not window.start_camera.isEnabled()
        window.start_camera.setEnabled(True)
        window.resize(398, 390)
        window.show()
        window.sections.setCurrentIndex(2)
        app.processEvents()
        assert window.sections.currentWidget().verticalScrollBar().maximum() > 0
        with patch.object(window.vision, "isRunning", return_value=True):
            window.refresh_demo_state()
            assert window.home_state.text() == "Calibrando"
            window.vision.neutral_prediction = [.5, .5]
            window.refresh_demo_state()
            assert window.home_state.text() == "Listo"
            window.armed = True
            window.refresh_demo_state()
            assert window.home_state.text() == "Activo"
            window.armed = False
            window.refresh_demo_state()
            assert window.home_state.text() == "Pausado"
        window.fixation_method.setCurrentIndex(0)
        window.fixation_confirm.setValue(90)
        window.close()
        resumed = desktop.Window(consensus=True, presentation=True)
        resumed.timer.stop()
        assert resumed.fixation_method.currentIndex() == 0
        assert resumed.fixation_confirm.value() == 90 and not resumed.armed and not resumed.clicks.isChecked()
        resumed.close()
    print("Presentation: sections, scrolling, live-state mapping and settings after restart verified.")


if __name__ == "__main__":
    verify()
