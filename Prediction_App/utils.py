# utils.py
import sys
from pathlib import Path

def get_base_path():
    if getattr(sys, 'frozen', False):
        # PyInstaller ile paketlenmiş uygulamalarda, _MEIPASS geçici çıkarma dizinidir.
        return Path(sys._MEIPASS)
    else:
        # Normal Python çalıştırırken
        return Path(__file__).parent

def get_absolute_path(relative_path):
    base_path = get_base_path()
    # base_path zaten _MEIPASS'i veya betik dizinini işaret eder.
    # relative_path'iniz 'Urunum_bekledigim_kalitede_degil/...' gibi olduğu için
    # 'Models' klasörünü birleştirirken bir kez ekliyoruz.
    return str(base_path / "Models" / relative_path)