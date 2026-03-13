# from mesas.sas._sas import solve
# from mesas.sas._sas import make_lookup
# from mesas.sas._sas_functions import create_function
# from mesas.sas._util import transport
import ctypes
import platform
from pathlib import Path

if platform.system() == "Windows":
    libs_dir = Path(__file__).parent.parent / ".libs"
    for lib in libs_dir.glob("*.dll"):
        ctypes.CDLL(str(lib))
