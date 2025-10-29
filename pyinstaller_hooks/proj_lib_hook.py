import os
import sys

if hasattr(sys, '_MEIPASS'):
    # Running as executable - use temp extraction folder
    os.environ['PROJ_LIB'] = os.path.join(sys._MEIPASS, "pyproj", "proj_dir", "share", "proj")

