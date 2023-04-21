import os
import site
import pathlib
# TODO remove this hack if not needed
APP_DIR = os.path.dirname(__file__)
try:
    __import__(os.path.basename(APP_DIR))
except ModuleNotFoundError:
    # some modules want to import lmuvegetationapp
    site.addsitedir(pathlib.Path(__file__).parents[1])
