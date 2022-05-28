import sys
import os

class Base:
    rootDir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    resourcesDir = os.path.join(rootDir, "resources")

