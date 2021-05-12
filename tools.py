import vtk
import numpy as np


class tree:
    def __init__(self, root, left = None, right = None):
        self.root = root
        self.left = left
        self.right = right


class bone2VTK:
    def __init__(self, bone, resolution = 16):
        self._bone = bone
        self._resolution = resolution
        self._angles = np.arange(resolution) * np.pi * 2. / resolution

        if len(bone) < 2:
            raise ValueError("This is a point source only")
        
    def _genring(self, pID, normDir = np.array([0, 0, 1])):
        if pID == 0:
            axialDir = self._bone[1] - self._bone[0]
            cirumDir = np.cross(axialDir, normDir)
            x_ = np.cos(self._angles)

