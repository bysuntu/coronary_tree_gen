import numpy as np
import scipy.stats as stats
import vtk
from matplotlib import pyplot as plt

# According the arterial tree model
# Tissue metabolism driven arterial tree generation

class bifucation:
    def __init__(self, radius, direction = None):
        self.radius = radius
        if direction is not None:
            self.direction = direction
        else:
            self.direction = np.array([0, 0, 1.])

        self.bones = np.zeros((6, 3))

        # Left Node
        _lr = [60 / 180 * np.pi, 180 / 180 * np.pi]
        self._left_angle = np.clip(np.random.normal(loc = 120 / 180 * np.pi, scale = 30 / 180 * np.pi), _lr[0], _lr[1])
        self._left_length = np.clip(np.random.normal(loc = self.radius * 5, scale = self.radius), self.radius, self.radius * 10)
        self._left_radius = np.clip(np.random.normal(loc = 0.5 * self.radius, scale = self.radius), 0.1 * self.radius, self.radius * 0.9)
        self.bones[4][0] = np.cos(self._left_angle) * self._left_length
        self.bones[4][1] = np.sin(self._left_angle) * self._left_length

        # Right Node
        _rr = [0, self._left_angle]
        self._right_angle = np.clip(np.random.normal(loc = 0.5 * (_rr[0] + _rr[1]), scale = 30 / 180 * np.pi), _rr[0], _rr[1])
        self._right_length = np.clip(np.random.normal(loc = self.radius * 5, scale = self.radius), self.radius, self.radius * 10)
        # self._right_radius = np.clip(np.random.normal(loc = 0.5 * self.radius, scale = self.radius), 0.1 * self.radius, self.radius)
        self._right_radius = (self.radius ** 2.5 - self._left_radius ** 2.5)**0.4
        self.bones[5][0] = np.cos(self._right_angle) * self._right_length
        self.bones[5][1] = np.sin(self._right_angle) * self._right_length

        # Bifurcation Node Eq 3
        initial_guess = (self.bones[5] + self.bones[4]) * 0.5

        track = []
        self._calculate_bifucation(initial_guess, 0.1, 0.01, track)

        print(len(track), self._left_radius, self._right_radius)
        plt.scatter(self.bones[0][0], self.bones[0][1], marker= 'o')
        plt.scatter(self.bones[-1][0], self.bones[-1][1], marker = 's')
        plt.scatter(self.bones[4][0], self.bones[4][1], marker = '+')
        plt.plot([row[0] for row in track], [row[1] for row in track], '-', lw = 0.2)
        plt.scatter(track[-1][0], track[-1][1], marker = '+')
        plt.show()
        

    def _calculate_bifucation(self, guess, step, criterion, track):

        l2b = np.linalg.norm(self.bones[4] - guess)
        r2b = np.linalg.norm(self.bones[5] - guess)
        p2b = np.linalg.norm(self.bones[0] - guess)

        gradient = np.zeros(3)
        for k in range(3):
            gradient[k] = self._left_radius**2/l2b * (guess[k] - self.bones[4][k])
            gradient[k] += self._right_radius**2/r2b * (guess[k] - self.bones[5][k])
            gradient[k] += self.radius**2/p2b * (guess[k] - self.bones[0][k])

        if (np.linalg.norm(gradient * step)) < criterion or len(track) > 500:
            return 
        else:
            if np.linalg.norm(gradient * step) > 0.1 * self.radius:
                step = step * 0.5
            guess = guess - gradient * step
            track.append(guess)
            self._calculate_bifucation(guess, step, criterion, track)


each = bifucation(5)

