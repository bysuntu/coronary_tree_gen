import numpy as np
import scipy.stats as stats
import vtk
from matplotlib import pyplot as plt

class tNode:
    def __init__(self, position = np.zeros(3), radius = 0, angle = 0, length = 0):
        self.position = position
        self.radius = radius
        self.angle = angle
        self.length = length

class bifurcation:
    def __init__(self, radius, direction = np.array([0, 1, 0])):
        self.parent = tNode(np.zeros(3), radius)
        self.radius = radius
        self._genLeft()
        self._genRight()
        self._genBifur()

        print(self.parent.position, self.left.position, self.right.position, self.bifur.position)

    def _genLeft(self):
        _lr = [60 / 180 * np.pi, 180 / 180 * np.pi]
        _left_angle = np.clip(np.random.normal(loc = 120 / 180 * np.pi, scale = 30 / 180 * np.pi), _lr[0], _lr[1])
        _left_length = np.clip(np.random.normal(loc = self.radius * 5, scale = self.radius), self.radius, self.radius * 10)
        
        radius = np.clip(np.random.normal(loc = 0.5 * self.radius, scale = self.radius), 0.1 * self.radius, self.radius * 0.9)
        position = np.zeros(3)
        position[0] = np.cos(_left_angle) * _left_length
        position[1] = np.sin(_left_angle) * _left_length
        self.left = tNode(position, radius, _left_angle, _left_length)

    def _genRight(self):
        _rr = [0, self.left.angle]
        _right_angle = np.clip(np.random.normal(loc = 0.5 * (_rr[0] + _rr[1]), scale = 30 / 180 * np.pi), _rr[0], _rr[1])
        _right_length = np.clip(np.random.normal(loc = self.radius * 5, scale = self.radius), self.radius, self.radius * 10)
        # self._right_radius = np.clip(np.random.normal(loc = 0.5 * self.radius, scale = self.radius), 0.1 * self.radius, self.radius)
        radius = (self.radius ** 2.5 - self.left.radius ** 2.5)**0.4
        position = np.zeros(3)
        position[0] = np.cos(_right_angle) * _right_length
        position[1] = np.sin(_right_angle) * _right_length
        self.right = tNode(position, radius, _right_angle, _right_length)

    def _genBifur(self):
        initial_guess = (self.left.position + self.right.position) * 0.5
        track = []
        self._calculate_bifurcation(initial_guess, 0.1, 0.01, track)
        volume = [row[1] for row in track]
        track = [row[0] for row in track]

        print(len(track), self.left.radius, self.right.radius)
        plt.subplot(121)
        plt.scatter(self.parent.position[0], self.parent.position[1], marker= 'o')
        plt.scatter(self.left.position[0], self.left.position[1], marker = '^')
        plt.scatter(self.right.position[0], self.right.position[1], marker = 's')
        plt.plot([row[0] for row in track], [row[1] for row in track], '-', lw = 0.2)
        plt.scatter(track[-1][0], track[-1][1], marker = 'x')
        plt.subplot(122)
        plt.plot(volume)
        plt.show()
        self.bifur = tNode(track[-1], self.radius)


    def _calculate_bifurcation(self, guess, step, criterion, track):
        l2b = np.linalg.norm(self.left.position - guess)
        r2b = np.linalg.norm(self.right.position - guess)
        p2b = np.linalg.norm(self.parent.position - guess)

        gradient = np.zeros(3)
        for k in range(3):
            gradient[k] = self.left.radius**2/l2b * (guess[k] - self.left.position[k])
            gradient[k] += self.right.radius**2/r2b * (guess[k] - self.right.position[k])
            gradient[k] += self.parent.radius**2/p2b * (guess[k] - self.parent.position[k])

        volume = l2b * self.left.radius**2
        volume += r2b * self.right.radius**2
        volume += p2b * self.parent.radius**2

        if (np.linalg.norm(gradient * step)) < criterion or len(track) > 500:
            return 
        else:
            if np.linalg.norm(gradient * step) > 0.1 * self.radius:
                step = step * 0.5
            guess = guess - gradient * step
            track.append([guess, volume])
            self._calculate_bifurcation(guess, step, criterion, track)

each = bifurcation(5)