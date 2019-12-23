# ----------------------------------------------------- #
# Project   :   Poisson Image Editing                   #
# Title     :   poisson.py                              #
# Detail    :   editing function                        #
#                                                       #
# Author    :   Xu Ruihan                               #
# ID        :   1700017793                              #
# ----------------------------------------------------- #

import numpy as np
import cv2
from scipy.sparse import linalg as linalg
from scipy.sparse import lil_matrix as lil_matrix


class Poisson:
    # enum
    INSIDE = 0
    BOUND = 1
    OUTSIDE = 2

    # init
    def __init__(self, source, target, mask):
        self.__source = source
        self.__target = target
        # Normalize __mask
        mask = np.atleast_3d(mask).astype(np.float) / 255
        # Make __mask binary
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        # Trim to one channel
        mask = mask[:, :, 0].astype(int)
        self.__mask = mask

        # Set Height and Width of image
        self.__Height = len(mask[:, 0])
        self.__Width = len(mask[0, :])

        # Number of channels
        self.__channels = source.shape[-1]

    # get neighbour points
    def __getNeighbour(self, point):
        x, y = point
        neighbour = []
        if x > 0:
            neighbour.append((x - 1, y))
        if x < self.__Height - 1:
            neighbour.append((x + 1, y))
        if y > 0:
            neighbour.append((x, y - 1))
        if y < self.__Width - 1:
            neighbour.append((x, y + 1))
        return neighbour

    # get location of a point(inside, bound, outside)
    def __getLocation(self, point):
        if self.__mask[point] == 1:
            return self.INSIDE
        else:
            for pt in self.__getNeighbour(point):
                if self.__mask[pt] == 1:
                    return self.BOUND
        return self.OUTSIDE

    # Find where the __mask is 1 to build fomular
    def __getMask(self):
        nonzero = np.nonzero(self.__mask)
        return zip(nonzero[0], nonzero[1])

    # Create the coeffient matrix
    def __getPoissonMatrix(self, points):
        N = len(points)
        A = lil_matrix((N, N))
        # Set up row for each point in __mask
        for i, coordinate in enumerate(points):
            neighbour = self.__getNeighbour(coordinate)
            A[i, i] = len(neighbour)
            for x in neighbour:
                if x not in points:
                    continue
                j = points.index(x)
                A[i, j] = -1
        return A

    # Get different gradients for different settings
    def __getGradients(self, source, target, p, setting):
        neighbour = self.__getNeighbour(p)
        b = 0
        if setting == 'ImportingGradients':
            for q in neighbour:
                b += int(source[p[0], p[1]]) - int(source[q[0], q[1]])  # v_pq
                if self.__getLocation(q) == self.BOUND:
                    b += target[q[0], q[1]]
        elif setting == 'MixingGradients':
            for q in neighbour:
                g = int(source[p[0], p[1]]) - int(source[q[0], q[1]])
                f_star = int(target[p[0], p[1]]) - int(target[q[0], q[1]])
                if abs(f_star) > abs(g):
                    b += f_star
                else:
                    b += g
                if self.__getLocation(q) == self.BOUND:
                    b += target[q[0], q[1]]

        return b

    # Process Poisson on one channel
    def __process(self, source, target, setting):
        MaskPositions = list(self.__getMask())
        N = len(MaskPositions)
        A = self.__getPoissonMatrix(MaskPositions)
        b = np.zeros(N)
        for i, coordinate in enumerate(MaskPositions):
            b[i] = self.__getGradients(source, target, coordinate, setting)
        print(f' * Equations built. {N} equations to be solved.')

        x = linalg.cg(A, b)
        composite = np.copy(target).astype(int)
        for i, coordinate in enumerate(MaskPositions):
            composite[coordinate] = x[0][i]
        print(' * Channel solved. Waiting for the next or merging channels')
        return composite

    # Editing through different channels
    def edit(self, setting='MixingGradients'):
        # Call the poisson method on each individual channel
        result_stack = [
            self.__process(self.__source[:, :, i],
                           self.__target[:, :, i],
                           setting=setting) for i in range(self.__channels)
        ]
        # Merge the channels
        result = cv2.merge(result_stack)
        return result
