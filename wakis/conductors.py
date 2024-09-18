import numpy as np
import scipy.optimize

class OutRect:
    def __init__(self, Lx, Ly, x_cent, y_cent):
        self.Lx = Lx
        self.Ly = Ly
        self.x_cent = x_cent
        self.y_cent = y_cent
        # self.theta = theta
        # self.R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        # self.mR = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])

    def out_conductor(self, x, y):
        # [xx, yy] = np.dot(self.mR, np.array([x, y]))
        return (-0.5 * self.Lx + self.x_cent < x < 0.5 * self.Lx + self.x_cent) and (
                -0.5 * self.Ly + self.y_cent < y < 0.5 * self.Ly + self.y_cent)

    def in_conductor(self, x, y):
        return not self.out_conductor(x, y)

    def intersec_x(self, x, y):
        # [xx, yy] = np.dot(self.mR, np.array([x, y]))
        inters_1 = -0.5 * self.Lx + self.x_cent
        inters_2 = inters_1 + self.Lx
        if abs(x - inters_1) < abs(x - inters_2):
            return inters_1
        else:
            return inters_2

        # [xxx, _] = np.dot(self.R, np.array([inters, yy]))
        # return xxx

    def intersec_y(self, x, y):
        # [xx, yy] = np.dot(self.mR, np.array([x, y]))
        inters_1 = -0.5 * self.Ly + self.y_cent
        inters_2 = inters_1 + self.Ly
        if abs(y - inters_1) < abs(y - inters_2):
            return inters_1
        else:
            return inters_2

        # [_, yyy] = np.dot(self.R, np.array([xx, inters]))
        # return yyy

class ImpFunc:
    def __init__(self, func):
        self.func = func

    def out_conductor(self, x, y):
        return self.func(x, y) < 0

    def in_conductor(self, x, y):
        return self.func(x, y) > 0

    def intersec_x(self, x, y):
        func_x = lambda t : self.func(t, y)

        return scipy.optimize.newton_krylov(func_x, x)

    def intersec_y(self, x, y):
        func_y = lambda t : self.func(x, t)

        return scipy.optimize.newton_krylov(func_y, y)
        
class Plane:
    def __init__(self, m_plane, q_plane, tol=0, sign=1):
        self.tol = tol  # 1e-16
        self.m_plane = m_plane
        self.q_plane = q_plane
        self.sign = sign

    def in_conductor(self, x, y):
        if self.sign == 1:
            return y - self.m_plane * x - self.q_plane >= self.tol
        elif self.sign == -1:
            return y - self.m_plane * x - self.q_plane <= self.tol
        else:
            print('sign must be + or - 1')

    def out_conductor(self, x, y):
        return not self.in_conductor(x, y)

    def intersec_x(self, x, y):
        return y / self.m_plane - self.q_plane / self.m_plane

    def intersec_y(self, x, y):
        return self.m_plane * x + self.q_plane


class InCircle:
    def __init__(self, radius, x_cent, y_cent):
        self.radius = radius
        self.x_cent = x_cent
        self.y_cent = y_cent

    def in_conductor(self, x, y):
        return np.square(x - self.x_cent) + np.square(y - self.y_cent) <= np.square(self.radius)

    def out_conductor(self, x, y):
        return not self.in_conductor(x, y)

    def intersec_x(self, x, y):
        if abs(y - self.y_cent) <= self.radius:
            inters_1 = np.sqrt(np.square(self.radius) - np.square(y - self.y_cent)) + self.y_cent
            inters_2 = -np.sqrt(np.square(self.radius) - np.square(y - self.y_cent)) + self.y_cent
            if abs(x - inters_1) < abs(x - inters_2):
                return inters_1
            else:
                return inters_2
        else:
            return np.inf

    def intersec_y(self, x, y):
        if abs(x - self.x_cent) <= self.radius:
            inters_1 = np.sqrt(np.square(self.radius) - np.square(x - self.x_cent)) + self.x_cent
            inters_2 = -np.sqrt(np.square(self.radius) - np.square(x - self.x_cent)) + self.x_cent
            if abs(y - inters_1) < abs(y - inters_2):
                return inters_1
            else:
                return inters_2
        else:
            return np.inf


class OutCircle:
    def __init__(self, radius, x_cent, y_cent):
        self.radius = radius
        self.x_cent = x_cent
        self.y_cent = y_cent

    def in_conductor(self, x, y):
        return np.square(x - self.x_cent) + np.square(y - self.y_cent) >= np.square(self.radius)

    def out_conductor(self, x, y):
        return not self.in_conductor(x, y)

    def intersec_x(self, x, y):
        # if abs(y - self.y_cent) > self.radius:
        inters_1 = np.sqrt(np.square(self.radius) - np.square(y - self.y_cent)) + self.x_cent
        inters_2 = -np.sqrt(np.square(self.radius) - np.square(y - self.y_cent)) + self.x_cent
        if abs(x - inters_1) < abs(x - inters_2):
            return inters_1
        else:
            return inters_2
        # else:
        #    return np.inf

    def intersec_y(self, x, y):
        # if abs(x - self.x_cent) > self.radius:
        inters_1 = np.sqrt(np.square(self.radius) - np.square(x - self.x_cent)) + self.y_cent
        inters_2 = -np.sqrt(np.square(self.radius) - np.square(x - self.x_cent)) + self.y_cent
        if abs(y - inters_1) < abs(y - inters_2):
            return inters_1
        else:
            return inters_2
        # else:
        #    return np.inf


class ConductorsAssembly:
    def __init__(self, conductors):
        self.conductors = conductors

    def in_conductor(self, x, y):
        for conductor in self.conductors:
            if conductor.in_conductor(x, y):
                return True

        return False

    def out_conductor(self, x, y):
        return not self.in_conductor(x, y)

    def intersec_x(self, x, y):
        list_inters = np.zeros_like(self.conductors)
        for ii, conductor in enumerate(self.conductors):
            list_inters[ii] = conductor.intersec_x(x, y)

        dist_inters = abs(list_inters - x)
        return list_inters[np.argmin(dist_inters)]

    def intersec_y(self, x, y):
        list_inters = np.zeros_like(self.conductors)
        for ii, conductor in enumerate(self.conductors):
            list_inters[ii] = conductor.intersec_y(x, y)

        dist_inters = abs(list_inters - y)
        return list_inters[np.argmin(dist_inters)]


class noConductor:
    def out_conductor(self, x, y):
        return True

    def in_conductor(self, x, y):
        return False

    def intersec_x(self, x, y):
        return 1000

    def intersec_y(self, x, y):
        return 1000

