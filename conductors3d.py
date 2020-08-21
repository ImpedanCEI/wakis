import numpy as np


class ConductorsAssembly:
    def __init__(self, conductors):
        self.conductors = conductors

    def in_conductor(self, x, y, z):
        for conductor in self.conductors:
            if conductor.in_conductor(x, y, z):
                return True

        return False

    def out_conductor(self, x, y, z):
        return not self.in_conductor(x, y, z)

    def intersec_x(self, x, y, z):
        list_inters = np.zeros_like(self.conductors)
        for ii, conductor in enumerate(self.conductors):
            list_inters[ii] = conductor.intersec_x(x, y, z)

        dist_inters = abs(list_inters - x)
        return list_inters[np.argmin(dist_inters)]

    def intersec_y(self, x, y, z):
        list_inters = np.zeros_like(self.conductors)
        for ii, conductor in enumerate(self.conductors):
            list_inters[ii] = conductor.intersec_y(x, y, z)

        dist_inters = abs(list_inters - y)
        return list_inters[np.argmin(dist_inters)]

    def intersec_z(self, x, y, z):
        list_inters = np.zeros_like(self.conductors)
        for ii, conductor in enumerate(self.conductors):
            list_inters[ii] = conductor.intersec_z(x, y, z)

        dist_inters = abs(list_inters - z)
        return list_inters[np.argmin(dist_inters)]


class InCube:

    def __init__(self, lx, ly, lz, x_cent, y_cent, z_cent):
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.x_cent = x_cent
        self.y_cent = y_cent
        self.z_cent = z_cent

    def out_conductor(self, x, y, z):
        # [xx, yy] = np.dot(self.mR, np.array([x, y]))
        return (-0.5 * self.lx + self.x_cent < x < 0.5 * self.lx + self.x_cent and
                -0.5 * self.ly + self.y_cent < y < 0.5 * self.ly + self.y_cent and
                -0.5 * self.lz + self.z_cent < z < 0.5 * self.lz + self.z_cent)

    def in_conductor(self, x, y, z):
        return not self.out_conductor(x, y, z)

    def intersec_x(self, x, y, z):
        # [xx, yy] = np.dot(self.mR, np.array([x, y]))
        inters_1 = -0.5 * self.lx + self.x_cent
        inters_2 = inters_1 + self.lx
        if abs(x - inters_1) < abs(x - inters_2):
            return inters_1
        else:
            return inters_2

        # [xxx, _] = np.dot(self.R, np.array([inters, yy]))
        # return xxx

    def intersec_y(self, x, y, z):
        # [xx, yy] = np.dot(self.mR, np.array([x, y]))
        inters_1 = -0.5 * self.ly + self.y_cent
        inters_2 = inters_1 + self.ly
        if abs(y - inters_1) < abs(y - inters_2):
            return inters_1
        else:
            return inters_2

    def intersec_z(self, x, y, z):
        # [xx, yy] = np.dot(self.mR, np.array([x, y]))
        inters_1 = -0.5 * self.lz + self.z_cent
        inters_2 = inters_1 + self.lz
        if abs(z - inters_1) < abs(z - inters_2):
            return inters_1
        else:
            return inters_2

class InSphere:

    def __init__(self, radius, x_cent, y_cent, z_cent):
        self.radius = radius
        self.x_cent = x_cent
        self.y_cent = y_cent
        self.z_cent = z_cent

    def out_conductor(self, x, y, z):
        # [xx, yy] = np.dot(self.mR, np.array([x, y]))
        return (np.square(x - self.x_cent) + np.square(y - self.y_cent) + np.square(z - self.z_cent)
                <= np.square(self.radius))

    def in_conductor(self, x, y, z):
        return not self.out_conductor(x, y, z)

    def intersec_x(self, x, y, z):
        # [xx, yy] = np.dot(self.mR, np.array([x, y]))
        inters_1 = (np.sqrt(np.square(self.radius) - np.square(y - self.y_cent)
                            - np.square(z - self.z_cent)) + self.x_cent)
        inters_2 = -(np.sqrt(np.square(self.radius) - np.square(y - self.y_cent)
                             - np.square(z - self.z_cent)) + self.x_cent)

        if abs(x - inters_1) < abs(x - inters_2):
            return inters_1
        else:
            return inters_2

        # [xxx, _] = np.dot(self.R, np.array([inters, yy]))
        # return xxx

    def intersec_y(self, x, y, z):
        # [xx, yy] = np.dot(self.mR, np.array([x, y]))
        inters_1 = (np.sqrt(np.square(self.radius) - np.square(x - self.x_cent)
                            - np.square(z - self.z_cent)) + self.y_cent)
        inters_2 = -(np.sqrt(np.square(self.radius) - np.square(x - self.x_cent)
                             - np.square(z - self.z_cent)) + self.y_cent)
        if abs(y - inters_1) < abs(y - inters_2):
            return inters_1
        else:
            return inters_2

    def intersec_z(self, x, y, z):
        # [xx, yy] = np.dot(self.mR, np.array([x, y]))
        inters_1 = (np.sqrt(np.square(self.radius) - np.square(x - self.x_cent)
                            - np.square(y - self.y_cent)) + self.z_cent)
        inters_2 = -(np.sqrt(np.square(self.radius) - np.square(x - self.x_cent)
                             - np.square(y - self.y_cent)) + self.z_cent)
        if abs(z - inters_1) < abs(z - inters_2):
            return inters_1
        else:
            return inters_2


class noConductor:
    #def __init__(self):

    def out_conductor(self, x, y, z):
        return True

    def in_conductor(self, x, y, z):
        return False

    def intersec_x(self, x, y, z):
        return 1000

    def intersec_y(self, x, y, z):
        return 1000

    def intersec_z(self, x, y, z):
        return 1000
