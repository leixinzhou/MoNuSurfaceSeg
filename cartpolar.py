import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy import interpolate

class CartPolar:
    """
    This class convert image and gt between Cartesian and Polar coordinate systems. 
        Args: 
            center: center coordinates (in Cart) of Polar, 
            final_radius: radius (physical) of Polar, 
            row_len: discretization number of Polar augular dim,
            col_len: discretization number of Polar radius dim, 
            sline_order: interpolation spline order (default 1)
            initial_radius: starting radius (default to be zero).
    """

    def __init__(self, center, final_radius, row_len, col_len, spline_order=1, initial_radius=0):

        self.center = center
        self.final_radius = final_radius
        self.row_len = row_len
        self.col_len = col_len
        self.spline_order = spline_order
        self.initial_radius = initial_radius

    def __polar2cart(self, r, theta):

        x = r * np.cos(theta) + self.center[0]
        y = r * np.sin(theta) + self.center[1]
        return x, y

    def __cart2polar(self, x, y):

        x_rela, y_rela = x - self.center[0], y - self.center[1]
        r = np.sqrt(x_rela**2 + y_rela**2)
        theta = np.zeros_like(x_rela)
        for i in range(len(theta)):
            if x_rela[i] == 0 and y_rela[i] > 0:
                theta[i] = 0.5*np.pi
            elif x_rela[i] == 0 and y_rela[i] < 0:
                theta[i] = 1.5*np.pi
            elif x_rela[i] > 0 and y_rela[i] >= 0:
                theta[i] = np.arctan(y_rela[i]/x_rela[i])
            elif x_rela[i] < 0:
                theta[i] = np.arctan(y_rela[i]/x_rela[i]) + np.pi
            else:
                theta[i] = np.arctan(y_rela[i]/x_rela[i]) + 2.*np.pi

        return r, theta

    def img2polar(self, img):

        theta, R = np.meshgrid(np.linspace(0, 2*np.pi, self.row_len),
                               np.linspace(self.initial_radius, self.final_radius, self.col_len))

        Xcart, Ycart = self.__polar2cart(R, theta)
        indices = np.reshape(Ycart, (-1, 1)), np.reshape(Xcart, (-1, 1))
        # print(indices[0].max(), indices[0].min(), indices[1].max(), indices[1].min())
        polar_img = []

        for i in range(3):
            polar_img.append(map_coordinates(img[i,], indices, order=self.spline_order, mode='reflect').reshape(
            (self.col_len, self.row_len)))

        return np.stack(polar_img)

    def gt2polar(self, gt):

        Rpol, Thetapol = self.__cart2polar(gt[:, 0], gt[:, 1])

        Rpol_map = Rpol*self.col_len/(self.final_radius-self.initial_radius)
        Thetapol_map = Thetapol*self.row_len/(2.*np.pi)

        Thetapol_dsz = np.arange(self.row_len)
        # replace linear interp with spline
        Rpol_dsz = np.interp(Thetapol_dsz, Thetapol_map,
                             Rpol_map, period=self.row_len)
        # tck = interpolate.splrep(Thetapol_map, Rpol_map)
        # Rpol_dsz = interpolate.splev(Thetapol_dsz,  tck, der=0)

        return Rpol_dsz

    def gt2cart(self, gt):

        R = gt*(self.final_radius-self.initial_radius)/self.col_len
        Theta = np.arange(self.row_len)*2.*np.pi / self.row_len

        return self.__polar2cart(R, Theta)
