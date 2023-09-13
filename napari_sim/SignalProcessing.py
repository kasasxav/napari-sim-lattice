import numpy as np
import napari_sim.Utility36 as U


def shift_image(im, shifts):
    im_ = np.fft.fft2(im)
    if len(np.shape(shifts)) == 1:
        shifts = np.reshape(shifts, (1, *np.shape(shifts)))
    if len(np.shape(im_)) == 2:
        im_ = np.reshape(im_, (1, *np.shape(im_)))
    matrix = shift_matrix(shifts[:, 0], shifts[:, 1], im_.shape)
    return np.fft.ifft2(im_*matrix)


def shift_matrix(kx, ky, shape):
    x, y = mesh_grid(int(shape[1] / 2), int(shape[2] / 2))
    x = np.tile(x, (len(kx), 1, 1))
    y = np.tile(y, (len(ky), 1, 1))
    kx = np.reshape(kx, (len(kx), 1, 1))
    ky = np.reshape(ky, (len(ky), 1, 1))
    return np.exp(2j*np.pi*(kx*x/shape[1] + ky*y/shape[2]))


def mesh_grid(nx, ny):
    x = np.arange(2 * nx)
    y = np.arange(2 * ny)
    xv, yv = np.meshgrid(x, y, indexing='ij', sparse=True)
    i_xv = xv[1:nx + 1, 0]
    xv[nx:2 * nx, 0] = -np.flip(i_xv, 0)
    i_yv = yv[0, 1:ny + 1]
    yv[0, ny:2 * ny] = -np.flip(i_yv, 0)
    return xv, yv


def cross_correlation(f, g, nx, ny):
    result = []
    x = []
    y = []
    f = f - np.mean(f)
    g = g - np.mean(g)
    for i in nx:
        for j in ny:
            g_ = shift_image(g, [i, j])
            value = np.sum(np.multiply(g_, np.conj(f)))
            sigma_f = np.sqrt(np.sum(np.multiply(f, np.conj(f))))
            sigma_g = np.sqrt(np.sum(np.multiply(g_, np.conj(g_))))
            result = np.append(result, value/(sigma_f*sigma_g))
            x = np.append(x, i)
            y = np.append(y, j)
    return x, y, result


def normalize_sequence(f):
    f_ = np.empty(f.shape)
    for i in range(f.shape[0]):
        f_[i, :, :] = normalize_image(f[i, :, :])
    return f_


def normalize_image(im):
    im_ = np.divide((im - np.min(np.abs(im))), np.max(np.abs(im)) - np.min(np.abs(im)))
    return im_


def fft_sequence(f):
    f_ = np.empty(f.shape, dtype=complex)
    for i in range(f.shape[0]):
        f_[i, :, :] = np.fft.fft2(f[i, :, :])
    return f_


def ifft_sequence(f):
    f_ = np.empty(f.shape, dtype=complex)
    for i in range(f.shape[0]):
        f_[i, :, :] = np.fft.ifft2(f[i, :, :])
    return f_


def find_shifts(f, sign):
    f_ = np.zeros(f.shape)
    shifts = np.empty([f.shape[0], 2])
    f_[0, :, :] = f[0, :, :]
    shifts[0, :] = [0, 0]

    c = cross_correlation(f[0, :, :], f[1, :, :], [0], np.arange(2, 3*sign, 0.1*sign))
    shifts[1, 0] = c[0][c[2] == max(c[2])][0]
    shifts[1, 1] = c[1][c[2] == max(c[2])][0]

    c = cross_correlation(f[0, :, :], f[2, :, :], [0], np.arange(4, 6*sign, 0.1*sign))
    shifts[2, 0] = c[0][c[2] == max(c[2])][0]
    shifts[2, 1] = c[1][c[2] == max(c[2])][0]

    c = cross_correlation(f[0, :, :], f[3, :, :], np.arange(2, 3*sign, 0.1*sign), [0])
    shifts[3, 0] = c[0][c[2] == max(c[2])][0]
    shifts[3, 1] = c[1][c[2] == max(c[2])][0]

    c = cross_correlation(f[0, :, :], f[4, :, :], np.arange(4, 6*sign, 0.1*sign), [0])
    shifts[4, 0] = c[0][c[2] == max(c[2])][0]
    shifts[4, 1] = c[1][c[2] == max(c[2])][0]

    return shifts


def de_scan(f, shifts):
    return shift_image(f, np.negative(shifts))


def gauss2d(mat, sigma, center):
    size = mat.shape
    [x, y] = np.mgrid[0:size[0], 0:size[1]]
    f = gauss_c(x, y, sigma, center)
    return f


def gauss_c(x, y, sigma, center):
    xc = center[0]
    yc = center[0]
    exponent = np.divide(np.square(x-xc) + np.square(y-yc), 2*sigma)
    val = np.exp(-exponent)
    return val


def otf2d(wl, na, dx, nx):
    bpp = U.discArray((nx, nx), (na/wl)/(1/(nx*dx)))
    psf = np.square(np.real(np.fft.fft2(bpp)))
    psf = psf/psf.sum()
    return np.fft.fft2(psf)


def interp(self,arr):
    ''' interpolate by padding in frequency space '''
    nx,ny = arr.shape
    outarr = np.zeros((2*nx,2*ny), dtype=arr.dtype)
    arrf = np.fft.fft2(arr)
    arro = self.pad(arrf)
    outarr = np.fft.ifft2(arro)
    return outarr


def freq_pad(arr):
    arr = np.fft.fft2(arr)
    nz, nx, ny = arr.shape
    out = np.zeros((nz, 2*nx, 2*nx), arr.dtype).astype(np.complex64)
    nxh = np.int_(nx/2)

    out[:, :nxh, :nxh] = arr[:, :nxh, :nxh]
    out[:, :nxh, 3 * nxh:4 * nxh] = arr[:, :nxh, nxh:nx]
    out[:, 3 * nxh:4 * nxh, :nxh] = arr[:, nxh:nx, :nxh]
    out[:, 3 * nxh:4 * nxh, 3 * nxh:4 * nxh] = arr[:, nxh:nx, nxh:nx]
    return np.fft.ifft2(out)


def zerosuppression(sx, sy, nx, ny, sigma):
    ''' suppress zero frequency in SIM reconstruction '''
    x, y = mesh_grid(int(nx / 2), int(ny / 2))
    g = 1 - np.exp(-((x-sx)**2.+(y-sy)**2.)/(2.*sigma**2.))
    g[g < 0.5] = 0.0
    g[g >= 0.5] = 1.0
    return g


def pad(arr, px):
    nz, nx, ny = np.shape(arr)
    out = np.zeros((nz, int(nx+2*px), int(ny+2*px))).astype(np.complex64)
    out[:, px:px+nx, px:px+ny] = arr
    return out