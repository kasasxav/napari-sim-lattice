from napari_sim.SignalProcessing import *


class Reconstruction:

    def __init__(self, pixel_size=65, periodicity=312, na=1.4, wvl=0.51, pad_px=0, offset=100, w=0.01,
                 rad=0.7, cd=1, t=1):
        self._pixel_size = pixel_size / 2
        self._periodicity = periodicity
        self._px = self._periodicity / self._pixel_size
        self._na = na
        self._wvl = wvl
        self._pad_px = pad_px
        self._offset = offset
        self._w = w
        self._rad = rad
        self._cd = cd
        self._t = t

    def run(self, image):
        _data = freq_pad(pad(np.double(image), self._pad_px))
        _data_desc = self.align_data(_data)
        c = self.calculate_bands(_data_desc)
        self.calculate_otf()
        self.finetune_periodicity(c)
        c_ = self.calculate_corrected_bands(c)
        a_ = self.estimate_parameters(c_)
        sr_ = self.combine_bands(c_, a_)
        return np.real(self.apodize(sr_))

    def get_wf(self, image):
        image_ = freq_pad(pad(np.double(image), self._pad_px))
        return np.real(np.sum(self.align_data(image_), 0))

    def align_data(self, _data):
        self._frames = np.shape(_data)[0]
        pos = np.sqrt(self._frames)
        ps = range(self._frames)
        x = np.divide(np.floor(np.divide(ps, pos)), pos-1)*self._px
        y = np.divide(np.mod(ps, pos), pos-1)*self._px
        shifts = np.transpose(np.array((x, y)))
        _data_desc = de_scan(_data, shifts)
        return _data_desc[:, 20:-20, 20:-20]

    def calculate_bands(self, _data):
        _data = np.fft.fft2(_data)
        self._nx = np.shape(_data)[1]
        self._ny = np.shape(_data)[2]
        pos = np.sqrt(self._frames)
        ps = range(self._frames)
        denom = np.sqrt(pos) * self._cd * self._t
        self._harm = (np.sqrt(self._frames) - 1) / 2
        x = np.reshape(np.floor(np.divide(ps, pos)), (1, self._frames))
        y = np.reshape(np.mod(ps, pos), (1, self._frames))
        k = np.tile(np.divide(x, pos - 1), (self._frames, 1)) * self._px
        l = np.tile(np.divide(y, pos - 1), (self._frames, 1)) * self._px
        self._x = x - self._harm
        self._y = y - self._harm
        m = np.tile(np.transpose(self._x), (1, self._frames))
        n = np.tile(np.transpose(self._y), (1, self._frames))
        d = np.reshape(_data, [self._frames, np.prod([np.shape(_data)[1], np.shape(_data)[2]])])
        matrix = np.exp(1j*(m*k + n*l))
        num = np.matmul(matrix, d)
        r = np.divide(num, denom)
        return np.reshape(r, np.shape(_data))

    def calculate_otf(self):
        self._otf = otf2d(self._wvl, self._na, self._pixel_size / 1000, int(self._nx))
        sx, sy = np.unravel_index(self._otf.argmax(), self._otf.shape)
        g = zerosuppression(sx, sy, int(self._nx), int(self._nx), 4)
        self._otf_z = g * self._otf
        self._p = self._nx / self._px

    def finetune_periodicity(self, c):
        c_0 = np.multiply(c[0, :, :], np.conj(self._otf))
        c_1 = np.multiply(shift_image(c[1, :, :], [-self._p, 0]), shift_image(np.conj(self._otf), [-self._p, 0]))
        (x, y, result) = cross_correlation(c_0, c_1, np.arange(-10, 0), [0])
        pk = np.argmax(np.abs(result))
        delta_1 = x[pk]

        c_2 = np.multiply(shift_image(c[2, :, :], [self._p, 0]), shift_image(np.conj(self._otf), [self._p, 0]))
        (x, y, result) = cross_correlation(c_0, c_2, np.arange(0, 10), [0])
        pk = np.argmax(np.abs(result))
        delta_2 = x[pk]

        c_3 = np.multiply(shift_image(c[3, :, :], [0, -self._p]), shift_image(np.conj(self._otf), [0, -self._p]))
        (x, y, result) = cross_correlation(c_0, c_3, [0], np.arange(-10, 0))
        pk = np.argmax(np.abs(result))
        delta_3 = y[pk]

        c_4 = np.multiply(shift_image(c[4, :, :], [0, self._p]), shift_image(np.conj(self._otf), [0, self._p]))
        (x, y, result) = cross_correlation(c_0, c_4, [0], np.arange(0, 10))
        pk = np.argmax(np.abs(result))
        delta_4 = y[pk]

        self._p = self._p + (delta_1 + delta_2 + delta_3 + delta_4)/4

    def calculate_corrected_bands(self, c):
        self._shifts = np.transpose(np.array((np.squeeze(self._x), np.squeeze(self._y))) * self._p)
        c_ = shift_image(c, self._shifts)
        otf_ = np.tile(np.reshape(np.conj(self._otf), (1, *np.shape(self._otf))), (np.shape(c_)[0], 1, 1))
        otf_ = shift_image(otf_, self._shifts)
        return np.multiply(c_, otf_)

    def estimate_parameters(self, c_):
        s0 = np.fft.fftshift(np.abs(U.discArray((self._nx, self._nx),
                                                (self._na / self._wvl) / (1 / (self._nx * (self._pixel_size / 1000))))))
        i_c = int((self._frames - 1) / 2)
        am = np.zeros((np.shape(c_)[0])).astype(complex)
        cp0_ = c_[i_c, :, :] * np.conj(c_[i_c, :, :])
        for i in range(np.shape(c_)[0]):
            s_i_1 = shift_image(s0, self._shifts[i, :])
            k = np.abs(s0 * s_i_1)
            cp1 = np.sum(c_[i_c, :, :] * np.conj(c_[i, :, :]) * k)
            cp0 = np.sum(cp0_ * k)
            am[i] = cp1 / cp0
        return am

    def combine_bands(self, c_, am):
        otf_0 = np.multiply(self._otf, np.conj(self._otf))
        otf_sum = 0
        num = 0
        for i in range(np.shape(am)[0]):
            otf_sum = otf_sum + shift_image(am[i]*np.conj(am[i])*otf_0, self._shifts[i, :])
            num = num + c_[i, :, :]*np.conj(am[i])
        return np.divide(num, self._w + otf_sum)

    def apodize(self, sr_):
        xx, yy = mesh_grid(int(self._nx / 2), int(self._nx / 2))

        r = np.hypot(xx * 2 / self._nx, yy * 2 / self._ny)
        window = np.multiply(np.exp(-np.square(r / self._rad * np.sqrt(2 * np.log(2)))), (r <= self._rad).astype(int))

        sr_w = sr_ * window
        return np.real(np.fft.ifft2(sr_w))
