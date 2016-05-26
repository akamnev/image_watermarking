"""
Реализованы алгоритмы добавления водяных знаков на изображение
"""
from skimage.util import view_as_blocks
import numpy as np
from skimage.morphology import remove_small_holes
from scipy.fftpack import dct, idct


def _ycc(rgb):
    """
    Convert color space from RGB to YCbCr.
    :return:
    """
    # check range
    min_color, max_color = np.min(rgb.ravel()), np.max(rgb.ravel())
    rgb = (rgb - min_color) / (max_color - min_color) * 255
    # split by colors
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    y = .299 * r + .587 * g + .114 * b
    cb = 128 - .168736 * r - .331364 * g + .5 * b
    cr = 128 + .5 * r - .418688 * g - .081312 * b
    return y, cb, cr


def _rgb(y, cb, cr):
    """
    Convert color space from YCbCr to RGB
    :param y:
    :param cb:
    :param cr:
    :return:
    """
    # B: (y, cb, cr) -> (r, g, b)
    b = np.linalg.inv(np.array([[0.299, 0.587, 0.114],
                                [-0.168736, -0.331364, 0.5],
                                [0.5, -0.418688, -0.081312]]))
    r = b[0, 0] * y + b[0, 1] * (cb - 128) + b[0, 2] * (cr - 128)
    g = b[1, 0] * y + b[1, 1] * (cb - 128) + b[1, 2] * (cr - 128)
    b = b[2, 0] * y + b[2, 1] * (cb - 128) + b[2, 2] * (cr - 128)
    image = np.zeros((r.shape[0], r.shape[1], 3))
    image[:, :, 0] = r
    image[:, :, 1] = g
    image[:, :, 2] = b
    image[image < 0] = 0
    image[image > 255] = 255
    return np.round(image).astype(np.uint8)


class BadWatermark(Exception):
    """Класс ошибки некорректного водяного знака."""
    pass


class UnknownAlgorithm(Exception):
    pass


class CoverGrid:
    """
    Класс разбивает изображение на блоки заданных размеров.
    """
    def __init__(self, image):
        self.image = image
        self.image_clone = None
        self.block_width = None
        self.block_height = None
        self.nrows = None
        self.ncols = None
        self.image_blocks = None
        self.image_blocks_magnitude = None
        self.image_blocks_phase = None
        self.dct_params = {'type': 2, 'norm': 'ortho'}  # параметры дискретного преобразования косинусов

    def create_grid(self, height, width):
        """
        Выполняет разбиение изображения на блоки размера (height, width)
        """
        self.block_height = height
        self.block_width = width
        # копируем область изображения, которая покрывается заданными блоками
        h, w = self.image.shape
        self.nrows = h // self.block_height
        self.ncols = w // self.block_width
        self.image_clone = self.image[:self.block_height * self.nrows, :self.block_width * self.ncols].copy()
        # разбиваем скопированную часть изображения на блоки
        self.image_blocks = view_as_blocks(self.image_clone, (self.block_height, self.block_width))

    def foreach_dft(self):
        """
        Выполняет дискретное преобразование Фурье для каждого блока.
        """
        self.image_blocks_magnitude = np.zeros(self.image_blocks.shape)
        self.image_blocks_phase = np.zeros(self.image_blocks.shape)
        # дискретное преобразование Фурье
        for i in range(self.nrows):
            for j in range(self.ncols):
                block_dft = np.fft.fft2(self.image_blocks[i, j, ...])
                magnitude = np.abs(block_dft)  # модуль
                self.image_blocks_phase[i, j, ...] = np.angle(block_dft)  # фаза
                # сдвигаем нулевую частоту в цетр матрицы Фурье-коэффициентов
                hh = magnitude.shape[0] // 2 + 1 if magnitude.shape[0] % 2 else magnitude.shape[0] // 2
                hw = magnitude.shape[1] // 2 + 1 if magnitude.shape[1] % 2 else magnitude.shape[1] // 2
                # axis = 0
                spectrum_positive, spectrum_negative = magnitude[:hh, :], magnitude[hh:, :]
                magnitude = np.concatenate((spectrum_negative, spectrum_positive), axis=0)
                # горизонталь
                spectrum_positive, spectrum_negative = magnitude[:, :hw], magnitude[:, hw:]
                magnitude = np.concatenate((spectrum_negative, spectrum_positive), axis=1)
                # magnitude - нулевая частота в центре матрицы
                self.image_blocks_magnitude[i, j, ...] = magnitude

    def foreach_idft(self):
        """
        Выполняет обратное дискретное преобразование Фурье для каждого блока
        :return:
        """
        # Здесь можно ускорить процесс учитывае только те области, которые были изменены, но тогда либо нарушается
        # принцип наследования, либо надо вводить флаг изменения
        for i in range(self.nrows):
            for j in range(self.ncols):
                # возврящаем представление матрцицы модулей к виду, где высокие частоты в центре матрицы
                magnitude = self.image_blocks_magnitude[i, j, ...]
                hh, hw = magnitude.shape[0] // 2, magnitude.shape[1] // 2
                # axis = 1
                spectrum_negative, spectrum_positive = magnitude[:, :hw], magnitude[:, hw:]
                magnitude = np.concatenate((spectrum_positive, spectrum_negative), axis=1)
                # axis = 0
                spectrum_negative, spectrum_positive = magnitude[:hh, :], magnitude[hh:, :]
                magnitude = np.concatenate((spectrum_positive, spectrum_negative), axis=0)
                # self.image_blocks_magnitude[i, j, ...] = magnitude
                # собираем изображение
                block_dft = np.multiply(magnitude, np.exp(self.image_blocks_phase[i, j, ...] * 1j))
                self.image_blocks[i, j, ...] = np.abs(np.fft.ifft2(block_dft))

    def foreach_dct(self):
        """
        Выполняет дискретное преобразование косинусов для каждого блока
        :return:
        """
        self.image_blocks_magnitude = np.zeros(self.image_blocks.shape)
        for i in range(self.nrows):
            for j in range(self.ncols):
                # транспонирование необходимо для того, чтобы сумма модулей первого столбца была меньше суммы модулей
                # первой строки. Важное свойство!
                self.image_blocks_magnitude[i, j, ...] = np.transpose(dct(self.image_blocks[i, j, ...],
                                                                          **self.dct_params))

    def foreach_idct(self):
        """
        Выполняет обратное дискретное преобразование косинусов
        :return:
        """
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.image_blocks[i, j, ...] = idct(np.transpose(self.image_blocks_magnitude[i, j, ...]),
                                                    **self.dct_params)

    def get_wm_image(self):
        """
        Собираем изображение со вставленным водяным знаком. Добавляет границы до размеров оригинального изображения.
        :return:
        """
        h, w = self.image_clone.shape
        wm_im = np.concatenate((self.image_clone, self.image[h:, :w]), axis=0)
        wm_im = np.concatenate((wm_im, self.image[:, w:]), axis=1)
        return wm_im


class SiPDataGrid(CoverGrid):
    """
    Класс содержит методы чтения и записи данных для алгрится SiP
    """
    def __init__(self, image):
        CoverGrid.__init__(self, image=image)
        self.AvgB = None
        self.AvgR = None
        self.D = None
        self.c_opt = 1.0  # аддитивная константа, повышающая надежность записи

    @staticmethod
    def __get_ellipse_mask(image_shape, n_step=0):
        """
        рисует эллипсную маску
        :param image_shape: размеры изображения
        :param n_step: отступ в пикселях от
        :return:
        """
        mask = np.zeros(image_shape, dtype=int)
        h, w = image_shape
        x_0, y_0 = (w - 1) / 2, (h - 1) / 2
        a, b = (w - 1) / 2 - n_step, (h - 1) / 2 - n_step
        # отображение x -> y
        x = np.arange(n_step, np.floor(x_0))
        y = np.round(y_0 - b * np.sqrt(1 - ((x - x_0) / a) ** 2))
        x, y = x.astype(int), y.astype(int)
        for j, i in zip(x, y):
            mask[i, j] = 1
            mask[h - 1 - i, j] = 1
            mask[i, w - 1 - j] = 1
            mask[h - 1 - i, w - 1 - j] = 1
        # отображение y -> (x, -x)
        y = np.arange(n_step, np.floor(y_0))
        x = np.round(x_0 - a * np.sqrt(1 - ((y - y_0) / b) ** 2))
        x, y = x.astype(int), y.astype(int)
        for j, i in zip(x, y):
            mask[i, j] = 1
            mask[h - 1 - i, j] = 1
            mask[i, w - 1 - j] = 1
            mask[h - 1 - i, w - 1 - j] = 1
        return mask

    def get_red_and_ellipsoidal_annuli(self):
        """
        возвращает маски красного и синего эллипсоидных колец
        :return:
        """
        # красное эллипсное кольцо
        ell_mask_red = self.__get_ellipse_mask((self.block_height, self.block_width), n_step=0)
        ell_mask_red += self.__get_ellipse_mask((self.block_height, self.block_width), n_step=1)
        # синие эллипсное кольцо
        ell_mask_blue = ell_mask_red.copy()
        ell_mask_blue += self.__get_ellipse_mask((self.block_height, self.block_width), n_step=2)
        ell_mask_blue += self.__get_ellipse_mask((self.block_height, self.block_width), n_step=3)
        # float -> bool
        ell_mask_red = ell_mask_red.astype(bool)
        ell_mask_blue = ell_mask_blue.astype(bool)
        # fill holes
        ell_mask_red = remove_small_holes(ell_mask_red)
        ell_mask_blue = remove_small_holes(ell_mask_blue)
        # вычитаем из синего элипсоидного кольца красное
        ell_mask_blue = np.logical_xor(ell_mask_blue, ell_mask_red)
        '''
        ell_mask_red = np.ones((self.block_height, self.block_width), dtype=int)
        ell_mask_red[2:-2, 2:-2] = 0
        ell_mask_blue = np.ones((self.block_height, self.block_width), dtype=int)
        ell_mask_blue[4:-4, 4:-4] = 0
        ell_mask_blue = ell_mask_blue - ell_mask_red
        ell_mask_red = ell_mask_red.astype(bool)
        ell_mask_blue = ell_mask_blue.astype(bool)
        '''
        return ell_mask_red, ell_mask_blue

    def red_blue2d(self):
        """
        Вычисление значений "красного" и "синего" эллипсоидов
        Сейчас пусть будут прямоугольники, потом переделать в эллипсы: Circle Generation Algorithm
        :return:
        """
        # маска для выделения эллипсоидов
        ell_mask_red, ell_mask_blue = self.get_red_and_ellipsoidal_annuli()

        self.AvgB = np.zeros((self.nrows, self.ncols))
        self.AvgR = np.zeros((self.nrows, self.ncols))

        for i in range(self.nrows):
            for j in range(self.ncols):
                self.AvgR[i, j] = self.image_blocks_magnitude[i, j, ell_mask_red].mean()
                self.AvgB[i, j] = self.image_blocks_magnitude[i, j, ell_mask_blue].mean()
        # Вычисляем матрицу D
        self.D = self.AvgB - self.AvgR

    def write2d(self, bitmap):
        """
        Записываем двумерное представления водяного знака в D матрицу in bitmap
        :return:
        """
        # максимальное значение для каждой строки
        max_d = np.min(self.D, axis=1)
        max_d[max_d > 0] = 0
        max_d = np.abs(max_d)
        # Пишем биты
        ell_mask_red, _ = self.get_red_and_ellipsoidal_annuli()
        for i, j in bitmap.keys():
            self.image_blocks_magnitude[i, j, ell_mask_red] += self.AvgB[i, j] - self.AvgR[i, j] + max_d[i] + self.c_opt

    def read_d(self):
        """
        Читаем водяной знак на D матрице
        :return:
        """
        return np.argmin(self.D, axis=1).tolist()


class WmSiP(SiPDataGrid):
    """
    Реализания алгоритма добавление/чтения водяных знаков наизображении
    Описание алгоритма дано в оригинальной статье:
    Maria Chroni, Angelos Fylakis, and Stavros D. Nikolopoulos,
    "WATERMARKING IMAGES IN THE FREQUENCY DOMAIN BY EXPLOITING SELF-INVERTING PERMUTATIONS"
    """
    def __init__(self, image):
        # TODO: почему необходимо масшабировать до диапазона [0, 1]?
        # scale to range [0, 1]
        image_cl = image.copy()
        img_max, img_min = np.max(image_cl.ravel()), np.min(image_cl.ravel())
        image_cl = (image_cl - img_min) / (img_max - img_min)
        SiPDataGrid.__init__(self, image=image_cl)
        self.wm = None  # водяной знак
        self.A = {}  # матричное представление водяного знака
        self.name = 'SiP'

    def wm_write(self, watermark, c=1.0):
        """
        Запись водяного знака на изображение.
        :return:
        """
        self.wm = watermark
        self.c_opt = c
        # ПРОВЕРИТЬ НА ВОЗМОЖНОСТЬ ДАННОГО РАЗБИЕНИЯ
        self.__get_2d_wm_rep()  # строим матричное представление
        # создаем разбиение изображения на блоки
        im_h, im_w = self.image.shape
        n = len(watermark)
        SiPDataGrid.create_grid(self, height=im_h // n, width=im_w // n)
        # дискретное Фурье преобразование каждого блока
        SiPDataGrid.foreach_dft(self)
        # вычисление матрицы D
        SiPDataGrid.red_blue2d(self)
        # записывем водяной знак на изображение
        # скорректировать на запись на нужный цвет
        SiPDataGrid.write2d(self, self.A)
        # обратное дискретное преобразование Фурье
        SiPDataGrid.foreach_idft(self)

    def wm_read(self, shape):
        """
        Чтение записанного на изображение водяного знака.
        :param shape: (len, ) длина водяного знака
        :return:
        """
        wm_len, = shape
        # создаем разбиение изображения на блоки
        im_h, im_w = self.image.shape
        SiPDataGrid.create_grid(self, height=im_h // wm_len, width=im_w // wm_len)
        # дискретное Фурье преобразование каждого блока
        SiPDataGrid.foreach_dft(self)
        # вычисление матрицы D
        SiPDataGrid.red_blue2d(self)
        # читаем записанный водяной знак
        self.wm = SiPDataGrid.read_d(self)
        # проверить водяной знак на допустимость
        self.__get_2d_wm_rep()
        return self.wm

    def __get_2d_wm_rep(self):
        """
        Строится двумерное представление водяного знака. Проверяется его допустимое значение.
        :return:
        """
        # Водяной знак содержит нумерацию с нуля!
        for (i, p_i) in enumerate(self.wm):
            self.A[(i, p_i)] = 1
        # в каждой строке матрицы должно быть только одно число
        n = len(self.wm)
        for i in range(n):
            row_sum = sum([self.A.get((i, j), 0) for j in range(n)])
            if row_sum != 1:
                raise BadWatermark
        # матрица должна быть симметричной
        for i, j in self.A.keys():
            if (j, i) not in self.A.keys():
                raise BadWatermark

    def get_wm_image(self):
        """
        возвращаме изображение с водяными знаками
        :return:
        """
        img = SiPDataGrid.get_wm_image(self)
        # scale to range [0, 255]
        img *= 255
        img[img < 0] = 0
        img[img > 255] = 255
        return np.round(img).astype(np.uint8)


class WmLuComp(CoverGrid):
    """
    Реализания алгоритма добавление/чтения водяных знаков наизображении
    Описание алгоритма дано в оригинальной статье:
    M. Yesilyurt, Y. Yalman, A. T. Ozcerit,
    "A New DCT Based Watermarking Method Using Luminance Component",
    ELEKTRONIKA IR ELEKTROTECHNIKA , ISSN  1392-1215 , VOL . 19 , NO . 4 , 2013
    http://dx.doi.org/10.5755/j01.eee.19.4.2015
    """
    def __init__(self, image):
        # convert color space from RGB to YCbCr
        y, self.cb, self.cr = _ycc(image)
        CoverGrid.__init__(self, image=y)
        # так как известны размеры блоков, то в конструкторе можем сразу сделать разбиение на блоки
        # и выполнить дискретное преобразование косинусов - последнее сложнее реализовать
        CoverGrid.create_grid(self, height=8, width=8)
        self.name = 'LuComp'

    def wm_write(self, watermark, c=1.0):
        """
        Запись водяного знака на изображение
        :param watermark:
        :param c:
        :return:
        """
        # TODO: проверить допустимость водяного знака
        # дискретное преобразование косинусов
        self.foreach_dct()
        # запись изображения.
        # TODO: Работа с усредненными элементаки не работает!
        for i in range(self.nrows):
            for j in range(self.ncols):
                m = self.image_blocks_magnitude[i, j, ...]
                if watermark[i, j] == 1:
                    if m[4, 1] < m[3, 2]:
                        m[4, 1], m[3, 2] = m[3, 2], m[4, 1]
                    m[4, 1] += c
                else:
                    if m[3, 2] < m[4, 1]:
                        m[4, 1], m[3, 2] = m[3, 2], m[4, 1]
                    m[3, 2] += c
        # обратное дискретное преобразовине косинусов: водяной знак записан
        self.foreach_idct()

    def wm_read(self, shape=(None, None)):
        """
        Читает записанный водяной знак
        :param shape: (nrows, ncols) количество строк в водяном знаке, если None, то максимально возможное количество
        :return: бинарное изображение водяного знака
        """
        nrows, ncols = shape
        nrows = nrows if nrows else self.nrows
        ncols = ncols if ncols else self.ncols

        watermark = np.zeros((nrows, ncols), dtype=np.uint8)
        # дискретное преобразование косинусов
        self.foreach_dct()
        # чтение
        for i in range(nrows):
            for j in range(ncols):
                m = self.image_blocks_magnitude[i, j, ...]
                watermark[i, j] = 1 if m[4, 1] > m[3, 2] else 0
        return watermark

    def get_wm_image(self):
        """
        Выдает изображение в RGB цветовой схеме
        :return:
        """
        y_wm = CoverGrid.get_wm_image(self)
        return _rgb(y_wm, self.cb, self.cr)


class Watermark:
    """
    Добавляет водяной знак на изображение.
    """
    def __init__(self, image, algorithm):
        self.image = image
        self.alg = []
        if algorithm == 'SiP':
            self.alg = [WmSiP(self.image[:, :, i]) for i in range(self.image.shape[2])]
        elif algorithm == 'LuComp':
            self.alg = [WmLuComp(self.image)]
        else:
            raise UnknownAlgorithm

    def watermark_write(self, watermark, c=1.0):
        """
        запись водяного знака
        :param watermark:
        :param c
        :return:
        """
        # пишем водяной знак
        for alg in self.alg:
            alg.wm_write(watermark, c)
        # возвращаем изображение
        im_wm = []
        for alg in self.alg:
            im_wm.append(alg.get_wm_image())
        # если изображение чернобелое или сразу собрано, то просто возвращаем его
        if len(im_wm) == 1:
            return im_wm[0]
        # иначе собираем изобаржение
        # TODO: сложно выглядит, надо попробовать через concatenate
        res_img = np.zeros(self.image.shape, dtype=np.uint8)
        for i in range(len(im_wm)):
            res_img[:, :, i] = im_wm[i]
        return res_img

    def watermark_read(self, shape=(None, None)):
        """
        чтение водяного знака
        :param algorithm:
        :param shape:
        :return:
        """
        watermark = []
        for alg in self.alg:
            try:
                watermark.append(alg.wm_read(shape))
            except BadWatermark:
                watermark.append(None)
        return  watermark


if __name__ == '__main__':
    import skimage.data
    # from skimage.io import imshow
    from skimage.color import rgb2gray
    # тестовый водный знак
    # p = [5, 6, 9, 8, 1, 2, 7, 4, 3]
    p = [6, 3, 2, 4, 5, 1]
    p = [x-1 for x in p]
    # тестовое изображение
    img = skimage.data.astronaut()
    img_gray = rgb2gray(img)
    # img_gray = img[:, :, 0]
    # создаем экземпляр класса с изображением im
    wm_sip = WmSiP(img_gray)
    # записываем на изображение водяной знак
    wm_sip.wm_write(p, c=1.0)
    # читаем изображение с водяным знаком
    wm_img1 = wm_sip.get_wm_image()
    # вычисляем различие между изображениями    
    delta_img = wm_img1 - img_gray
    print(delta_img.ravel().sum())
    # Читаем водяной знак
    wm_sip = WmSiP(wm_img1)
    try:
        s = wm_sip.wm_read((len(p),))
    except BadWatermark:
        print('watermark not found')
    else:
        print('sip', s)

    # создаем случайное бинарное изображение
    bitmap = np.random.randint(2, size=(img.shape[0] // 8, img.shape[1] // 8))
    # write watermark
    wm_lum = WmLuComp(img)
    wm_lum.wm_write(bitmap, c=2.0)
    wm_img2 = wm_lum.get_wm_image()
    # read watermark
    wm_lum = WmLuComp(wm_img2)
    wm_bitmap = wm_lum.wm_read()
    err = np.abs(bitmap - wm_bitmap)
    print('error:', err.ravel().sum(), err.ravel().sum() / len(err.ravel()))
    # skimage.io.imsave('tmp.tiff', wm_img)
    # skimage.io.imshow(wm_img)
    print('Try Watermark class')
    # SiP
    wm = Watermark(img, 'SiP')
    wm_img3 = wm.watermark_write(p, c=1.5)
    wm = Watermark(wm_img3, 'SiP')
    print('Watermark:', wm.watermark_read((len(p),)))
    # LuComp    
    wm = Watermark(img, 'LuComp')
    wm_img4 = wm.watermark_write(bitmap, c=2.5)
    wm = Watermark(wm_img4, 'LuComp')
    wm_bitmap = wm.watermark_read()
    err = np.abs(bitmap - wm_bitmap)
    print('Watermark error:', err.ravel().sum(), err.ravel().sum() / len(err.ravel()))

