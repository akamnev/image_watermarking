"""
Реализованы алгоритмы добавления водяных знаков на изображение
"""
import skimage.util
import numpy as np


class BadWatermarking(Exception):
    """Класс ошибки некорректного водяного знака."""
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
        self.image_blocks = skimage.util.view_as_blocks(self.image_clone, (self.block_height, self.block_width))

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
        self.Pr = 2  # ширина "красного" эллипсоида
        self.Pb = 2  # ширина "синего" эллипсоида
        self.c_opt = 0.1  # аддитивная константа, повышающая надежность записи
        self.ell_mask_red = None
        self.ell_mask_blue = None

    def red_blue2d(self):
        """
        Вычисление значений "красного" и "синего" эллипсоидов
        Сейчас пусть будут прямоугольники, потом переделать в эллипсы: Circle Generation Algorithm
        :return:
        """
        # маска для выделения эллипсоидов
        self.ell_mask_red = np.ones((self.block_height, self.block_width), dtype=int)
        self.ell_mask_red[2:-2, 2:-2] = 0
        self.ell_mask_blue = np.ones((self.block_height, self.block_width), dtype=int)
        self.ell_mask_blue[4:-4, 4:-4] = 0
        self.ell_mask_blue = self.ell_mask_blue - self.ell_mask_red
        self.ell_mask_red = self.ell_mask_red.astype(bool)
        self.ell_mask_blue = self.ell_mask_blue.astype(bool)

        self.AvgB = np.zeros((self.nrows, self.ncols))
        self.AvgR = np.zeros((self.nrows, self.ncols))

        for i in range(self.nrows):
            for j in range(self.ncols):
                self.AvgR[i, j] = self.image_blocks_magnitude[i, j, self.ell_mask_red].mean()
                self.AvgB[i, j] = self.image_blocks_magnitude[i, j, self.ell_mask_blue].mean()
        # Вычисляем матрицу D
        self.D = self.AvgB - self.AvgR

    def write2d(self, bitmap):
        """
        Записываем двумерное представления водяного знака в D матрицу in bitmap
        :return:
        """
        d = self.D.copy()
        d[d > 0] = 0
        d = abs(d)
        # максимальное значение для каждой строки
        max_d = np.max(d, axis=1)
        # Пишем биты
        for i, j in bitmap.keys():
            self.image_blocks_magnitude[i, j, self.ell_mask_red] += self.AvgB[i, j] - self.AvgR[i, j] + max_d[i] + self.c_opt

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
        SiPDataGrid.__init__(self, image=image)
        self.wm = None  # водяной знак
        self.A = {}  # матричное представление водяного знака

    def wm_write(self, watermark):
        """
        Запись водяного знака на изображение.
        :return:
        """
        self.wm = watermark
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

    def wm_read(self, wm_len):
        """
        Чтение записанного на изображение водяного знака.
        :return:
        """
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
                raise BadWatermarking
        # матрица должна быть симметричной
        for i, j in self.A.keys():
            if (j, i) not in self.A.keys():
                raise BadWatermarking


class WmLuComp:
    """
    Реализания алгоритма добавление/чтения водяных знаков наизображении
    Описание алгоритма дано в оригинальной статье:
    M. Yesilyurt, Y. Yalman, A. T. Ozcerit,
    "A New DCT Based Watermarking Method Using Luminance Component",
    ELEKTRONIKA IR ELEKTROTECHNIKA , ISSN  1392-1215 , VOL . 19 , NO . 4 , 2013
    http://dx.doi.org/10.5755/j01.eee.19.4.2015
    """
    pass


if __name__ == '__main__':
    import skimage.data
    import skimage.io
    import skimage.color
    # тестовый водный знак
    # p = [5, 6, 9, 8, 1, 2, 7, 4, 3]
    p = [6, 3, 2, 4, 5, 1]
    p = [x-1 for x in p]
    # тестовое изображение
    img = skimage.data.astronaut()
    img = skimage.color.rgb2gray(img)
    # создаем экземпляр класса с изображением im
    wm_sip = WmSiP(img)
    # записываем на изображение водяной знак
    wm_sip.wm_write(p)
    # читаем изображение с водяным знаком
    wm_img = wm_sip.get_wm_image()
    # вычисляем различие между изображениями    
    delta_img = wm_img - img
    print(delta_img.ravel().sum())
    # Читаем водяной знак
    wm_sip = WmSiP(wm_img)
    s = wm_sip.wm_read(len(p))
    print('sip', s)
    # skimage.io.imsave('tmp.tiff', wm_img)
    # skimage.io.imshow(wm_img)

    print('Test OK')
