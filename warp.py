import cv2 as cv
import numpy as np
import scipy.ndimage as ndimage

def panorama(h, from_img, to_img, padding=2400, delta=2400):
    """
        Создать горизонтальную панораму путем объединения двух изображений
        с применением гомографии H (вычисленной методом RANSAC). В результате
        получится изображение такой же высоты, как и to_img.
        :param h: гомография, вычисленная методом RANSAC
        :param from_img: первое изображение
        :param to_img: второе изображение
        :param padding: число пикселей заполнения
        :param delta: дополнительный параллельный перенос
        :return: панорамное фото, сост. из двух данных
    """

    is_color = len(from_img.shape) == 3

    # гомографичекое преобразование для функции geometric_transform (scipy.ndimage)
    def transform(p):
        p2 = np.dot(h, [p[0], p[1], 1])
        return (p2[0]/p2[2], p[1]/p[2])

    if (h[1, 2] < 0): # from_img справа
        print("warp right")
        # преобразовать from_img
        if is_color:
            # дополнить конечное изображение нулями справа
            to_img_t = np.hstack((to_img, np.zeros((to_img.shape[0], padding, 3))))
            from_img_t = np.zeros((to_img.shape[0], to_img.shape[1]+padding, to_img.shape[2]))
            for col in range(3):
                from_img_t[:, :, col] = ndimage.geometric_transform(from_img[:, :, col],
                                                                    transform,
                                                                    (to_img.shape[0], to_img.shape[1]+padding))
        else:
            # дополнить конечное изображение нулями справа
            to_img_t = np.hstack((to_img, np.zeros((to_img.shape[0], padding))))
            from_img_t = ndimage.geometric_transform(from_img, transform,
                                                     (to_img.shape[0], to_img.shape[1]+padding))
    else:
        print("warp left")
        # добавить паралелльный перенос для компенсации дополнения слева
        h_delta = np.array([[1, 0, 0], [0, 1, -delta], [0, 0, 1]])
        h = np.dot(h, h_delta)

        # преобразовать from_img
        if is_color:
            # дополнить конечное изображение нулями справа
            to_img_t = np.hstack((np.zeros((to_img.shape[0], padding, 3)), to_img))
            from_img_t = np.zeros((to_img.shape[0], to_img.shape[1] + padding, to_img.shape[2]))
            for col in range(3):
                from_img_t[:, :, col] = ndimage.geometric_transform(from_img[:, :, col],
                                                                    transform,
                                                                    (to_img.shape[0], to_img.shape[1] + padding))
        else:
            # дополнить конечное изображение нулями справа
            to_img_t = np.hstack((np.zeros((to_img.shape[0], padding)), to_img))
            from_img_t = ndimage.geometric_transform(from_img, transform,
                                                     (to_img.shape[0], to_img.shape[1] + padding))

    # объединить и вернуть (поместив from_img над to_img)
    if is_color:
        # все нечерные пикселы
        alpha = ((from_img_t[:, :, 0] * from_img_t[:, :, 1] + from_img_t[:, :, 2]) > 0)
        for col in range(3):
            to_img_t[:, :, col] = from_img_t[:, :, col] * alpha + to_img_t[:, :, col] * (1 - alpha)
    else:
        alpha = (from_img_t > 0)
        to_img_t = from_img_t * alpha + to_img_t * (1 - alpha)

    return to_img_t


