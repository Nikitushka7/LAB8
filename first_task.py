import cv2
import numpy as np
from matplotlib import pyplot

img = cv2.imread('variant-8.jpg')  # Открываем изображение
width, height = img.shape[:2]  # Размеры изображения

# Координаты центра
x_center, y_center = width // 2, height // 2

# Новое изображение: квадрат 400 на 400 из центра
new_image = img[x_center - 200:x_center + 200, y_center - 200:y_center + 200]

# Приведем картинку к палитре RGB для корренктного отображения цвета. В ином
# случае получаем картинку с преимущественно синими оттенками
fix_color_img = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
pyplot.imshow(fix_color_img)

# Сохраняем
pyplot.imsave('new_image.jpg', fix_color_img)
# Смотрим на то, что у нас получилось
pyplot.show()