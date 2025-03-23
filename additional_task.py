import cv2
import numpy as np

# Загрузка изображения мухи
fly_image = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)  # Загружаем с альфа-каналом (прозрачность)
if fly_image is None:
    print("Ошибка: изображение мухи не найдено!")
    exit()

# Получаем размеры изображения мухи
fly_height, fly_width = fly_image.shape[:2]

capture = cv2.VideoCapture(0)

while True:
    no_errors, frame_color = capture.read()

    if not no_errors:
        print('Ошибка при чтении кадра!')
        break

    frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
    frame = cv2.blur(frame, (15, 15))
    _, frame = cv2.threshold(frame, 200, 250, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    frame = cv2.dilate(frame, kernel, iterations=16)
    conts, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for i in range(0, len(conts)):
        cv2.drawContours(frame_color, conts, i, (0, 255, 0), 10)

    # Нахождение центра метки
    if len(conts) > 0:
        M = cv2.moments(conts[0])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Рисование вертикальной и горизонтальной прямых
            cv2.line(frame_color, (cX, 0), (cX, frame_color.shape[0]), (255, 0, 0), 2)
            cv2.line(frame_color, (0, cY), (frame_color.shape[1], cY), (255, 0, 0), 2)

            # Наложение изображения мухи
            # Вычисляем координаты верхнего левого угла для размещения мухи
            x1 = cX - fly_width // 2
            y1 = cY - fly_height // 2
            x2 = x1 + fly_width
            y2 = y1 + fly_height

            # Проверяем, чтобы координаты не выходили за пределы кадра
            if x1 >= 0 and y1 >= 0 and x2 <= frame_color.shape[1] and y2 <= frame_color.shape[0]:
                # Наложение изображения мухи с учетом прозрачности
                for c in range(0, 3):
                    frame_color[y1:y2, x1:x2, c] = frame_color[y1:y2, x1:x2, c] * (1 - fly_image[:, :, 3] / 255.0) + \
                                                    fly_image[:, :, c] * (fly_image[:, :, 3] / 255.0)

    cv2.imshow('frame', frame_color)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()