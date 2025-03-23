import cv2
import numpy as np

capture = cv2.VideoCapture(0)

while True:
    no_errors, frame_color = capture.read()

    if not no_errors:
        print('Error while read!')
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

    cv2.imshow('frame', frame_color)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()