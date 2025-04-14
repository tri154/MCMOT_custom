import cv2

img = cv2.imread("camera1/1_temp.jpg")

f = open("camera1/2d_point.txt")

line = f.readline()
while line:
    line = line.strip()
    if line == '':
        line = f.readline()
        continue
    x, y = line.split(' ')
    x = int(x)
    y = int(y)
    cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=2)
    line = f.readline()

cv2.imwrite('../test_cali_res.jpg', img)