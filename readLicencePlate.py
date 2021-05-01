import cv2
import pytesseract
import numpy as np
import imutils

img = cv2.imread('licence_plate.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filtered_img = cv2.bilateralFilter(gray_img, 5, 250, 250)
edge = cv2.Canny(filtered_img, 30, 200)

img_contours = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
grab_contours = imutils.grab_contours(img_contours)
grab_contours = sorted(grab_contours, key=cv2.contourArea, reverse=True)[:10]
plate_coord = None

for cnts in grab_contours:
    eps = 0.018 * cv2.arcLength(cnts, True)
    approx = cv2.approxPolyDP(cnts, eps, True)
    # this is rectangle if len approx == 4
    if len(approx) == 4:
        plate_coord = approx
        break

# The image looks black except the licence plate. This mask using for this.
mask = np.zeros(gray_img.shape, np.uint8)
masked_img = cv2.drawContours(mask, [plate_coord], 0, (255, 255, 255), -1)
plate = cv2.bitwise_and(img, img, mask=mask)

# This param equal to (mask == (255,255,255))
(x, y) = np.where(mask == 255)
(top_x, top_y) = (np.min(x), np.min(y))
(bot_x, bot_y) =(np.max(x), np.max(y))
# crop variable showing only plate area
crop = gray_img[top_x:bot_x+1, top_y:bot_y+1]

# Read the plate values
text = pytesseract.image_to_string(crop, lang='eng')
print('The licence plate' + text)

cv2.imshow("image", crop)

cv2.waitKey(0)
cv2.destroyAllWindows()