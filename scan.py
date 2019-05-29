from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help="Path to the image to be scanned")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height =500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
gray = cv2.GaussianBlur(gray, (5,5),0)
edged = cv2.Canny(gray, 75, 200)

print("STEP 1")
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(cv2.contourArea)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.1 * peri, True)
	print(len(approx))
	if len(approx)==4:
		screenCnt = approx
		break


print("Step 2")
cv2.drawContours(image,[screenCnt],-1,(0,255,0),2)
cv2.imshow("outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()




warped = four_point_transform(orig, screenCnt.reshape(4,2)* ratio)

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset =10, method ="gaussian")
warped = (warped > T).astype("uint8") * 255
cv2.imshow("original", imutils.resize(orig, height =650))
cv2.imshow("scanned", imutils.resize(warped, height =650))
cv2.waitKey(0)
cv2.destroyAllWindows()
