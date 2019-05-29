from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import re

def winco_receipt_line(line):
	s = re.search(r'((I|T)F) | ((I|T)x)', line)
	if( s is None):
		return None
	TF_ind = s.start()

	output_string = ', '.join((re.sub(r'\W+', '', line[0:20]), ' ', line[TF_ind-5:TF_ind].rstrip()))

	return output_string


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help="Path to the image to be scanned")
ap.add_argument("-s", "--scanned", required = False, default='scanned', help="where to save the scanned image")
ap.add_argument("-c", "--csv", required = False, help="where to save the scanned image")
ap.add_argument("-w", "--which_receipt", required = False,default='winco', help="where to save the scanned image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height =500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
gray = cv2.GaussianBlur(gray, (5,5),0)
#gray = cv2.equalizeHist(gray)
edged = cv2.Canny(gray, 75, 200)

print("STEP 1")
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
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


sp = args['image'].split('.')
save_filename = sp[0]+'_'+args['scanned'] +'.'+sp[1]
cv2.imwrite(save_filename, warped)

csv_filename = sp[0] +'.csv'
csv_file = open(csv_filename,"w")

if(args['which_receipt'] == 'winco'):
	process_line = winco_receipt_line
im = Image.open(save_filename)
im = im.filter(ImageFilter.MedianFilter())
enhancer = ImageEnhance.Contrast(im)
im = enhancer.enhance(2)
im = im.convert('1 eng')
st = pytesseract.image_to_string(im, config="-psm 6")
for cur_line in st.split('\n'):
	print(cur_line)
	ret = process_line(cur_line)

	if( ret is None):
		continue

	cs_file.write(ret +'\n')
csv_file.close()
