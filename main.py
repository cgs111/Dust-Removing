

import cv2
import numpy as np


def main():
    img = cv2.imread("./test.jpg")
    h, w, c = img.shape
    # if h > w:
    #     img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # h, w, c = img.shape
    # img = cv2.resize(img, (1000, int(1000 * h / w)))
    blurimg = cv2.fastNlMeansDenoisingColored(img, None, 50, 50, 11, 25)
    kernel = np.array([[-1, -1, -1], 
                       [-1, 9,-1], 
                       [-1, -1, -1]])
    dst = cv2.filter2D(blurimg, -1, kernel)
    cv2.imshow("", dst)
    
if __name__ == '__main__':
    main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()