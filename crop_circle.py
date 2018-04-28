import cv2
import numpy as np
import argparse

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
def get_circle(img):
    output = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    method = cv2.HOUGH_GRADIENT
    dp = 2
    minDist = 500
    minRadius = 200


    circles = cv2.HoughCircles(gray, method, dp, minDist, minRadius=minRadius)
    circles = np.round(circles[0, :]).astype('int')

    cropped_circle = circles[0]

    return cropped_circle

def get_cropped_image(img, cropped_circle):
    height, width, depth = img.shape
    mask = np.zeros((height, width), np.uint8)

    x, y , r = cropped_circle

    cv2.circle(mask, (x, y), r, (255, 255, 255), thickness=-1)
    masked_data = cv2.bitwise_and(img, img, mask=mask)
    _,thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    crop = masked_data[y:y+h, x:x+w] 

    return crop   

def main():
    parser = argparse.ArgumentParser(description="Find circle within fundus image")
    parser.add_argument("image", help="Image file")

    args = parser.parse_args()
    image_file = args.image

    img = cv2.imread(image_file)
    img = image_resize(img, height=512)

    circle = get_circle(img)
    cropped_circle = get_cropped_image(img, circle)


    cv2.imshow("cropped", cropped_circle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # if circles is not None:
    #     circles = np.round(circles[0, :]).astype('int')

    #     for (x, y, r) in circles:
    #         cv2.circle(output, (x,y), r, (0, 255, 0), 4)
    #         cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    #     cv2.imshow("output", np.hstack([img, output]))
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()




if __name__ == "__main__":
    main()