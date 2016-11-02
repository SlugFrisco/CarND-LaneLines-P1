import os
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from math import *


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    slopes = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slopes.append([((float(y2)-float(y1))/(float(x2)-float(x1))), line])
    # now we have an array of all line slopes with their respective lines
    # separate out into left and right side
    left_side = [slope for slope in slopes if slope[0] > 0.5]
    right_side = (slope for slope in slopes if slope[0] < -0.5)

    # left lane: get mean x1, y1, x2, y2
    left_lines = [line[1] for line in left_side]
    left_x1s = [x[0][0] for x in left_lines]
    mean_left_x1 = sum(left_x1s) / float(len(left_x1s))
    left_x2s = [x[0][2] for x in left_lines]
    mean_left_x2 = sum(left_x2s) / float(len(left_x2s))
    left_y1s = [y[0][1] for y in left_lines]
    mean_left_y1 = sum(left_y1s) / float(len(left_y1s))
    left_y2s = [y[0][3] for y in left_lines]
    mean_left_y2 = sum(left_y2s) / float(len(left_y2s))

    left_max_x = max(max(left_x1s), max(left_x2s))
    left_min_x = min(min(left_x1s), min(left_x2s))
    left_max_y = max(max(left_y1s), max(left_y2s))
    left_min_y = min(min(left_y1s), min(left_y2s))
    left_slope = float(mean_left_y2 - mean_left_y1) / (mean_left_x2 - mean_left_x1)

    # right lane: get mean x1, y1, x2, y2
    right_lines = [line[1] for line in right_side]
    right_x1s = [x[0][0] for x in right_lines]
    mean_right_x1 = sum(right_x1s) / float(len(right_x1s))
    right_x2s = [x[0][2] for x in right_lines]
    mean_right_x2 = sum(right_x2s) / float(len(right_x2s))
    right_y1s = [y[0][1] for y in right_lines]
    mean_right_y1 = sum(right_y1s) / float(len(right_y1s))
    right_y2s = [y[0][3] for y in right_lines]
    mean_right_y2 = sum(right_y2s) / float(len(right_y2s))

    right_max_x = max(max(right_x1s), max(right_x2s))
    right_min_x = min(min(right_x1s), min(right_x2s))
    right_max_y = max(max(right_y1s), max(right_y2s))
    right_min_y = min(min(right_y1s), min(right_y2s))
    right_slope = float(mean_right_y2 - mean_right_y1) / (mean_right_x2 - mean_right_x1)

    # get top and bottom of lane
    global_max_y = max(right_max_y, left_max_y)
    global_min_y = min(right_min_y, left_min_y)

    # draw left line; extrapolate out
    cv2.line(img, (int(left_max_x + float(global_max_y - left_max_y)/float(left_slope)), global_max_y),
             (int(left_min_x + (float(global_min_y - left_min_y)/float(left_slope))), global_min_y), color, thickness)
    # draw right line; extrapolate out
    cv2.line(img, (int(right_max_x + ((float(global_min_y - right_min_y))/(float(right_slope)))), global_min_y),
             (int(right_min_x + float(global_max_y - right_max_y)/float(right_slope)), global_max_y), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    # this had to be 8-bit grayscale to use Hough transform; make it color again
    # so that draw_lines() can draw in red
    line_img_c = cv2.cvtColor(line_img, cv2.COLOR_GRAY2BGR)
    try:
        draw_lines(line_img_c, lines)
    except:
        # sometimes we just can't draw good lines (yet!), so do nothing (draw a blank layer)
        pass
    return line_img_c


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * alpha + img * beta + gamma
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)

def run_everything(image, video_mode = False):

    # reading in an image
    # but no need for process_video because what's fed in is already a numpy array
    # image = (mpimg.imread(image_path)*255).astype('uint8')
    if video_mode == False:
        image = mpimg.imread(image)
    else:
        pass
    # get image dimensions for use later
    ysize = image.shape[0]
    xsize = image.shape[1]
    # printing out some stats and plotting
    print('This image is:', type(image), 'with dimesions:', image.shape)
    plt.imshow(image)  # call as plt.imshow(gray, cmap='gray') to show a grayscaled image
    # plt.show()

    # grayscale the image
    gray = grayscale(image)
    plt.imshow(gray, cmap='gray')
    # plt.show()

    # then apply gaussian blur
    # set kernel size
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # then use Canny: calc gradient, find edges
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    plt.imshow(edges, cmap='gray')
    # plt.show()

    # create the mask

    #define vertices of mask
    vertices = np.array([[
                        (0, ysize),
                        (xsize, ysize),
                        (0.53 * xsize, ysize * 0.6),
                        (0.48 * xsize, ysize * 0.6)]], dtype=np.int32)

    masked_edges = region_of_interest(edges, vertices)

    # now apply Hough transform to locate edges

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 5     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 20 # minimum number of pixels making up a line
    max_line_gap = 2    # maximum gap in pixels between connectable line segments
    # line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image, spit lines back
    # Output "lines" is an array containing endpoints of detected line segments
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

    # try to overlay lines on edges
    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))

    # display mask on top of lines
    poly = np.copy(image)*0 # blank image, same size as original
    filled_poly = cv2.fillPoly(poly, vertices, (255, 0, 0))
    overlay_w_mask = cv2.addWeighted(color_edges, 0.7, filled_poly, 1, 0)
    plt.imshow(overlay_w_mask)
    if video_mode == False:
        plt.show()
    else:
        pass

    # show lines on top of image

    # first turn lines into 3-channel, to match image (redundant, comment out)
    # three_c_lines = cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR)
    lines_edges = cv2.addWeighted(image, 0.6, lines, 1, 0)
    plt.imshow(lines_edges)
    if video_mode == False:
        plt.show()
    else:
        pass

    # needs to return something!!
    return lines_edges




def main():
    for filename in os.listdir('test_images/'):
        # if filename == "solidWhiteRight.jpg":
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # print(os.path.join(directory, filename))
            # save new image
            new_img = run_everything('test_images/' + filename)
            cv2.imwrite(img=new_img, filename="laned_" + filename)
            print("saved!")

        else:
            pass


if __name__ == "__main__":
    main()

