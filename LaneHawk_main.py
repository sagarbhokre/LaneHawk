import cv2, os
import numpy as np
import cPickle
#import pdb; pdb.set_trace()

UNDISTORT_EXAMPLES = False
PREPROCESS_EXAMPLES = False

import glob
CHESSBOARD_WIDTH  = 9
CHESSBOARD_HEIGHT = 6
DEBUG = 0
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
def calibrate_camera():
    imgpoints, objpoints = [], []
    resfile = 'cam_calib_params.pkl'

    if(os.path.isfile("./" + resfile)):
        print ("File exists :" + "./" + resfile)
        print ("Loading already saved calib params")
        data = cPickle.load(open(resfile, "rb"))
        return data

    # 1. Read calibration images
    for filename in glob.glob("camera_cal/*"):
        img = cv2.imread(filename)
        if DEBUG == 1:
            cv2.imshow("Calibration images", img)
            cv2.waitKey(100)

        # 2. Compute calibration coefficients
        # 2a. Convert to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        print(gray.shape[::-1])
 
        # 2b. Generate object points
        objp = np.zeros((CHESSBOARD_WIDTH*CHESSBOARD_HEIGHT, 3), np.float32)
        objp[:,:2] = np.mgrid[0:CHESSBOARD_WIDTH, 0:CHESSBOARD_HEIGHT].T.reshape(-1, 2)
    
        # 2b. Detect chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT),None)

        # 2c. Store object points and image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
        else:
            print ("Failed to detect chessboard corners for file: " + filename)

        if DEBUG == 1:
            img_cb = cv2.drawChessboardCorners(img, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), corners, ret)
            cv2.imshow("Chessboard corners detected", img_cb)
            cv2.waitKey(0)
    
    # 2c. Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (IMAGE_WIDTH, IMAGE_HEIGHT),None,None)
    print (ret, mtx, dist)

    for filename in glob.glob("camera_cal/*"):
        img = cv2.imread(filename)

        # 2d. Undistort input image
        dst = cv2.undistort(img, mtx, dist, None, mtx)

        # 2e. Show undistorted image
        cv2.imshow("Undistorted image", dst)
        cv2.waitKey(50)

    # Save calibration params
    with open(resfile,'w') as f:
        print "Saving calib params to pkl file"
        cPickle.dump([mtx, dist], f)

    return [mtx, dist]

def example_image_generator():
    for filename in glob.glob("test_images/*"):
        img = cv2.imread(filename)
        yield filename, img

def hstack_img(src1, src2):
    if(len(src2.shape) == 2):
        src2_img = np.zeros((src2.shape[0], src2.shape[1], 3), dtype=np.uint8)
        src2_img[:,:, 0] = src2 * 255
        src2_img[:,:, 1] = src2 * 255
        src2_img[:,:, 2] = src2 * 255
        src2 = src2_img
    if(len(src1.shape) == 2):
        src1_img = np.zeros((src1.shape[0], src1.shape[1], 3), dtype=np.uint8)
        src1_img[:,:, 0] = src1 * 255
        src1_img[:,:, 1] = src1 * 255
        src1_img[:,:, 2] = src1 * 255
        src1 = src1_img
    dst = np.concatenate((src1, src2), axis=1)
    dst = cv2.resize(dst, src1.shape[:2][::-1])
    return dst

def undistort_example_images(mtx, dist):
    gen = example_image_generator()
    try:
        while(1):
            filename, img = gen.next()
            dst = cv2.undistort(img, mtx, dist, None, mtx)

            cv2.imshow("Input image : Undistorted image", hstack_img(img, dst))
            print ("Saving undistorted images to file : undistorted_"+filename)
            cv2.imwrite("undistorted_" + filename, dst)
            cv2.waitKey(0)
    except StopIteration:
        pass
    return

# applies Sobel x or y, then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if(orient == 'x'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output
    
def mag_thresh(img, sobel_kernel=9, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # 3) Calculate the magnitude 
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobelxy = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    
    # 5) Create a binary mask where mag thresholds are met
    thresh_min = mag_thresh[0]
    thresh_max = mag_thresh[1]
    mbinary = np.zeros_like(scaled_sobelxy)
    mbinary[(scaled_sobelxy >= thresh_min) & (scaled_sobelxy <= thresh_max)] = 1
    
    # 6) Return this mask as your binary_output image
    return mbinary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

# Choose a Sobel kernel size
ksize = 7 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions

def preprocess_image(img):
    # Run the function
    gradx = np.zeros_like(img[:, :, 0])
    grady = np.zeros_like(img[:, :, 0])
    mag_binary = np.zeros_like(img[:, :, 0])
    dir_binary = np.zeros_like(img[:, :, 0])
    hls_binary = np.zeros_like(img[:, :, 0])
    
    #gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(100, 255))
    #grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(120, 200))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(90, 250))
    #dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(np.pi/4, np.pi/2))

    #hls_binary = hls_select(img, (200, 255))

    combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1)) | (hls_binary  == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined[((gradx == 1) & (grady == 1)) | (hls_binary  == 1) | ((mag_binary == 1) | (dir_binary == 1))] = 1

    cv2.imshow("Preprocessing sobel x", hstack_img(img, combined))
    cv2.waitKey(0)
    return combined

def preprocess_example_images():
    gen = example_image_generator()
    try:
        while(1):
            filename, img = gen.next()
            preprocess_image(img)
    except StopIteration:
        pass

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 1

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    S = hls[:, :, 2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    #import pdb; pdb.set_trace()
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    print "========================================="
    lines = np.squeeze(lines)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    slope_l, intercept_l, slope_r, intercept_r = draw_lines(line_img, lines)
    return line_img, slope_l, intercept_l, slope_r, intercept_r

ROI = {'l':0.01, 't':0.65, 'r':0.01, 'b':0.01, 'tw':0.19} # left, top, right, bottom, trapezoid_minor_width
x1ROI, y1ROI = int(IMAGE_WIDTH/2 * (1 - ROI['tw'])), int(IMAGE_HEIGHT * ROI['t'])
x2ROI, y2ROI = int(IMAGE_WIDTH/2 * (1 + ROI['tw'])), int(IMAGE_HEIGHT * ROI['t'])
x3ROI, y3ROI = int(IMAGE_WIDTH * ( 1 - ROI['r'])), int(IMAGE_HEIGHT * ( 1 - ROI['b']))
x4ROI, y4ROI = int(IMAGE_WIDTH * ROI['l']), int(IMAGE_HEIGHT * ( 1 - ROI['b']))
Minv = []
def visualize_prespective_transform(img):
    global Minv
    src = np.array([[x1ROI, y1ROI], [x2ROI, y2ROI], [x3ROI, y3ROI], [x4ROI, y4ROI]], np.int32)
    img_ROI = region_of_interest(img, [src])
    #cv2.imshow("Image ", img)
    #cv2.imshow("ROI Image ", img_ROI)
    cv2.polylines(img, [src], True, color=(0, 255, 0), thickness=2) 
    #cv2.imshow("Input image : ROI image", hstack_img(img, img_ROI))
    dst = np.array([[x4ROI, y1ROI], [x3ROI, y2ROI], [x3ROI, y3ROI], [x4ROI, y4ROI]], np.float32)
    src = np.float32(src)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img_ROI, M, (IMAGE_WIDTH, IMAGE_HEIGHT), flags=cv2.INTER_LINEAR)
    
    #cv2.imshow("ROI image : Warped image", hstack_img(img, warped))
    return warped

def overlay_on_image(image, left_fitx, right_fitx, yvals):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image).astype(np.uint8)
    color_warp = warp_zero #np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    pts = np.squeeze(pts)
    pts = pts.astype(int)
    #import pdb; pdb.set_trace()
    # Draw the lane onto the warped blank image
    #cv2.imshow("test", color_warp)
    #cv2.waitKey(0)
    cv2.fillPoly(color_warp, [pts], (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    image = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return image

#Edge detection
CANNY_LOW = 50
CANNY_HIGH = 150

# Noise filter
GAUSS_KER = 3

#Hough transform params
HOUGH_THRESHOLD = 10
HOUGH_MIN_LEN = 20
HOUGH_MAX_GAP = 80

#Lane detection slope threshold
SLOPE_THRESHOLD = 0.4
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def slope_line(line):
    dx = line[2] - line[0]
    dy = line[3] - line[1]
    return (dy/(dx+0.000001))

def weighted_img(img, initial_img, a=0.8, b=1., l=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    `initial_img` should be the image before any processing.
    The result image is computed as follows:
    initial_img * alpha + img * beta + lambda
    NOTE: initial_img and img must be the same shape!
    #"""
    return cv2.addWeighted(initial_img, a, img, b, l)


from scipy import stats
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
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
    x_l, y_l, x_r, y_r = [], [], [] ,[]

    y1 = int(1.1 * IMAGE_HEIGHT * ROI['t'])
    y2 = IMAGE_HEIGHT
    W = IMAGE_WIDTH
    XU_OFFSET = (ROI['tw']) * IMAGE_WIDTH/2
    DEBUG = 0
    if (DEBUG == 1):
        for (xi1, yi1, xi2, yi2) in lines:
            cv2.line(img, (xi1, yi1), (xi2, yi2), color, thickness=2)
        cv2.imshow("input lines w/o filtering", img)

    for idx, line in enumerate(lines):
        slope = slope_line(line)

        # Filter out left lane markers ensuring none of the points cross right top x boundary
        if((slope > SLOPE_THRESHOLD) and (line[0] > (W/2 - XU_OFFSET))):
            x_r.append(line[0])
            x_r.append(line[2])
            y_r.append(line[1])
            y_r.append(line[3])
        # Filter out right lane markers ensuring none of the points cross left top x boundary
        elif((slope < (-1 * SLOPE_THRESHOLD)) and (line[0] < (W/2 + XU_OFFSET))):
            x_l.append(line[0])
            x_l.append(line[2])
            y_l.append(line[1])
            y_l.append(line[3])

    # Plot left lane marker
    if((len(x_l) != 0) and (len(y_l) != 0)):
        slope_l, intercept_l, r_value, p_value, std_err = stats.linregress(x_l,y_l)
        x1 = int((y1 - intercept_l)/slope_l)
        x2 = int((y2 - intercept_l)/slope_l)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness=8)

    # Plot right lane marker
    if((len(x_r) != 0) and (len(y_r) != 0)):
        slope_r, intercept_r, r_value, p_value, std_err = stats.linregress(x_r,y_r)
        x1 = int((y1 - intercept_r)/slope_r)
        x2 = int((y2 - intercept_r)/slope_r)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness=8)

    # Try to detect both lane markers and return 1 if either/both are missing
    if((len(x_l) == 0) or (len(x_r) == 0)):
        if(DEBUG == 1):
            for line in lines:
                for (x1, y1, x2, y2) in line:
                    cv2.line(img, (x1, y1), (x2, y2), color=[255, 255, 255], thickness=2)

        missing_markers = (len(x_l) == 0) + (len(x_r) == 0)
        print(str(missing_markers) + "lane markers not detected")
        cv2.waitKey(0)
        return 0, 0, 0, 0

    #return 0 if both lane markers were detected
    return slope_l, intercept_l, slope_r, intercept_r

# main
if __name__ == '__main__':
    # Calibrate camera
    mtx, dist = calibrate_camera()

    # Show undistorted images and save them
    if UNDISTORT_EXAMPLES:
        undistort_example_images(mtx, dist)

    # TODO: Create a writeup and refer to undistorted images in it

    # Color, gradient-mag, gradient-dir and other methods for generating binary image
    if PREPROCESS_EXAMPLES:
        preprocess_example_images()

    # Perspective transform to get parallel lines for lane markers 
    gen = example_image_generator()
    count = 0
    try:
        while(1):
            filename, img = gen.next()
            print "Processing frame " + str(count) + " :: " + filename + " :: "
            count += 1
            img = cv2.undistort(img, mtx, dist, None, mtx)
            pp_img = preprocess_image(img)  
            continue
            ROI_warped = visualize_prespective_transform(pp_img)

            '''
            # Convert to greyscale to avoid further computations
            image_grey = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
            # HSV representation of yellow is [30, 255, 255]
            # filtering out yellow with some delta colors around it
            hyl = np.array([20, 50, 50], dtype = "uint8")
            hyu = np.array([40, 255, 255], dtype="uint8")
        
            # Create a mask to extract yello color from frame
            image_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
            mask_y = cv2.inRange(image_hsv, hyl, hyu)

            #cv2.imshow("Y mask" , mask_y)
            # Extract white/bright color from greyscale image. This will give data for while lane markers
            mask_w = cv2.inRange(image_grey, 200, 255)
        
            #cv2.imshow("W mask" , mask_w)
            # Combine the mask for white and yellow
            mask = cv2.bitwise_or(mask_w, mask_y)
        
            #cv2.imshow("mask" , mask)
            # Extract white and yellow colored pixels from image frame
            image_yw_mask = cv2.bitwise_and(image_grey, mask)
            #cv2.imshow("img yw mask" , image_yw_mask)
        
            # From grey image extract edges
            image_canny = canny(image_yw_mask, CANNY_LOW, CANNY_HIGH)
            #cv2.imshow("img canny" , image_canny)
            # Blur out image to reduce effect of noise
            '''
            image_gb = gaussian_blur(ROI_warped, GAUSS_KER)

            #cv2.imshow("img ROI warped" , ROI_warped)

            image_gb = image_gb.astype(np.uint8)
            ROI_warped = ROI_warped.astype(np.uint8)

            cv2.imshow("All", hstack_img(hstack_img(img[:, :, 0], pp_img), hstack_img(ROI_warped, image_gb)))
            # Find lines(lane markers) in the image which fit the edges detected by canny filter
            image_line, sl, il, sr, ir = hough_lines(ROI_warped, 1, np.pi/180, HOUGH_THRESHOLD, HOUGH_MIN_LEN, HOUGH_MAX_GAP);

            yvals = np.linspace(y2ROI, y3ROI, num=3)
            left_x, right_x = [], []
            for y in yvals:
                left_x.append((y - il)/sl)
                right_x.append((y - ir)/sr)
            # overlay the lane markers on original image
            image_weighted = weighted_img(image_line, img)

            output = overlay_on_image(img, left_x, right_x, yvals)
            cv2.imshow("Input: Output", hstack_img(img, output))
            cv2.waitKey(1000)
            #break
    except StopIteration:
        pass

# Detect lane markers and save them to the folder and add to writeup

# polynomial coefficient fitting on lane markers

# Calculate radius of curvature of road in meter

# Plot result back on the road such that lane markings are correctly tracked

# Run the algo on video and store output to another video

# Discuss problems/issues. Possibility of failure. How cant it be made robust?
