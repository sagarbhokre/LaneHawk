import cv2, os, sys, copy
import numpy as np
import cPickle
from Tkinter import *
import Image, ImageTk

#import pdb; pdb.set_trace()

UNDISTORT_EXAMPLES = False
PREPROCESS_EXAMPLES = False

ym_per_pix = 100 * 30/720 # meters per pixel in y dimension
xm_per_pix = 100 * 3.7/700 # meters per pixel in x dimension

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

def example_image_generator(location):
    for filename in glob.glob(location + "test_images/*"):
        img = cv2.imread(filename)
        yield filename, img

def scale_img_vals(src, factor):
    dst = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
    dst[:,:, 0] = src * factor
    dst[:,:, 1] = src * factor
    dst[:,:, 2] = src * factor
    return dst

def hstack_img(src1, src2):
    if(len(src2.shape) == 2):
        src2 = scale_img_vals(src2, 255)
    if(len(src1.shape) == 2):
        src1 = scale_img_vals(src1, 255)

    dst = np.concatenate((src1, src2), axis=1)
    dst = cv2.resize(dst, src1.shape[:2][::-1])
    return dst

def undistort_example_images(mtx, dist):
    gen = example_image_generator("")
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

# Read configuration parameters

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Run the function
    gradx = np.zeros_like(img[:, :, 0])
    grady = np.zeros_like(img[:, :, 0])
    mag_binary = np.zeros_like(img[:, :, 0])
    dir_binary = np.zeros_like(img[:, :, 0])
    hls_binary = np.zeros_like(img[:, :, 0])

    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(GRADX_MIN, GRADX_MAX))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(GRADY_MIN, GRADY_MAX))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(MAG_MIN, MAG_MAX))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(DIR_MIN, DIR_MAX))

    hls_binary = hls_select(img, (HLS_H_MIN, HLS_H_MAX), (HLS_S_MIN, HLS_S_MAX))

    combined = np.zeros_like(dir_binary)
    combined[(hls_binary == 1) | (((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)))] = 1
    #combined[((gradx == 1) | (grady == 1)) | (hls_binary  == 1) | ((mag_binary == 1) | (dir_binary == 1))] = 1

    ROI_warped = prespective_transform(combined)
    image_gb = gaussian_blur(ROI_warped, GAUSS_KER)
    return image_gb

def preprocess_example_images():
    gen = example_image_generator("")
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
def hls_select(img, thresh_h=(0, 255), thresh_s=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    S = hls[:, :, 2]
    H = hls[:, :, 0]
    binary_output = np.zeros_like(S)
    binary_output[((S > thresh_s[0]) & (S <= thresh_s[1])) & ((H > thresh_h[0]) & (H <= thresh_h[1]))] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    #import pdb; pdb.set_trace()
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    if not lines.all():
        return img, 0,0,0,0
    #if lines.shape[1] > 1:
        #lines = np.squeeze(lines)
    lines = np.reshape(lines, (lines.shape[1], lines.shape[2]))
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    slope_l, intercept_l, slope_r, intercept_r = draw_lines(line_img, lines)
    return line_img, slope_l, intercept_l, slope_r, intercept_r

Minv = []
from LaneHawkConfig import *
x1ROI, y1ROI = int(IMAGE_WIDTH/2 * (1 - ROI['tw'])), int(IMAGE_HEIGHT * ROI['t'])
x2ROI, y2ROI = int(IMAGE_WIDTH/2 * (1 + ROI['tw'])), int(IMAGE_HEIGHT * ROI['t'])
x3ROI, y3ROI = int(IMAGE_WIDTH * ( 1 - ROI['bw'])), int(IMAGE_HEIGHT * ( 1 - ROI['b']))
x4ROI, y4ROI = int(IMAGE_WIDTH * ROI['bw']), int(IMAGE_HEIGHT * ( 1 - ROI['b']))

def update_ROI_coords():
    global x1ROI, y1ROI, x2ROI, y2ROI, x3ROI, y3ROI, x4ROI, y4ROI
    x1ROI, y1ROI = int(IMAGE_WIDTH/2 * (1 - ROI['tw'])), int(IMAGE_HEIGHT * ROI['t'])
    x2ROI, y2ROI = int(IMAGE_WIDTH/2 * (1 + ROI['tw'])), int(IMAGE_HEIGHT * ROI['t'])
    x3ROI, y3ROI = int(IMAGE_WIDTH * ( 1 - ROI['bw'])), int(IMAGE_HEIGHT * ( 1 - ROI['b']))
    x4ROI, y4ROI = int(IMAGE_WIDTH * ROI['bw']), int(IMAGE_HEIGHT * ( 1 - ROI['b']))

def prespective_transform(img):
    global Minv
    src = np.array([[x1ROI, y1ROI], [x2ROI, y2ROI], [x3ROI, y3ROI], [x4ROI, y4ROI]], np.int32)
    img_ROI = region_of_interest(img, [src])
    dst = np.array([[x4ROI, y1ROI], [x3ROI, y2ROI], [x3ROI, y3ROI], [x4ROI, y4ROI]], np.float32)
    src = np.float32(src)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img_ROI, M, (IMAGE_WIDTH, IMAGE_HEIGHT), flags=cv2.INTER_LINEAR)
    return warped

def overlay_ROI_on_image(image):
    warp_zero = np.zeros_like(image).astype(np.uint8)
    pts = np.array([[x1ROI, y1ROI], [x2ROI, y2ROI], [x3ROI, y3ROI], [x4ROI, y4ROI]], np.int32)
    cv2.fillPoly(warp_zero, [pts], (0,255, 255))
    image = cv2.addWeighted(image, 1, warp_zero, 0.3, 0)
    return image

def overlay_on_image(image, left_fitx, right_fitx, yvals, invWarp=True):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image).astype(np.uint8)
    color_warp = warp_zero #np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    pts = np.squeeze(pts)
    pts = pts.astype(int)
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, [pts], (0,255, 0))

    newwarp = color_warp
    if(invWarp == True):
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

    if lines.dtype != np.int32 and not lines:
        return 0, 0, 0, 0

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
            for (x1, y1, x2, y2) in lines:
                cv2.line(img, (x1, y1), (x2, y2), color=[255, 255, 255], thickness=2)

        missing_markers = (len(x_l) == 0) + (len(x_r) == 0)
        print(str(missing_markers) + " lane marker/s not detected")
        return 0, 0, 0, 0

    #return 0 if both lane markers were detected
    return slope_l, intercept_l, slope_r, intercept_r

def calculate_curvature(yvals, xl, xr):
    xlm = [x*xm_per_pix for x in xl]
    xrm = [x*xm_per_pix for x in xr]
    ym = [y*ym_per_pix for y in yvals]

    lfit_cr = np.polyfit(ym, xlm, 2)
    rfit_cr = np.polyfit(ym, xrm, 2)
    y_eval = np.max(ym)*ym_per_pix

    off_centre_pixels = abs((IMAGE_WIDTH/2) - (xl[-1] - xr[-1]))
    off_centre_m = off_centre_pixels*ym_per_pix/100

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*lfit_cr[0]*y_eval + lfit_cr[1])**2)**1.5) / np.absolute(2*lfit_cr[0])/100
    right_curverad = ((1 + (2*rfit_cr[0]*y_eval + rfit_cr[1])**2)**1.5) / np.absolute(2*rfit_cr[0])/100

    return left_curverad, right_curverad, off_centre_m

def poly_lanes(img, orig, invWarp=True):
    copy_img = scale_img_vals(img, 255)
    nz = np.nonzero(img)
    pts_l = []
    pts_r = []
    for i in range(len(nz[0])):
        x,y = nz[1][i], nz[0][i]
        if x > img.shape[1]/2:
            pts_l.append((x, y))
        else:
            pts_r.append((x, y))

    yvals = np.linspace(y2ROI, y3ROI, num=50)

    if len(pts_l) != 0 and len(pts_r) != 0:
        pts_l = np.asarray(pts_l) 
        pts_r = np.asarray(pts_r) 
        poly_l = np.poly1d(np.polyfit(pts_l[:,1], pts_l[:,0], 2))
        poly_r = np.poly1d(np.polyfit(pts_r[:,1], pts_r[:,0], 2))
        xl, xr = [], []
        for y in yvals:
            xl.append(poly_l(y))
            xr.append(poly_r(y))

        orig_ROI = orig
        if invWarp:
            orig_ROI = overlay_ROI_on_image(orig)
        output = overlay_on_image(orig, xl, xr, yvals, invWarp)
        lrad, rrad, offc = calculate_curvature(yvals, xl, xr)
        cv2.putText(output, 'curvature of lanes %.4f m %.4f m'%(lrad, rrad), (230, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.CV_AA)
        cv2.putText(output, 'Position from centre of lane %.4f m'%(offc), (230, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.CV_AA)
        cv2.imshow("Input: Output", hstack_img(orig_ROI, output))
        cv2.moveWindow("Input: Output", 0, 0)
    else:
        print("Coult not locate lane")

    return img

def worker_func(img):
    # Color, gradient-mag, gradient-dir and other methods for generating binary image
    if PREPROCESS_EXAMPLES:
        preprocess_example_images()

    # Perspective transform to get parallel lines for lane markers 
    img = cv2.undistort(img, mtx, dist, None, mtx)
    pp_img = preprocess_image(img)  

    image_line = poly_lanes(pp_img, img)

    c = cv2.waitKey(waitTimeout)
    c = c & 0x7F
    if chr(c) == 'n':
        return 0
    elif c == 27:
        exit(0)
    elif c == 127: #Timeout
        return 0
    else:
        return 1

# Detect lane markers and save them to the folder and add to writeup

# polynomial coefficient fitting on lane markers

# Calculate radius of curvature of road in meter

# Plot result back on the road such that lane markings are correctly tracked

# Run the algo on video and store output to another video

# Discuss problems/issues. Possibility of failure. How cant it be made robust?

def write_to_config_file():
    f = open('LaneHawkConfig.py', 'w')
    f.write('import numpy as np\n')
    f.write('ROI = {\'t\':%.4f, \'b\':%.4f, \'tw\':%.4f, \'bw\':%.4f}\n'%(ROI['t'], ROI['b'],
                                                                          ROI['tw'], ROI['bw']))
    f.write('GRADX_MIN = %d\n'%(GRADX_MIN))
    f.write('GRADX_MAX = %d\n'%(GRADX_MAX))
    f.write('GRADY_MIN = %d\n'%(GRADY_MIN))
    f.write('GRADY_MAX = %d\n'%(GRADY_MAX))
    f.write('MAG_MIN   = %d\n'%(MAG_MIN))
    f.write('MAG_MAX   = %d\n'%(MAG_MAX))
    f.write('DIR_MIN   = %.2f\n'%(DIR_MIN))
    f.write('DIR_MAX   = %.2f\n'%(DIR_MAX))
    f.write('HLS_H_MIN = %d\n'%(HLS_H_MIN))
    f.write('HLS_H_MAX = %d\n'%(HLS_H_MAX))
    f.write('HLS_S_MIN = %d\n'%(HLS_S_MIN))
    f.write('HLS_S_MAX = %d\n'%(HLS_S_MAX))

def updateImage(event):
    global GRADX_MIN, GRADX_MAX, GRADY_MIN, GRADY_MAX, MAG_MIN, MAG_MAX, DIR_MIN, DIR_MAX, HLS_H_MIN, HLS_H_MAX, HLS_S_MIN, HLS_S_MAX, ROI
    GRADX_MIN = wgxm.get() #100 #w1.get()
    GRADX_MAX = wgxx.get() #255 #w2.get()
    GRADY_MIN = wgym.get() #50 #w1.get()
    GRADY_MAX = wgyx.get() #100 #w2.get()
    MAG_MIN = wmm.get() #50 #w1.get() #90
    MAG_MAX = wmx.get() #255 #w2.get() #250
    DIR_MIN = wdm.get() #np.pi/4
    DIR_MAX = wdx.get() #np.pi/2
    HLS_H_MIN = whhm.get() #50 #w1.get() #200
    HLS_H_MAX = whhx.get() #255 #w2.get() #255
    HLS_S_MIN = whsm.get() #50 #w1.get() #200
    HLS_S_MAX = whsx.get() #255 #w2.get() #255
    ROI['tw'] = rtw.get() #255 #w2.get() #255
    ROI['bw'] = rbw.get() #255 #w2.get() #255

    update_ROI_coords()
    write_to_config_file()
    img2 = preprocess_image(img)
    img_ROI = overlay_ROI_on_image(img)
    image_line = poly_lanes(img2, scale_img_vals(img2, 255), invWarp=False)
    cv2.imshow("Preprocessed i/p", hstack_img(img_ROI, img2))
    cv2.moveWindow("Preprocessed i/p", 0, 300)
    cv2.waitKey(1)
    #im = Image.fromarray(img2)
    #imgtk = ImageTk.PhotoImage(image=im) 
    #l1.configure(image=imgtk)
    #l1.image = imgtk

def setup_GUI(img):
    w,h,x,y = 1920, 600, 0, 0
    master.geometry('%dx%d+%d+%d' % (w, h, x, y))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im) 
    wgxm = Scale(master, from_=0, to=255, length=img.shape[1], orient=HORIZONTAL, command=updateImage,
                 resolution=1, borderwidth=1)
    wgxx = Scale(master, from_=0, to=255, length=img.shape[1], orient=HORIZONTAL, command=updateImage,
                 resolution=1)
    wgym = Scale(master, from_=0, to=255, length=img.shape[1], orient=HORIZONTAL, command=updateImage,
                 resolution=1)
    wgyx = Scale(master, from_=0, to=255, length=img.shape[1], orient=HORIZONTAL, command=updateImage,
                 resolution=1)
    wmm = Scale(master, from_=0, to=255, length=img.shape[1], orient=HORIZONTAL, command=updateImage,
                resolution=1)
    wmx = Scale(master, from_=0, to=255, length=img.shape[1], orient=HORIZONTAL, command=updateImage,
                resolution=1)
    wdm = Scale(master, from_=0, to=np.pi/2, length=img.shape[1], orient=HORIZONTAL, command=updateImage,
                resolution=0.1)
    wdx = Scale(master, from_=0, to=np.pi/2, length=img.shape[1], orient=HORIZONTAL, command=updateImage,
                resolution=0.1)
    whhm = Scale(master, from_=0, to=255, length=img.shape[1], orient=HORIZONTAL, command=updateImage,
                resolution=1)
    whhx = Scale(master, from_=0, to=255, length=img.shape[1], orient=HORIZONTAL, command=updateImage,
                resolution=1)
    whsm = Scale(master, from_=0, to=255, length=img.shape[1], orient=HORIZONTAL, command=updateImage,
                resolution=1)
    whsx = Scale(master, from_=0, to=255, length=img.shape[1], orient=HORIZONTAL, command=updateImage,
                resolution=1)
    rtw = Scale(master, from_=0, to=1, length=img.shape[1], orient=HORIZONTAL, command=updateImage,
                resolution=0.001)
    rbw = Scale(master, from_=0, to=1, length=img.shape[1], orient=HORIZONTAL, command=updateImage,
                resolution=0.001)
    # Put it in the display window
    l1 = Label(master, image=imgtk) 

    wgxm.pack()
    wgxx.pack()
    wgym.pack()
    wgyx.pack()
    wmm.pack()
    wmx.pack()
    wdm.pack()
    wdx.pack()
    whhm.pack()
    whhx.pack()
    whsm.pack()
    whsx.pack()
    rtw.pack()
    rbw.pack()
    #l1.pack()

    wgxm.set(GRADX_MIN)
    wgxx.set(GRADX_MAX)
    wgym.set(GRADY_MIN)
    wgyx.set(GRADY_MAX)
    wmm.set(MAG_MIN)
    wmx.set(MAG_MAX)
    wdm.set(DIR_MIN)
    wdx.set(DIR_MAX)
    whhm.set(HLS_H_MIN)
    whhx.set(HLS_H_MAX)
    whsm.set(HLS_S_MIN)
    whsx.set(HLS_S_MAX)
    rtw.set(ROI['tw'])
    rbw.set(ROI['bw'])
    return wgxm, wgxx, wgym, wgyx, wmm, wmx, wdm, wdx, whhm, whhx, whsm, whsx, rtw, rbw, l1

def close(event):
    master.withdraw() # if you want to bring it back
    sys.exit() # if you want to exit the entire thing

gen = None
img_source = "image"
source_loc = " " 
waitTimeout = 0
def init_img_source(src="image"):
    global gen, img_source, waitTimeout, source_loc
    if src == "image":
        img_source = "image"
        waitTimeout = 0
        gen = example_image_generator("")
    else:
        img_source = "video"
        source_loc = src
        waitTimeout = 1
        gen = cv2.VideoCapture(src)

frame_count = 0
def get_frame():
    global frame_count
    frame_count += 1
    if img_source == "image":
        fname, img = gen.next()
        return fname + " frame: "+str(frame_count), img
    else:
        ret, img = gen.read()
        if ret == False:
            sys.exit(0)
        return source_loc + " frame: "+str(frame_count), img 
# main
if __name__ == '__main__':
    # Calibrate camera
    mtx, dist = calibrate_camera()

    # Show undistorted images and save them
    if UNDISTORT_EXAMPLES:
        undistort_example_images(mtx, dist)

    if len(sys.argv) == 2:
        source = sys.argv[1]
        init_img_source(source)
    else:
        init_img_source()

    # TODO: Create a writeup and refer to undistorted images in it

    try:
        while(1):
            filename, img = get_frame()

            IMAGE_WIDTH = img.shape[1]
            IMAGE_HEIGHT = img.shape[0]
            print "Processing " + filename + " :: "
            ret = worker_func(img)
            if ret == 1:
                master = Tk()
                master.bind('<Escape>', close)
                wgxm, wgxx, wgym, wgyx, wmm, wmx, wdm, wdx, whhm, whhx, whsm, whsx, rtw, rbw, l1 = setup_GUI(img)
                mainloop()
    except StopIteration:
        print "Failed "
        exit(0)

