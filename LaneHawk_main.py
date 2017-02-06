import cv2, os, sys, copy, glob
import numpy as np
import cPickle
from Tkinter import *
import Image, ImageTk
import matplotlib.pyplot as plt

ym_per_pix = 100 * 30/720 # meters per pixel in y dimension
xm_per_pix = 100 * 3.7/700 # meters per pixel in x dimension

CHESSBOARD_WIDTH  = 9
CHESSBOARD_HEIGHT = 6
DEBUG = 0
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
SKIP_COUNT = 0

def calibrate_camera():
    imgpoints, objpoints = [], []
    resfile = 'cam_calib_params.pkl'

    if(os.path.isfile("./" + resfile)):
        print ("File exists :" + "./" + resfile)
        print ("Loading already saved calib params")
        data = cPickle.load(open(resfile, "rb"))
        return data

    # Read calibration images
    for filename in glob.glob("camera_cal/*"):
        img = cv2.imread(filename)
        if DEBUG == 1:
            cv2.imshow("Calibration images", img)
            cv2.waitKey(100)

        # Compute calibration coefficients
        # Convert to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        print(gray.shape[::-1])
 
        # Generate object points
        objp = np.zeros((CHESSBOARD_WIDTH*CHESSBOARD_HEIGHT, 3), np.float32)
        objp[:,:2] = np.mgrid[0:CHESSBOARD_WIDTH, 0:CHESSBOARD_HEIGHT].T.reshape(-1, 2)
    
        # Detect chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT),None)

        # Store object points and image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
        else:
            print ("Failed to detect chessboard corners for file: " + filename)

        if DEBUG == 1:
            img_cb = cv2.drawChessboardCorners(img, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), corners, ret)
            cv2.imshow("Chessboard corners detected", img_cb)
            cv2.waitKey(0)
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (IMAGE_WIDTH, IMAGE_HEIGHT),None,None)

    # Verify calibration parameters by distorting input images
    for filename in glob.glob("camera_cal/*"):
        img = cv2.imread(filename)

        filename = filename.split("/")
        #  Undistort input image
        dst = cv2.undistort(img, mtx, dist, None, mtx)

        #  Show undistorted image
        cv2.imshow("Undistorted image", dst)

        cv2.imwrite("./undistorted_camera_cal/"+filename[1], dst)
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
    # Undistort input image
    img = cv2.undistort(img, mtx, dist, None, mtx)

    # Convert to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Filter out pixels based on thresholds for gradients along x and y orientation
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(GRADX_MIN, GRADX_MAX))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(GRADY_MIN, GRADY_MAX))

    # Filter out pixels based on imagnitude and direction gradient thresholds
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(MAG_MIN, MAG_MAX))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(DIR_MIN, DIR_MAX))

    # Filter out yellow and white pixels with good amount of saturation
    hls_binary = hls_select(img, (HLS_H_MIN, HLS_H_MAX), (HLS_S_MIN, HLS_S_MAX))

    # Combine output from various filters above
    combined = np.zeros_like(img[:,:,0])
    combined[(hls_binary == 1) | (((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)))] = 1

    # Filter out noise using gaussian noise filter
    combined = gaussian_blur(combined, GAUSS_KER)

    # Cut out Region of interest and apply perspective transform to detect lane marker orientation
    ROI_warped, ROI_warped_full = prespective_transform(combined)

    return ROI_warped, ROI_warped_full

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
    global Minv, Minv_full

    # Source ROI Trapeziod
    src = np.array([[x1ROI, y1ROI], [x2ROI, y2ROI], [x3ROI, y3ROI], [x4ROI, y4ROI]], np.int32)

    # Input image masked with ROI trapezoid 
    img_ROI = region_of_interest(img, [src])

    # Destination ROI rectangle    
    dst = np.array([[x4ROI, y1ROI], [x3ROI, y2ROI], [x3ROI, y3ROI], [x4ROI, y4ROI]], np.float32)
    dst_full = np.array([[0,0], [IMAGE_WIDTH, 0], [IMAGE_WIDTH, IMAGE_HEIGHT], [0,IMAGE_HEIGHT]], np.float32)

    src = np.float32(src)
    # Perspective transform to get bird's eye view
    M = cv2.getPerspectiveTransform(src, dst)
    Mfull = cv2.getPerspectiveTransform(src, dst_full)

    # compute inverse perspective transform matrix to project lanes and markings on image
    Minv = cv2.getPerspectiveTransform(dst, src)
    Minv_full = cv2.getPerspectiveTransform(dst_full, src)

    # Warp input image based on perspective transform matrix just computed
    warped = cv2.warpPerspective(img_ROI, M, (IMAGE_WIDTH, IMAGE_HEIGHT), flags=cv2.INTER_LINEAR)
    warped_full = cv2.warpPerspective(img_ROI, Mfull, (IMAGE_WIDTH, IMAGE_HEIGHT), flags=cv2.INTER_LINEAR)

    return warped, warped_full

def overlay_ROI_on_image(image):
    warp_zero = np.zeros_like(image).astype(np.uint8)
    pts = np.array([[x1ROI, y1ROI], [x2ROI, y2ROI], [x3ROI, y3ROI], [x4ROI, y4ROI]], np.int32)
    cv2.fillPoly(warp_zero, [pts], (0,255, 255))
    image = cv2.addWeighted(image, 1, warp_zero, 0.3, 0)
    return image

def overlay_on_image(image, left_fitx, right_fitx, yvals, invWarp=True):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image).astype(np.uint8)
    color_warp = warp_zero

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    pts = np.squeeze(pts)
    pts = pts.astype(int)

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, [pts], (0,255, 0))

    newwarp = color_warp

    invMtrx = Minv
    if ALGORITHM == 2:
        invMtrx = Minv_full

    if(invWarp == True):
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, invMtrx, (image.shape[1], image.shape[0]))
 
    # Combine the result with the original image
    image = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return image

# Noise filter
GAUSS_KER = 3

#Lane detection slope threshold
SLOPE_THRESHOLD = 0.4
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def calculate_curvature(yvals, xl, xr):
    # convert values from pixels to meter per pixel scale
    xlm = [x*xm_per_pix for x in xl]
    xrm = [x*xm_per_pix for x in xr]
    ym = [y*ym_per_pix for y in yvals]

    # Fit the lane markings on coordinates
    lfit_cr = np.polyfit(ym, xlm, 2)
    rfit_cr = np.polyfit(ym, xrm, 2)
    y_eval = np.max(ym)*ym_per_pix

    # Calculate radii of curvature
    left_curverad = ((1 + (2*lfit_cr[0]*y_eval + lfit_cr[1])**2)**1.5) / np.absolute(2*lfit_cr[0])/100
    right_curverad = ((1 + (2*rfit_cr[0]*y_eval + rfit_cr[1])**2)**1.5) / np.absolute(2*rfit_cr[0])/100

    # Compute centre of lane markings and centre of vehicle
    off_centre_pixels = ((IMAGE_WIDTH/2) - (xl[-1] + xr[-1])/2)
    off_centre_m = off_centre_pixels*ym_per_pix/100

    return left_curverad, right_curverad, off_centre_m

def fit_lanes_algo2(orig, binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)

    out_img = []
    if DEBUG == 1:
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if DEBUG == 1:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for lane marking
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    if DEBUG == 1:
        lpts = np.hstack((np.int32(np.reshape(left_fitx, (len(left_fitx), 1))),
                          np.int32(np.reshape(ploty, (len(ploty), 1)))))
        rpts = np.hstack((np.int32(np.reshape(right_fitx, (len(right_fitx), 1))),
                          np.int32(np.reshape(ploty, (len(ploty), 1)))))

        # Mark lane pixels detected within sliding windows
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Mark detected lanes
        cv2.polylines(out_img, [lpts], False, (255, 255, 255), thickness=3)
        cv2.polylines(out_img, [rpts], False, (255, 255, 255), thickness=3)

        # Show lane markings along with respective sliding windows
        cv2.imshow("Debug Lane markings", out_img)
        cv2.imwrite("./debug_images/Debug_lane_markings.jpg", out_img)

    return ploty, left_fitx, right_fitx

def fit_lanes_algo1(orig, img):
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
        return yvals, xl, xr
    else:
        print("Failed to detect lane markers")
        exit(0)
        return yvals, [], []

def poly_lanes(img, img_full, orig, invWarp=True):
    orig_ROI = orig
    if invWarp:
        orig_ROI = overlay_ROI_on_image(orig)

    if ALGORITHM == 1:
        ys, xls, xrs = fit_lanes_algo1(orig, img)
    elif ALGORITHM == 2:
        ys, xls, xrs = fit_lanes_algo2(orig, img_full)

    output = overlay_on_image(orig, xls, xrs, ys)

    lrad, rrad, offc = calculate_curvature(ys, xls, xrs)
    cv2.putText(output, 'curvature of lanes %.4f m %.4f m'%(lrad, rrad), (230, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.CV_AA)
    cv2.putText(output, 'Position from centre of lane %.4f m'%(offc), (230, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.CV_AA)
    cv2.imshow("Input: Output", hstack_img(orig_ROI, output))
    cv2.moveWindow("Input: Output", 0, 0)
    return output

def worker_func(img):
    # Color, gradient-mag, gradient-dir and other methods for generating binary image
    pp_img, pp_img_full = preprocess_image(img)  

    # Fit a second order polynomial over lane markings for both lanes separately
    image_line = poly_lanes(pp_img, pp_img_full, img)

    # Write processed frame to output video
    if img_source == "video":
        out.write(image_line)

    # Check if the user wants to change parameters or exit
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
    f.write('ALGORITHM = %d\n'%(ALGORITHM))
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
    GRADX_MIN = wgxm.get()
    GRADX_MAX = wgxx.get()
    GRADY_MIN = wgym.get()
    GRADY_MAX = wgyx.get()
    MAG_MIN = wmm.get()
    MAG_MAX = wmx.get()
    DIR_MIN = wdm.get()
    DIR_MAX = wdx.get()
    HLS_H_MIN = whhm.get()
    HLS_H_MAX = whhx.get()
    HLS_S_MIN = whsm.get()
    HLS_S_MAX = whsx.get()
    ROI['tw'] = rtw.get()
    ROI['bw'] = rbw.get()

    update_ROI_coords()
    write_to_config_file()
    img2, img2_full = preprocess_image(img)
    img_ROI = overlay_ROI_on_image(img)
    image_line = poly_lanes(img2, img2_full, scale_img_vals(img2, 255), invWarp=False)
    cv2.imshow("Preprocessed i/p", hstack_img(img_ROI, img2))
    cv2.moveWindow("Preprocessed i/p", 0, 300)
    cv2.waitKey(1)

def setup_GUI(img):
    w,h,x,y = 1920, 600, 0, 0
    master.geometry('%dx%d+%d+%d' % (w, h, x, y))

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

    return wgxm, wgxx, wgym, wgyx, wmm, wmx, wdm, wdx, whhm, whhx, whsm, whsx, rtw, rbw

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

    out = None
    fourcc = cv2.cv.CV_FOURCC(*'XVID')

    # Show undistorted images and save them
    if len(sys.argv) == 2: # Handle video frames
        source = sys.argv[1]
        init_img_source(source)
    else:                  # Handle images
        init_img_source()

    # TODO: Create a writeup and refer to undistorted images in it
    try:
        while(1):
            filename, img = get_frame()
            if frame_count > SKIP_COUNT:
                IMAGE_WIDTH = img.shape[1]
                IMAGE_HEIGHT = img.shape[0]
                if out is None:
                    out = cv2.VideoWriter('output.avi',fourcc, 25.0, (IMAGE_WIDTH,IMAGE_HEIGHT))
                print "Processing " + filename + " :: "
                ret = worker_func(img)
                if ret == 1:
                    master = Tk()
                    master.bind('<Escape>', close)
                    wgxm, wgxx, wgym, wgyx, wmm, wmx, wdm, wdx, whhm, whhx, whsm, whsx, rtw, rbw = setup_GUI(img)
                    mainloop()
    except StopIteration:
        print "Failed "
        exit(0)
