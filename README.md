# LaneHawk
An advanced lane finding using OpenCV and python. As seen in the previous implementation [SDC_LaneDetection](https://github.com/sagarbhokre/SDC_LaneDetection), the lane finding is based on edge detection and line fitting on top of the edges detected. In the advanced approach mentioned below, a histogram based lane detection is detailed which has a capability of tracking the lanes detected earlier. This approach also calculates the curvature of the lane and position of the vehicle from centre of the lane marking facilitating precise control of actuators.

Steps involved in finding lanes

-   Compute camera calibration matrix and distortion coefficients given a set of chessboard images
-   Apply a distortion correction to raw images
-   Use color transforms, gradients, etc., to create a thresholded binary image
-   Apply a perspective transform to rectify binary image ("birds-eye view")
-   Detect lane pixels and fit to find the lane boundary
-   Determine the curvature of the lane and vehicle position with respect to center
-   Warp the detected lane boundaries back onto the original image
-   Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position



## Compute camera calibration matrix and distortion coefficients given a set of chessboard images
The task is accomplished by function "calibrate_camera()" in the code (file: [LaneHawk_main.py](https://github.com/sagarbhokre/LaneHawk/blob/master/LaneHawk_main.py))

Following are the sub-steps involved
1. Read calibration images (images inside [camera_cal](https://github.com/sagarbhokre/LaneHawk/tree/master/camera_cal) folder 
2. Convert to grayscale
3. Generate object points: These are the points on a grid. These would be used when projecting chessboard points on undistorted images
4. Detect chessboard corners: This is done using OpenCV API cv2.findChessboardCorners(). cv2.drawChessboardCorners() can be used to verify if the chessboard points are detected correctly. This call is currently controlled by a global variable "DEBUG".
5. Store object points and image points (detected chessboard corners)
6. Calibrate camera: This is done using API cv2.calibrateCamera()
7. Verify calibration parameters by distorting input images
8. Store calibration parameters in a pickle file: Calibration parameters of a camera remain the same hence this computation can be avoided everytime we execute the code. Hence the parameters once computed are stored in file "cam_calib_params.pkl". If this file already exists, the parameters are not computed again but loaded from the pkl file.
    
All input images for camera calibration could be found in [camera_cal folder](https://github.com/sagarbhokre/LaneHawk/tree/master/camera_cal) and all undistorted images could be found in [undistorted_camera_cal folder]((https://github.com/sagarbhokre/LaneHawk/tree/master/undistorted_camera_cal)

One such output is as shown below:
![Input distorted image](https://github.com/sagarbhokre/LaneHawk/blob/master/camera_cal/calibration1.jpg "Input distorted image") ![Undistorted image](https://github.com/sagarbhokre/LaneHawk/blob/master/undistorted_camera_cal/calibration1.jpg "Undistorted image")


## Preprocess input images

### Apply distortion correction to raw images
This helps to correct out camera distortion which could be used to check if the lane markers are curved and also compute their curvature 
### Convert undistorted image to RGB format
Input image captured using imread() or capture.read() APIs is in BGR format. This is converted to RGB format
 
### Filter out pixels based on thresholds for gradients along x and y orientation
### Filter out pixels based on imagnitude and direction gradient thresholds
### Filter out yellow and white pixels with good amount of saturation
### Filter out yellow and white pixels with good amount of saturation
### Filter out noise using gaussian noise filter
### Cut out Region Of Interest(ROI) and apply perspective transform to detect lane marker orientation
Following steps are involved in transforming the input image
1. Specify source ROI Trapeziod
2. Mask Input image with ROI trapezoid
3. Specify estination ROI rectangle
4. Compute Perspective transform matrix which would be used to get bird's eye view
5. Compute inverse perspective transform matrix to project lanes and markings on image
6. Warp input image based on perspective transform matrix just computed (this gives us bird's eye view of the ROI)

## Fit a second order polynomial over lane markings for both lanes separately
### Compute histogram of binary image along y axis
### Find peaks in left and right halves of the histogram.
Location of these form starting points for detecting lane markers
### Define width, height, confidence pixel count for the window
### Step through the window starting from bottom to the top of binary image
- In this step a window boundary is computed every iteration until we reach the top of image. 
- In each window we try to find number of non-zero pixels, store and count them
- If the number of pixels in those windows is more than minpix count the window is considered to contain a lane marking and is marked as a good window for updating lane start for upcoming iteration
- The lane start for next iteration is the point where a good window is found
- This way the points are gathered throughout the execution for an image

One of the representations of lane markings captured in "DEBUG" mode is as shown below 
![Pixels considered for detecting lanes with corresponding sliding windows](https://github.com/sagarbhokre/LaneHawk/blob/master/debug_images/Debug_lane_markings.jpg "Pixels considered for detecting lanes with corresponding sliding windows")
    
### Fit a second order polynomial over detected lane pixels
A polynomial of order 2 is fit over points for left and right lane markings separately using polyfit() API
    
## Overlay the lane markings on input image
- Draw lanes onto the warped blank image
- Warp the blank back to original image space using inverse perspective matrix (Minv)
- Combine the result with the original image

## Calculate radius of curvature and position of vehicle in the lane
- convert values from pixels to meter per pixel scale
- Fit the lane markings on coordinates
- Calculate the radii of curvature
- Compute centre of lane markings and centre of vehicle
- Subtract both centre points to know how off the centre of lanes are from image centre. Assuming image centre is centre of the car
- Convert the difference in centres to meters

## Overlay radius of curvature and centre of car on image

## Save overlayed image into video and render on screen

