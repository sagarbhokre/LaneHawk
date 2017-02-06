# LaneHawk
An advanced lane finding using OpenCV and python. As seen in the previous implementation [SDC_LaneDetection](https://github.com/sagarbhokre/SDC_LaneDetection), the lane finding is based on edge detection and line fitting on top of the edges detected. In the advanced approach mentioned below, a histogram based lane detection is detailed which has a capability of tracking the lanes detected earlier. This approach also calculates the curvature of the lane and position of the vehicle from centre of the lane marking facilitating precise control of actuators.

### Note: Headings of sections mentioned below are mentioned in the code as comments. Hence vey less reference is given to location of corresponding section in the code

Steps involved in finding lanes markings:

 1.   Compute camera calibration matrix and distortion coefficients given a set of chessboard images
 2.   Preprocess input images
 3.   Fit a second order polynomial over lane markings for both lanes separately
 4.   Overlay the lane markings on input image
 5.   Calculate radius of curvature and position of vehicle in the lane
 6.   Overlay radius of curvature and centre of car on image
 7.   Save overlayed image into video and render on screen


## 1. Compute camera calibration matrix and distortion coefficients given a set of chessboard images
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
    
All input images for camera calibration could be found in [camera_cal folder](https://github.com/sagarbhokre/LaneHawk/tree/master/camera_cal) and all undistorted images could be found in [undistorted_camera_cal folder](https://github.com/sagarbhokre/LaneHawk/tree/master/undistorted_camera_cal)

One such output is as shown below:
![Input distorted image](https://github.com/sagarbhokre/LaneHawk/blob/master/camera_cal/calibration1.jpg "Input distorted image")
#### Input distorted image

![Undistorted image](https://github.com/sagarbhokre/LaneHawk/blob/master/undistorted_camera_cal/calibration1.jpg "Undistorted image")
#### Undistorted image


## 2. Preprocess input images
In this section various conversion and gradient calculations are performed. These thresholds were computed using slide bar in GUI (tkinter implementation of GUI) The computed thresholds are stored in LaneHawkConfig.py file. At any point in time if the thresholds appear wrong (manifested as wrong lane estimates) press any key to launch GUI. This tool proves to be very handy and useful to finetune the performance.

![GUI tool](https://github.com/sagarbhokre/LaneHawk/blob/master/debug_images/GUI_threshold_adjustments.jpg "GUI based threshold adjustment")
#### GUI based threshold adjustment


#### a. Apply distortion correction to raw images
This helps to correct out camera distortion which could be used to check if the lane markers are curved and also compute their curvature 
#### b. Convert undistorted image to RGB format
Input image captured using imread() or capture.read() APIs is in BGR format. This is converted to RGB format
 
#### c. Filter out pixels based on thresholds for gradients along x and y orientation
#### d. Filter out pixels based on imagnitude and direction gradient thresholds
#### e. Filter out yellow and white pixels with good amount of saturation
#### f. Filter out yellow and white pixels with good amount of saturation
#### g. Filter out noise using gaussian noise filter
#### h. Cut out Region Of Interest(ROI) and apply perspective transform to detect lane marker orientation
Following steps are involved in transforming the input image
1. Specify source ROI Trapeziod
2. Mask Input image with ROI trapezoid
3. Specify estination ROI rectangle
4. Compute Perspective transform matrix which would be used to get bird's eye view
5. Compute inverse perspective transform matrix to project lanes and markings on image
6. Warp input image based on perspective transform matrix just computed (this gives us bird's eye view of the ROI)

## 3. Fit a second order polynomial over lane markings for both lanes separately
#### a. Compute histogram of binary image along y axis
#### b. Find peaks in left and right halves of the histogram.
Location of these form starting points for detecting lane markers
#### c. Define width, height, confidence pixel count for the window
#### d. Step through the window starting from bottom to the top of binary image
- In this step a window boundary is computed every iteration until we reach the top of image. 
- In each window we try to find number of non-zero pixels, store and count them
- If the number of pixels in those windows is more than minpix count the window is considered to contain a lane marking and is marked as a good window for updating lane start for upcoming iteration
- The lane start for next iteration is the point where a good window is found
- This way the points are gathered throughout the execution for an image

One of the representations of lane markings captured in "DEBUG" mode is as shown below 
![Pixels considered for detecting lanes with corresponding sliding windows](https://github.com/sagarbhokre/LaneHawk/blob/master/debug_images/Debug_lane_markings.jpg "Pixels considered for detecting lanes with corresponding sliding windows")
#### Pixels considered for detecting lanes with corresponding sliding windows
    
#### e. Fit a second order polynomial over detected lane pixels
A polynomial of order 2 is fit over points for left and right lane markings separately using polyfit() API
    
## 4. Overlay the lane markings on input image
- Draw lanes onto the warped blank image
- Warp the blank back to original image space using inverse perspective matrix (Minv)
- Combine the result with the original image

## 5. Calculate radius of curvature and position of vehicle in the lane
- convert values from pixels to meter per pixel scale
- Fit the lane markings on coordinates
- Calculate the radii of curvature
- Compute centre of lane markings and centre of vehicle
- Subtract both centre points to know how off the centre of lanes are from image centre. Assuming image centre is centre of the car
- Convert the difference in centres to meters

## 6. Overlay radius of curvature and centre of car on image

## 7. Save overlayed image into video and render on screen
Output of execution is input image with ROI overlayed along with output image containing lane marking, curvature and position of car values
   
![Output windoe](https://github.com/sagarbhokre/LaneHawk/blob/master/debug_images/Output_during_execution.jpg "Output window during execution")
#### Output image rendered during execution


[![EXECUTION_PREVIEW](http://img.youtube.com/vi/R97Ifw_2rio/0.jpg)](http://www.youtube.com/watch?v=R97Ifw_2rio "Output video stored during execution")
https://www.youtube.com/watch?v=R97Ifw_2rio
#### Output video saved during execution (Uploaded on youtube)



An output video is also included in the project

[![EXECUTION_PREVIEW](http://img.youtube.com/vi/R97Ifw_2rio/0.jpg)](https://github.com/sagarbhokre/LaneHawk/blob/master/output.avi "Output video included in project")
https://github.com/sagarbhokre/LaneHawk/blob/master/output.avi
#### Output video included in the project


## Problems/issues faced

#### General threshold adjustment
While working on this implementation, adjusting the thresholds and tuning the parameters seemed to be a challenge. The x, y gradients would allow lane markings to be visible and screw up detection in other regions of the video. To tackle this problem a GUI based threshold adjustment was implemented which allowed visualization at preprocessing step while generate binary images. This reduced development and debugging effort
 
#### Frame based parameter adjustment
Another issue while implementing and debugging was to wait for the frame to be rendered and then figure out which frame failed to detect lanes properly. This was tackled by adding keyboard interface and using the latest frame being rendered to adjust the threshold parameters. In the python application whenever a lane marking is being detected properly, press any key (except 'ESC') and the GUI would be opened to adjust preprocessing parameters

#### Skipping frames to reach frame of interest
After changing threshold parameters, it was very time consuming to wait for the frame to arrive and then debug. Repeating this process until the parameters were fine tuned would have been tedious. To overcome this situation 'frame_skip' logic was implemented. This avoids processing any frames until the skip count elapses (See variable 'SKIP_COUNT' line 16 in LaneHawk_main.py)

## Improvements in algorithm/pipeline
1. To further improve this algorithm, a logic could be implemented which keeps a track of the place in image where a lane marking was detected last. Next iteration would start from the last known location of lane marking. This would reduce the histogram computation step to estimate the starting point for lane marking
2. Another improvement could be to detect one lane marking (say left lane) and use that as an estimate to detect the other lane marking (right lane)

### Failure scenarios (hypothetical cases)
1. This algorithm could fail to work during bad weather conditions (e.g. fog, rain)
2. Dusty roads would also significantly affect the performance of this algorithm
3. During bright sunny days on roads with lighter surface, the algorithm could fail to detect lane markings
4. Occlusion or presence another car in front could raise serious concerns
5. The implementation could also fail for very sharp turns where only one lane marker is visible
