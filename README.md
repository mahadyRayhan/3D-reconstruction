##  3D reconstruction

### What is 3D reconstruction?
> In computer vision and computer graphics, 3D reconstruction is the process of capturing the shape and appearance of real objects.

### Whe it is needed?
The task of generating fast and accurate 3D image reconstruction has found its application in the field of computer vision like 
- 	robotics/industrial automation (if you like to move objects using a robotics arm)
- 	entertainment
- 	reverse engineering
- 	augmented reality
- 	human computer interaction and 
- 	animation

In robotics/industrial automation 6D pose estimation is a very important task nowadays. According to some recent papers like **"FFB6D: A Full Flow Bidirectional Fusion Network for 6D Pose Estimation"** or **"FS-Net: Fast Shape-based Network for Category-Level 6D Object Pose Estimation with Decoupled Rotation Mechanism"** depth maps are used as inputs for these papers, which improves the model performance.

In this article, you will learn how can we generate depth maps from stereo images, and later you can use this depth map as an input to these papers.

<hr>

To recreate our surrounding world in 3D from 2D images, the main thing we need is an actual depth map.

A depth map is a picture where every pixel has depth information (instead of color information).

It is normally represented like a grayscale picture where lighter shades of gray will signify objects close to the camera lens with progressively darker shades to distinguish objects further away.

The main steps needed to recreate your surrounding world in 3D are:

- Camera calibration
- Undistort image
- Feature matching
- Build point cloud
- Build mesh to get an actual 3D model

## Camera calibration

Camera calibration is the first, most important, easy task for 3D reconstruction.

I said easily because OpenCV provides a handful of functions that you can use to do that. You don't have to calculate the rotation and translation vectors by urself. OpenCV will do that for you.

### Different types of camera calibration

There are 4 major type of camera calibration process.
- Calibration pattern based
- Deep Learning based
- Geometric clue based

Among these methods **"Calibration pattern-based"** (checkerboard-based method, which I will use) method is the most popular. But to use this method you have to have complete control over the imaging process.

If you do not have control over your imaging process and also have a very small number of images (like only one image), you can use this **Deep Learning** method. Deep Learning is a very powerful tool to obtain certain information from an image.

Also, you may have some geometric clues like straight lines and vanishing points which can be used for calibrating the camera. This method is known as the **Geometric clue-based** method.

**Why checkerboard based method??**

- Checkerboard patterns are distinct and easy to detect in an image. 
- The corners of squares on the checkerboard are ideal for localizing them because they have sharp gradients in two directions
- Also, these corners are related by the fact that they are at the intersection of checkerboard lines.

#### Accure Images

The first step of camera calibration is to acquire images. As I am using the checkerboard-based method you can download a checkerboard image online, print it, and put it on a white wall. Then capture as many pictures as you need to calibrate the camera. 

Also, you can download the checkerboard images for camera calibration as I do. I use 41 images to calibrate the camera. You can use more.

#### Import librarys


```python
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
```

First, define the dimensions of the checkerboard


```python
CHECKERBOARD = (6,9)
```

If you take the images by yourself, you make take a picture of a portion of a checkerboard that is smaller or larger than my input images. Adjust this **CHECKERBOARD** size accordingly.

Now, define two array named **objpoints** and **imgpoints**

- Imgpoints **--** 2d points in image plane and
- Objpoints  **--** 3d point in real world space


```python
# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []
# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
```

#### Corner detection

Next, we will read all the images we take/downloaded for camera calibration and pass them to **findChessboardCorners()** function from OpenCV. This function will find all the corner points for us along with a boolean flag to indicate whether the function finds the corner points or not. The function takes grayscale images only, so you have to convert the input images to grayscale.

To make corner detection more robust we will use a sub-pixel level of accuracy. Again, for this sub-pixel level of accuracy calculation, you do not have to do anything new. OpenCV already have a function called **cornerSubPix()**. Achieving 100% accuracy in the sub-pixel level is difficult. So, you have to define a **criteria** that how long the function will look for the optimal solution. After meeting your desire criteria, the function will stop the iterative process of corner refinement.


```python
# ( type, max_iter, epsilon )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
```

There are option for criteria to stop the iteration.
- End condition
- max_iter
- epsilon/Accuracy

- End condition type: Has the following three flags:

- cv2.TERM_CRITERIA_EPS **--** Ends the iterative calculation when the specified accuracy ( epsilon ) is reached.
- cv2.TERM_CRITERIA_MAX_ITER **--** The iterative calculation ends when the specified number of repetitions ( max_iter ) is reached.
- cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER **--** The iterative calculation ends when either of the above conditions is met.

- max_iter **--** An integer value for specifying the maximum value for iterative calculation.

- epsilon **--** Required accuracy.


```python
# Extracting path of individual image stored in a given directory
images = glob.glob('./images/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray,
                                             (6,9),
                                             cv2.CALIB_CB_ADAPTIVE_THRESH+
                                             cv2.CALIB_CB_FAST_CHECK+
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
        plt.imshow(img)
        plt.show()
```


![png](./doc_image/output_31_0.png)



![png](./doc_image/output_31_1.png)



<!---![png](./doc_image/output_31_2.png)-->



<!---![png](./doc_image/output_31_3.png)-->



<!---![png](./doc_image/output_31_4.png)-->



<!---![png](./doc_image/output_31_5.png)-->



<!---![png](./doc_image/output_31_6.png)-->



<!---![png](./doc_image/output_31_7.png)-->



<!---![png](./doc_image/output_31_8.png)-->



<!---![png](./doc_image/output_31_9.png)-->



<!---![png](./doc_image/output_31_10.png)-->



<!---![png](./doc_image/output_31_11.png)-->



<!---![png](./doc_image/output_31_12.png)-->



<!---![png](./doc_image/output_31_13.png)-->



<!---![png](./doc_image/output_31_14.png)-->



![png](./doc_image/output_31_15.png)-->



<!---![png](./doc_image/output_31_16.png)-->



<!---![png](./doc_image/output_31_17.png)-->



<!---![png](./doc_image/output_31_18.png)-->



<!---![png](./doc_image/output_31_19.png)-->



<!---![png](./doc_image/output_31_20.png)-->



<!---![png](./doc_image/output_31_21.png)-->



<!---![png](./doc_image/output_31_22.png)-->



<!---![png](./doc_image/output_31_23.png)-->



<!---![png](./doc_image/output_31_24.png)-->



<!---![png](./doc_image/output_31_25.png)-->



<!---![png](./doc_image/output_31_26.png)-->



<!---![png](./doc_image/output_31_27.png)-->



<!---![png](./doc_image/output_31_28.png)-->



<!---![png](./doc_image/output_31_29.png)-->



<!---![png](./doc_image/output_31_30.png)-->



<!---![png](./doc_image/output_31_31.png)-->


#### Calibrate Camera

Finally, we will pass the 3D points in world coordinates and their 2D locations in all images to OpenCV **calibrateCamera()** method.


```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
```

This algorithm will output the camera parameters. The algorithm returns the camera matrix (mtx) distortion coefficients (dist) and the rotation and translation vectors (rvecs and tvecs).


```python
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)
```

    Camera matrix : 
    
    [[503.68477278   0.         313.67563674]
     [  0.         503.37989194 243.25575476]
     [  0.           0.           1.        ]]
    dist : 
    
    [[ 2.08346324e-01 -4.68650266e-01  4.51079181e-04 -1.93373893e-03
       2.37592401e-01]]
    rvecs : 
    
    [array([[-0.31060359],
           [ 0.1515558 ],
           [ 1.57720394]]), array([[0.3139181 ],
           [0.58736866],
           [1.39910762]]), array([[0.33307297],
           [0.55841704],
           [1.39532836]]), array([[0.78743976],
           [0.53294469],
           [1.52233962]]), array([[0.42519317],
           [0.68127945],
           [1.34442524]]), array([[-0.40676505],
           [-0.72194813],
           [ 1.36334531]]), array([[-0.41607396],
           [-0.7647259 ],
           [ 1.33850247]]), array([[-0.23343341],
           [-0.04621596],
           [ 1.522275  ]]), array([[ 0.11068043],
           [-0.46502899],
           [ 1.48272651]]), array([[0.84715318],
           [0.19610195],
           [1.46485457]]), array([[0.28284572],
           [0.58780705],
           [1.3194868 ]]), array([[0.15172276],
           [0.57093073],
           [1.41375183]]), array([[-0.12600935],
           [ 0.26517559],
           [ 1.52229255]]), array([[-0.1929761 ],
           [ 0.37179966],
           [ 1.55313095]]), array([[-0.33899421],
           [-0.47377128],
           [ 1.59483249]]), array([[-0.15838481],
           [ 0.05212478],
           [ 1.52017069]]), array([[-0.15347862],
           [ 0.39951894],
           [ 1.49239868]]), array([[-0.2151081 ],
           [ 0.31910018],
           [ 1.53296626]]), array([[-0.21575755],
           [ 0.38055436],
           [ 1.50528033]]), array([[1.51596833e-03],
           [6.69656748e-02],
           [1.56241940e+00]]), array([[-1.46827972e-03],
           [ 4.76768779e-02],
           [ 1.56724499e+00]]), array([[-9.09292387e-04],
           [ 3.66898485e-02],
           [ 1.56666819e+00]]), array([[-0.44555026],
           [-0.30730569],
           [ 1.53807621]]), array([[-0.01540714],
           [ 0.08681116],
           [ 1.54169614]]), array([[0.45057599],
           [0.54006295],
           [1.38860638]]), array([[-0.69506538],
           [-0.57631163],
           [ 1.45019819]]), array([[-0.55132416],
           [-0.99705698],
           [ 1.23526952]]), array([[1.02215382],
           [0.41241127],
           [1.47957469]]), array([[0.61951873],
           [0.08486798],
           [1.49114099]]), array([[-0.40374551],
           [-0.06016137],
           [ 1.53168687]]), array([[-0.29895082],
           [ 0.18643046],
           [ 1.57442937]]), array([[-0.23749973],
           [ 0.22031738],
           [ 1.55996673]])]
    tvecs : 
    
    [array([[ 5.48323978],
           [-1.00919763],
           [12.77906849]]), array([[ 2.56380074],
           [-2.03560398],
           [ 9.88163046]]), array([[ 2.78198101],
           [-2.71047535],
           [ 9.46752665]]), array([[ 1.53264336],
           [-2.40441741],
           [ 8.16064041]]), array([[ 0.68602451],
           [-2.48343549],
           [10.55202064]]), array([[ 3.45373764],
           [-3.32563976],
           [12.92869075]]), array([[ 2.927001  ],
           [-3.8124096 ],
           [13.08614645]]), array([[ 5.73455513],
           [-2.70716913],
           [14.02946082]]), array([[ 4.82906644],
           [-2.55201216],
           [ 9.98282155]]), array([[ 1.0790422 ],
           [-2.10327764],
           [ 7.64270833]]), array([[ 1.91174366],
           [-3.16265821],
           [ 8.87645259]]), array([[ 1.67727174],
           [-2.44980132],
           [ 9.10099492]]), array([[ 3.4677332 ],
           [-2.25213533],
           [ 9.84036389]]), array([[ 4.5987613 ],
           [-2.88581944],
           [11.2694964 ]]), array([[ 5.09427351],
           [-2.62523423],
           [14.86684381]]), array([[ 3.00290714],
           [-2.37996175],
           [19.24801655]]), array([[ 3.49959139],
           [-2.02204682],
           [14.0114402 ]]), array([[ 3.1533137 ],
           [-2.52597187],
           [11.1719889 ]]), array([[ 3.58609878],
           [-2.24346754],
           [10.52411658]]), array([[ 4.19975796],
           [-2.45106708],
           [ 9.34149497]]), array([[ 3.9567282 ],
           [-2.67422839],
           [ 8.57511753]]), array([[ 3.74275493],
           [-2.91015673],
           [ 8.21848023]]), array([[ 5.43496492],
           [-2.41530305],
           [14.43833827]]), array([[ 3.71941416],
           [-2.58185513],
           [ 8.4887555 ]]), array([[ 3.05920063],
           [-2.48136724],
           [ 9.46742967]]), array([[ 2.15837973],
           [-2.69161395],
           [14.54502612]]), array([[ 0.94099621],
           [-4.46958494],
           [14.14987195]]), array([[ 2.16201438],
           [-2.05542549],
           [ 7.39413122]]), array([[ 2.84015628],
           [-2.06069187],
           [ 7.18834522]]), array([[ 2.21391313],
           [-2.32755227],
           [14.87011037]]), array([[ 5.11289273],
           [-0.85351624],
           [13.36840857]]), array([[ 2.75337631],
           [-1.26097867],
           [12.69840455]])]
    

You can save these camera parameters for further use.


```python
#Save parameters into numpy file
np.save("./camera_params/ret", ret)
np.save("./camera_params/mtx", mtx)
np.save("./camera_params/dist", dist)
np.save("./camera_params/rvecs", rvecs)
np.save("./camera_params/tvecs", tvecs)
```

You can also calculate the projection errors as follows.


```python
#Calculate projection error. 
mean_error = 0
for i in range(len(objpoints)):
	img_points2, _ = cv2.projectPoints(objpoints[i],rvecs[i],tvecs[i], mtx, dist)
	error = cv2.norm(imgpoints[i], img_points2, cv2.NORM_L2)/len(img_points2)
	mean_error += error

total_error = mean_error/len(objpoints)
print (total_error)
```

    0.03108792528908075
    

### Stereo 3D reconstruction 

Now our camera is calibrated with a low error. Next, we will try to reconstruct a 3D image from 2 2D images.


```python
#Load pictures
img_1 = cv2.imread('./reconstruct_this/st_left.png')
img_2 = cv2.imread('./reconstruct_this/st_right.png')
```

**For this reconstruction process both images shape must be same. For me both image is already in the same size.**


```python
img_1.shape, img_2.shape
```




    ((480, 640, 3), (480, 640, 3))



If your input images are not same sized please resize them to same size.


```python
h,w = img_2.shape[:2]
```

#### Undistortion

Next, we have to undistortion the input images. For better undistortion we can use OpenCV's **getOptimalNewCameraMatrix()**. This function will use our existing camera matrix, distortion coefficients and refine them for new images.


```python
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
```


```python
#Undistort images
img_1_undistorted = cv2.undistort(img_1, mtx, dist, None, new_camera_matrix)
img_2_undistorted = cv2.undistort(img_2, mtx, dist, None, new_camera_matrix)
```

#### Compute disparity  map/depth map

Next will compute the disparity map for the undistorted images. First I set disparity parameters. These parameters are arbitrarily chosen. You can improve the final output by tuning these parameters.


```python
#Set disparity parameters 
win_size = 5
min_disp = -1
max_disp = 63 #min_disp * 9
num_disp = max_disp - min_disp # Needs to be divisible by 16
```

Then we create a Semi-Global Block Matching algorithm from OpenCV called **"StereoSGBM_create()"**. This function will generate a depth map for our input images.


```python
#Create Block matching object. 
stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
	numDisparities = num_disp,
	blockSize = 5,
	uniquenessRatio = 5,
	speckleWindowSize = 5,
	speckleRange = 5,
	disp12MaxDiff = 2,
	P1 = 8*3*win_size**2,
	P2 =32*3*win_size**2)

#Compute disparity map
print ("\nComputing the disparity  map...")
disparity_map = stereo.compute(img_1_undistorted, img_2_undistorted)

#Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
plt.imshow(disparity_map,'gray')
plt.show()
```

    
    Computing the disparity  map...
    


![png](./doc_image/output_56_1.png)


In the beginning, we said that lighter shades of gray will signify objects close to the camera lens with progressively darker shades to distinguish objects further away.

We can plot all three images side by side (two input images and one generated gray-scale depth map image) and verify our statement.

#### Compare output


```python
fig = plt.figure(figsize=(10, 7))

# showing left image
fig.add_subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_RGB2BGR))
plt.axis('off')
plt.title("Left")

# showing depth mapimage
fig.add_subplot(1, 3, 2)
plt.imshow(disparity_map, cmap='gray')
plt.axis('off')
plt.title("Depth Map")

# showing right image
fig.add_subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR))
plt.axis('off')
plt.title("Right")
plt.show()
```


![png](./doc_image/output_59_0.png)


The output is not perfect but we can see that the objects which are comparatively close to the camera are marked as lite-gray (white) and objects that are far from the camera are marked as dark-gray (black).


```python
def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')

```

### Visulize output

We can visualize the output by generating a point cloud. Meshlab is a great tool to visualize the reconstructed images.

#### Build point cloud

To create a point cloud, we need the focal length of the camera we use to take the images. Focal length can be described from Exif data contained in the picture.

<pre>
exif_img = PIL.Image.open('./reconstruct_this/st_left.png')
exif_data = {
 PIL.ExifTags.TAGS[k]:v
 for k, v in exif_img._getexif().items()
 if k in PIL.ExifTags.TAGS}
#Get focal length in tuple form
focal_length_exif = exif_data['FocalLength']
#Get focal length in decimal form
focal_length = focal_length_exif[0]/focal_length_exif[1]
</pre>


```python
#Generate  point cloud. 
print ("\nGenerating the 3D map...")
#Get width and height 
h,w = img_2_undistorted.shape[:2]
#Load focal length. 
focal_length = np.load('./camera_params/FocalLength.npy')
#Perspective transformation matrix
#This transformation matrix is from the openCV documentation, didn't seem to work for me. 
Q = np.float32([[1,0,0,-w/2.0],
    [0,-1,0,h/2.0],
    [0,0,0,-focal_length],
    [0,0,1,0]])
#This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision. 
#Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
Q2 = np.float32([[1,0,0,0],
    [0,-1,0,0],
    [0,0,focal_length*0.05,0], #Focal length multiplication obtained experimentally. 
    [0,0,0,1]])
#Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)
#Get color points
colors = cv2.cvtColor(img_1_undistorted, cv2.COLOR_BGR2RGB)
#Get rid of points with value 0 (i.e no depth)
mask_map = disparity_map > disparity_map.min()
#Mask colors and points. 
output_points = points_3D[mask_map]
output_colors = colors[mask_map]
#Define name for output file
output_file = 'output/reconstructed.ply'
#Generate point cloud 
print ("\nCreating the output file... \n")
create_output(output_points, output_colors, output_file)
```

    
    Generating the 3D map...
    
    Creating the output file... 
    
    


```python
# create figure
fig = plt.figure(figsize=(20, 15))
  
# setting values to rows and column variables
rows = 2
columns = 2

fig.add_subplot(rows, columns, 1)
plt.imshow(cv2.cvtColor(cv2.imread('./output/front.png'), cv2.COLOR_RGB2BGR))
plt.axis('off')
plt.title("Front view")
  
fig.add_subplot(rows, columns, 2)
plt.imshow(cv2.cvtColor(cv2.imread('./output/front-left.png'), cv2.COLOR_RGB2BGR))
plt.axis('off')
plt.title("Front-left view")

fig.add_subplot(rows, columns, 3)
plt.imshow(cv2.cvtColor(cv2.imread('./output/left.png'), cv2.COLOR_RGB2BGR))
plt.axis('off')
plt.title("Left view")

fig.add_subplot(rows, columns, 4)
plt.imshow(cv2.cvtColor(cv2.imread('./output/right.png'), cv2.COLOR_RGB2BGR))
plt.axis('off')
plt.title("Right view")
plt.show()
```


![png](./doc_image/output_68_0.png)


### Resourses

I did not described the theory behind the process. If you are interested about the theory, you can read the additional resourse files.

<a href="https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration">Camera Calibration</a>

<a href="https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html#epipolar-geometry">Epipolar Geometry</a>

<a href="https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_depthmap/py_depthmap.html#py-depthmap">Depth Map from Stereo Images</a>
