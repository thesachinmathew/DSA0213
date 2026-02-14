!pip install opencv-python opencv-python-headless

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files

# 1a. GRAYSCALE CONVERSION
uploaded = files.upload()
filename = list(uploaded.keys())[0]
img = cv2.imdecode(np.frombuffer(uploaded[filename], np.uint8), cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Original:")
cv2_imshow(img)
print("\nGrayscale:")
cv2_imshow(gray)

# 1b. GAUSSIAN BLUR
uploaded = files.upload()
img = cv2.imdecode(np.frombuffer(uploaded[list(uploaded.keys())[0]], np.uint8), cv2.IMREAD_COLOR)

blurred = cv2.GaussianBlur(img, (15, 15), 0)

print("Original:")
cv2_imshow(img)
print("\nBlurred:")
cv2_imshow(blurred)

# 1c. CANNY EDGE DETECTION (OUTLINE)
uploaded = files.upload()
img = cv2.imdecode(np.frombuffer(uploaded[list(uploaded.keys())[0]], np.uint8), cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

print("Original:")
cv2_imshow(img)
print("\nEdges:")
cv2_imshow(edges)

# 1d. DILATE IMAGE
uploaded = files.upload()
img = cv2.imdecode(np.frombuffer(uploaded[list(uploaded.keys())[0]], np.uint8), cv2.IMREAD_COLOR)

kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(img, kernel, iterations=1)

print("Original:")
cv2_imshow(img)
print("\nDilated:")
cv2_imshow(dilated)

# 1e. ERODE IMAGE
uploaded = files.upload()
img = cv2.imdecode(np.frombuffer(uploaded[list(uploaded.keys())[0]], np.uint8), cv2.IMREAD_COLOR)

kernel = np.ones((5, 5), np.uint8)
eroded = cv2.erode(img, kernel, iterations=1)

print("Original:")
cv2_imshow(img)
print("\nEroded:")
cv2_imshow(eroded)

# 2 & 3. VIDEO PROCESSING (Slow and Fast Motion)
uploaded = files.upload()
video_file = list(uploaded.keys())[0]

cap = cv2.VideoCapture(video_file)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_slow = cv2.VideoWriter('slow_motion.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
out_fast = cv2.VideoWriter('fast_motion.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    out_slow.write(frame)
    out_slow.write(frame)
    
    if frame_count % 2 == 0:
        out_fast.write(frame)
    
    frame_count += 1

cap.release()
out_slow.release()
out_fast.release()

print(f"Processed {frame_count} frames")
files.download('slow_motion.mp4')
files.download('fast_motion.mp4')

# 4. IMAGE SCALING
uploaded = files.upload()
img = cv2.imdecode(np.frombuffer(uploaded[list(uploaded.keys())[0]], np.uint8), cv2.IMREAD_COLOR)

bigger = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

smaller = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

print("Original:")
cv2_imshow(img)
print("\nBigger (2x):")
cv2_imshow(bigger)
print("\nSmaller (0.5x):")
cv2_imshow(smaller)

# 5. IMAGE ROTATION
uploaded = files.upload()
img = cv2.imdecode(np.frombuffer(uploaded[list(uploaded.keys())[0]], np.uint8), cv2.IMREAD_COLOR)

height, width = img.shape[:2]
center = (width // 2, height // 2)

matrix_cw = cv2.getRotationMatrix2D(center, -90, 1.0)
rotated_cw = cv2.warpAffine(img, matrix_cw, (height, width))

matrix_ccw = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated_ccw = cv2.warpAffine(img, matrix_ccw, (height, width))

print("Original:")
cv2_imshow(img)
print("\nClockwise:")
cv2_imshow(rotated_cw)
print("\nCounter-Clockwise:")
cv2_imshow(rotated_ccw)

# 6. IMAGE TRANSLATION (MOVING)
uploaded = files.upload()
img = cv2.imdecode(np.frombuffer(uploaded[list(uploaded.keys())[0]], np.uint8), cv2.IMREAD_COLOR)

height, width = img.shape[:2]

tx, ty = 100, 50
matrix = np.float32([[1, 0, tx], [0, 1, ty]])
translated = cv2.warpAffine(img, matrix, (width, height))

print("Original:")
cv2_imshow(img)
print("\nTranslated:")
cv2_imshow(translated)

# 7. AFFINE TRANSFORMATION
uploaded = files.upload()
img = cv2.imdecode(np.frombuffer(uploaded[list(uploaded.keys())[0]], np.uint8), cv2.IMREAD_COLOR)

height, width = img.shape[:2]

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

matrix = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(img, matrix, (width, height))

print("Original:")
cv2_imshow(img)
print("\nAffine Transformed:")
cv2_imshow(affine)

# 8. PERSPECTIVE TRANSFORMATION - IMAGE
uploaded = files.upload()
img = cv2.imdecode(np.frombuffer(uploaded[list(uploaded.keys())[0]], np.uint8), cv2.IMREAD_COLOR)

pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
perspective = cv2.warpPerspective(img, matrix, (300, 300))

print("Original:")
cv2_imshow(img)
print("\nPerspective Transformed:")
cv2_imshow(perspective)

# 9. PERSPECTIVE TRANSFORMATION - VIDEO
uploaded = files.upload()
video_file = list(uploaded.keys())[0]

cap = cv2.VideoCapture(video_file)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

pts1 = np.float32([[50, 50], [width-50, 50], [50, height-50], [width-50, height-50]])
pts2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)

out = cv2.VideoWriter('perspective_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (400, 400))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    transformed = cv2.warpPerspective(frame, matrix, (400, 400))
    out.write(transformed)

cap.release()
out.release()
files.download('perspective_video.mp4')

# 10. HOMOGRAPHY TRANSFORMATION
uploaded = files.upload()
img = cv2.imdecode(np.frombuffer(uploaded[list(uploaded.keys())[0]], np.uint8), cv2.IMREAD_COLOR)

height, width = img.shape[:2]

src = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
dst = np.float32([[50, 0], [width-50, 50], [0, height-50], [width, height]])

H, _ = cv2.findHomography(src, dst)
homography = cv2.warpPerspective(img, H, (width, height))

print("Original:")
cv2_imshow(img)
print("\nHomography Transformed:")
cv2_imshow(homography)
print("\nHomography Matrix:")
print(H)


# 11 Direct Linear Transformation (DLT)
import cv2
import numpy as np
pts1 = np.float32([[0,0],[100,0],[0,100],[100,100]])
pts2 = np.float32([[10,20],[120,30],[20,140],[130,150]])
H, _ = cv2.findHomography(pts1, pts2, 0)
print(H)

#question 12
# Canny Edge Detection
img = cv2.imread('image.jpg',0)
edges = cv2.Canny(img,100,200)
cv2.imshow('Canny',edges)
cv2.waitKey(0)

#question 13
# Sobel X
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
cv2.imshow('Sobel X',sobelx)
cv2.waitKey(0)

#question 14
# Sobel Y
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
cv2.imshow('Sobel Y',sobely)
cv2.waitKey(0)

#question 15
# Sobel XY
sobelxy = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3)
cv2.imshow('Sobel XY',sobelxy)
cv2.waitKey(0)

#question 16
# Laplacian Negative Center
kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
sharp = cv2.filter2D(img,-1,kernel)
cv2.imshow('Sharp16',sharp)
cv2.waitKey(0)

#question 17
# Laplacian Diagonal Extension
kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
sharp = cv2.filter2D(img,-1,kernel)
cv2.imshow('Sharp17',sharp)
cv2.waitKey(0)

#question 18
# Laplacian Positive Center
kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
sharp = cv2.filter2D(img,-1,kernel)
cv2.imshow('Sharp18',sharp)
cv2.waitKey(0)

#question 19
# Unsharp Masking
img_color = cv2.imread('image.jpg')
blur = cv2.GaussianBlur(img_color,(5,5),0)
unsharp = cv2.addWeighted(img_color,1.5,blur,-0.5,0)
cv2.imshow('Unsharp',unsharp)
cv2.waitKey(0)

#question 20
# High Boost Filtering
mask = cv2.subtract(img_color,blur)
highboost = cv2.add(img_color,mask)
cv2.imshow('HighBoost',highboost)
cv2.waitKey(0)

#question 21
# Gradient Masking
gradx = cv2.Sobel(img,cv2.CV_64F,1,0,3)
grady = cv2.Sobel(img,cv2.CV_64F,0,1,3)
gradient = cv2.magnitude(gradx,grady)
cv2.imshow('Gradient',gradient)
cv2.waitKey(0)

#question 22
# Watermark
watermark = cv2.imread('logo.png')
watermark = cv2.resize(watermark,(100,100))
img_color[0:100,0:100] = watermark
cv2.imshow('Watermarked',img_color)
cv2.waitKey(0)

#question 23
# Boundary using Kernel
kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
boundary = cv2.filter2D(img,-1,kernel)
cv2.imshow('Boundary',boundary)
cv2.waitKey(0)

#question 24
# Erosion
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,1)
cv2.imshow('Erosion',erosion)
cv2.waitKey(0)

#question 25
# Dilation
dilation = cv2.dilate(img,kernel,1)
cv2.imshow('Dilation',dilation)
cv2.waitKey(0)

#question 26
# Opening
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
cv2.imshow('Opening',opening)
cv2.waitKey(0)

#question 27
# Closing
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
cv2.imshow('Closing',closing)
cv2.waitKey(0)

#question 28
# Morphological Gradient
morph_grad = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
cv2.imshow('MorphGradient',morph_grad)
cv2.waitKey(0)

#question 29
# Top Hat
tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
cv2.imshow('TopHat',tophat)
cv2.waitKey(0)

#question 30
# Black Hat
blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
cv2.imshow('BlackHat',blackhat)
cv2.waitKey(0)

#question 31
# Object Detection using Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img2 = cv2.imread('watch.jpg')
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
objects = face_cascade.detectMultiScale(gray2,1.3,5)
for (x,y,w,h) in objects:
    cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('Object',img2)
cv2.waitKey(0)

#question 32
# Play Video in Reverse
cap = cv2.VideoCapture('video.mp4')
frames=[]
while True:
    ret,frame = cap.read()
    if not ret: break
    frames.append(frame)
for f in reversed(frames):
    cv2.imshow('Reverse',f)
    if cv2.waitKey(30)==27: break

#question 33
# Face Detection
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('Face',frame)
    if cv2.waitKey(1)==27: break

#question 34
# Vehicle Detection
car_cascade = cv2.CascadeClassifier('cars.xml')
cap = cv2.VideoCapture('traffic.mp4')
while True:
    ret,frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray,1.1,3)
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('Vehicle',frame)
    if cv2.waitKey(1)==27: break

#question 35
# Draw Rectangle and Extract Object
img3 = cv2.imread('image.jpg')
cv2.rectangle(img3,(50,50),(200,200),(0,255,0),2)
roi = img3[50:200,50:200]
cv2.imshow('Rectangle',img3)
cv2.imshow('Extracted',roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
