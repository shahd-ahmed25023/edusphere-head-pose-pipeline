Face Direction Detection in Video
A computer vision project that uses MediaPipe and OpenCV to detect facial landmarks and estimate head pose from a video. The system analyzes face orientation frame by frame, displays the detected direction and rotation angles on the video, and saves the processed result as a new output video.

Features
Real-time facial landmark detection using MediaPipe Face Landmarker.
Head pose estimation based on facial feature geometry.
Detection of face direction (right, left, up, down, and forward).
Visualization of head orientation directly on video frames.
Display of rotation values around the x, y, and z axes.
Automatic saving of the processed video with annotations.
How It Works
The program loads the input video Aaa.mp4.
Each frame is horizontally flipped before processing.
Facial landmarks are detected using the MediaPipe Face Landmarker model.
Head pose is estimated using selected facial landmark coordinates.
Rotation angles are calculated and analyzed.
The face direction is classified based on angle thresholds.
Direction labels, angle values, and a visualization line are drawn on each frame.
The processed frames are written to a new output video.
Direction Codes
Face Direction	Displayed Label
Right	a
Left	b
Up	c
Down	d
Forward	Forward
The direction is determined using head rotation angles with an approximate threshold of 10°.

Project Structure
project/
├── Aaa.mp4
├── face_landmarker.task
├── head_pose.py
└── output.mp4
Required Files
Aaa.mp4 - Input video to be analyzed.
face_landmarker.task - MediaPipe Face Landmarker model file.
head_pose.py - Main project script.
output.mp4 - Generated output video containing head pose information.
Requirements
Python 3.x
OpenCV
MediaPipe
NumPy
Installation
pip install opencv-python-headless mediapipe numpy
Usage
python head_pose.py
After execution, the processed video will be saved automatically in the project directory.

Output
The output video contains:

Face direction labels (a, b, c, d, or Forward)
Head rotation angles:
Pitch (x)
Yaw (y)
Roll (z)
A directional line indicating the estimated head orientation
Technologies Used
Python
MediaPipe
OpenCV
NumPy
Applications
Student attention monitoring
Human-computer interaction systems
Driver monitoring systems
Behavioral analysis
Face tracking and pose estimation research
License
This project is intended for educational and research purposes.
