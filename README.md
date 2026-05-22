# all-sky-camera-calibration-pipeline
Multi-stage computational lens distortion correction pipeline for an all sky camera (ASC). A calibration image was taken of a 3D printed 'pinhole dome'. The pinhole pixel coordinates were then extracted using SAOImage DS9, as the light exposure was too high to use centroiding algorithms. I coded a 'physical model' to match the extracted coordinate data as precisely as possible using physical parameters such as focal length, tilt, and rotation. This was followed by a linear regression polynomial correction model to 'undistort' the image to an ideal projection model. I then used OpenCv's remap function to perform interlinear interpolation to apply this polynomial model to subsequent images taken by the ASC. 

The final correction model reached a precision of 0.478 ± 0.364 px above 60° altitude.

Developed as a final year astrophysics project at the University of Bristol, PHYS30034 Research Project in Physics. Mark yet to be released.

<img width="600" height="458" alt="Distortion Heatmap ASC" src="https://github.com/user-attachments/assets/312a08ff-60cc-4af4-bc67-0a415262e72e" />
<img width="519" height="500" alt="Residuals" src="https://github.com/user-attachments/assets/8ab26165-dc8e-4038-af91-2b87c9a326f5" />
<img width="600" height="458" alt="Polynomial Distortion Map" src="https://github.com/user-attachments/assets/fa759d74-edd2-4a51-825b-64aa158ee4aa" />
<img width="551" height="572" alt="Model Alignment" src="https://github.com/user-attachments/assets/5ebd9706-f0d2-4da6-8857-be4a2bf5c032" />

