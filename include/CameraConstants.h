// CameraConstants.h

#ifndef CAMERA_CONSTANTS_H
#define CAMERA_CONSTANTS_H

// Camera-related constants
const int SENSOR_RESOLUTION_X = 1280;        // Horizontal resolution (pixels)
const int SENSOR_RESOLUTION_Y = 720;         // Vertical resolution (pixels)
const int SENSOR_FPS = 30;                   // Frames per second

// Updated focal lengths and principal points from calibration
const float FOCAL_LENGTH_PX = 1246.21885f;   // fx from camera matrix
const float FOCAL_LENGTH_PY = 1248.72410f;   // fy from camera matrix
const float PRINCIPAL_POINT_X = 481.34518f;  // cx from camera matrix
const float PRINCIPAL_POINT_Y = 307.974119f; // cy from camera matrix

// Distortion coefficients from calibration
const float DISTORTION_COEFFS[5] = {
    0.09761881f,  // k1
    0.04685588f,  // k2
    0.01135237f,  // p1
    0.00166419f,  // p2
    0.67687489f   // k3
};

// Known dimensions of objects (in centimeters)
const float PARCEL_WIDTH = 18.0f;            // Parcel width in cm
const float PARCEL_HEIGHT = 12.0f;           // Parcel height in cm
const float PARCEL_DIAMETER = 6.0f;          // Ring diameter in cm

// Class IDs for object detection
const int CLASS_ID_PARCEL = 0;               // Adjust according to your model's class IDs
const int CLASS_ID_RING = 1;

#endif  // CAMERA_CONSTANTS_H
