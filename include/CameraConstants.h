// CameraConstants.h

#ifndef CAMERA_CONSTANTS_H
#define CAMERA_CONSTANTS_H

// Camera-related constants
const float SENSOR_WIDTH_MM = 3.68f;     // Sensor width in mm
const float SENSOR_HEIGHT_MM = 2.76f;    // Sensor height in mm
const int SENSOR_RESOLUTION_X = 1280;    // Horizontal resolution (pixels)
const int SENSOR_RESOLUTION_Y = 720;     // Vertical resolution (pixels)
const int SENSOR_FPS = 30;               // Frames per second
const float FOCAL_LENGTH_MM = 3.04f;     // Focal length in mm

// Known dimensions of objects (in centimeters)
const float PARCEL_WIDTH = 18.0f;        // Parcel width in cm
const float PARCEL_HEIGHT = 12.0f;       // Parcel height in cm
const float PARCEL_DIAMETER = 6.0f;      // Ring diameter in cm

// Class IDs for object detection
const int CLASS_ID_PARCEL = 0;           // Adjust according to your model's class IDs
const int CLASS_ID_RING = 1;

#endif  // CAMERA_CONSTANTS_H
