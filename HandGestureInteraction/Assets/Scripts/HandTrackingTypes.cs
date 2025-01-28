using UnityEngine;
using System;

namespace HandTracking
{
    [Serializable]
    public class HandLandmark
    {
        public float x;
        public float y;
        public float z;
    }

    [Serializable]
    public class HandTrackingData
    {
        public string type;
        public bool camera_active;
        public bool hands_detected;
        public HandLandmark[][] hand_landmarks;
    }
}
