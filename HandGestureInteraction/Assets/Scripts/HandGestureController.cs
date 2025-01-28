using UnityEngine;
using NativeWebSocket;
using System;
using System.Collections.Generic;

[System.Serializable]
public class FingerJoint
{
    public Transform joint;
    public Vector3 openRotation;
    public Vector3 closedRotation;
}

[System.Serializable]
public class Finger
{
    public string name;
    public List<FingerJoint> joints;
}

[System.Serializable]
public class FingerState
{
    public bool isExtended;
    public float bendAngle;  // 0 = straight, 1 = fully bent
}

[System.Serializable]
public class HandGestureMessage : HandTrackingMessage
{
    public string gesture_type;
    public float gesture_confidence;
    public FingerState[] fingerStates;  // thumb, index, middle, ring, pinky
    public string dynamic_gesture_type;
    public float dynamic_gesture_confidence;
}

public class HandGestureController : MonoBehaviour
{
    WebSocket websocket;
    bool isConnected = false;

    [Header("Finger References")]
    public List<Finger> fingers;  // Set up in inspector

    [Header("Gesture Settings")]
    public float gestureSpeed = 8.0f;  // Speed of finger movement
    public bool invertFingerRotation = false;  // In case we need to flip rotation direction

     [Header("Position Settings")]
    public float moveSpeed = 5.0f;
    public float horizontalBound = 8.0f;  // Increased for wider movement
    public float verticalBound = 6.0f;    // Increased for taller movement
    public float depthBound = 4.0f;  
    public float minDepthOffset = -14.0f;  // New variable
    public float maxDepthOffset = -10.0f;  // New variable
    public float sensitivityMultiplier = 2.0f;  // Amplifies hand movement
    public bool enablePositionTracking = true;

    [Header("Flower Settings")]
    public Transform ground;  // Reference to the ground object (pot)
    public float snapDistance = 2.0f;  // Maximum distance to snap to pot
    private Dictionary<string, Vector3> flowerSnapPositions = new Dictionary<string, Vector3>()
    {
        {"flowers_1", new Vector3(8.58f, 6.13f, -3.56f)},
        {"flowers_02", new Vector3(8.24f, 6.13f, -3.56f)},
        {"flowers_02_01", new Vector3(8.77f, 6.13f, -3.56f)},
        {"flowers_02_02", new Vector3(8.44f, 6.13f, -3.56f)}
    };

    [Header("Flower References")]
    public GameObject flowers_1;
    public GameObject flowers_02;
    public GameObject flowers_02_01;
    public GameObject flowers_02_02;

    [Header("Day Night Controller")]
    public GameObject dayNightController;  // Assign this in Inspector
    private bool isDayNightEnabled = true;  // Track if day/night cycle is active
    private bool wasPinching = false;
    private float targetRotation = 0f;  // Target Z rotation
    //private float rotationSpeed = 10f;   // Speed of rotation transition
    //private bool isSwipeActive = false;  // Track if swipe is active

    private bool isHandDetected = false;
    private float reconnectTimer = 0f;
    private bool isReconnecting = false;
    private readonly float RECONNECT_INTERVAL = 2f;
    private string currentGesture = "NONE";
    
    // Position tracking variables
    private Vector3 initialPosition;
    private Vector3 targetPosition;
    private Quaternion initialRotation;
    private float zPosition;  // Store fixed Z position

    private GameObject grabbedObject;
    private bool isGrabbing = false;
    private Vector3 grabbedObjectOffset;
    private readonly float grabDistance = 1.0f; // Maximum distance to grab object
    private readonly float grabSpeed = 10f; 
    // Add these at top of class with other private variables
    private bool isAnimating = false;  // Track if animation is in progress
    private float currentAngle = 0f;   // Current rotation angle
    private float targetAngle = 0f;    // Target rotation angle
    private float returnDelay = 0.2f;  // Time to wait at max angle before returning
    private float returnTimer = 0f;    // Timer for return delay
    private float rotationSpeed = 240f; // Degrees per second (adjust for speed)

    private BoxCollider handCollider;
    private bool isColliding = false;
    [Header("Interaction Settings")]
    public float pushForce = 5f;  // How strongly the hand pushes objects
    private Vector3 previousHandPosition;  // To calculate hand velocity


    private void HandleDynamicGesture(string gestureType, float confidence)
    {
        // Skip if confidence too low or already animating
        if (confidence < 0.7f || isAnimating)
            return;

        switch (gestureType)
        {
            case "SWIPE_LEFT":
                targetAngle = -30f;
                isAnimating = true;
                returnTimer = returnDelay;
                break;
            case "SWIPE_RIGHT":
                targetAngle = 30f;
                isAnimating = true;
                returnTimer = returnDelay;
                break;
        }
    }


    void Start()
    {
        Debug.Log("Hand Gesture Controller initializing...");
        handCollider = gameObject.AddComponent<BoxCollider>();
        handCollider.isTrigger = true;
        handCollider.size = new Vector3(0.15f, 0.15f, 0.15f);
        previousHandPosition = transform.position;
        SaveInitialTransform();
        SetupFlowerConstraints();
        ConnectToServer();
    }

    void SaveInitialTransform()
    {
        initialPosition = transform.position;
        zPosition = initialPosition.z;  // Store initial Z position
        targetPosition = initialPosition;
        initialRotation = transform.rotation;
        Debug.Log($"Initial position set to: {initialPosition}");
    }

    void SetupFlowerConstraints()
    {
        GameObject[] flowers = { flowers_1, flowers_02, flowers_02_01, flowers_02_02 };
        foreach (GameObject flower in flowers)
        {
            if (flower != null)
            {
                Rigidbody rb = flower.GetComponent<Rigidbody>();
                if (rb == null)
                {
                    rb = flower.AddComponent<Rigidbody>();
                }
                
                // Only freeze Y position
                rb.constraints = RigidbodyConstraints.FreezePositionY;
                rb.isKinematic = true;
                rb.useGravity = false;
                Debug.Log($"Set minimal constraints for {flower.name}");
            }
        }
    }

    private void ResetFlowerRotations()
    {
        GameObject[] flowers = { flowers_1, flowers_02, flowers_02_01, flowers_02_02 };
        foreach (GameObject flower in flowers)
        {
            if (flower != null)
            {
                Vector3 currentRotation = flower.transform.rotation.eulerAngles;
                flower.transform.rotation = Quaternion.Euler(currentRotation.x, currentRotation.y, 0f);
                Debug.Log($"Reset rotation for {flower.name}");
            }
        }
    }

    async void ConnectToServer()
    {
        websocket = new WebSocket("ws://127.0.0.1:8765");
        
        websocket.OnOpen += () =>
        {
            isConnected = true;
            isReconnecting = false;
            Debug.Log("Connected to server!");
        };

        websocket.OnError += (e) =>
        {
            Debug.LogError($"WebSocket Error: {e}");
            isConnected = false;
        };

        websocket.OnClose += (e) =>
        {
            Debug.Log("Connection closed");
            isConnected = false;
            isReconnecting = true;
        };

        websocket.OnMessage += (bytes) =>
        {
            ProcessWebSocketMessage(bytes);
        };

        Debug.Log("Attempting to connect...");
        await websocket.Connect();
    }

    void ProcessWebSocketMessage(byte[] bytes)
    {
        try
        {
            string message = System.Text.Encoding.UTF8.GetString(bytes);
            HandGestureMessage data = JsonUtility.FromJson<HandGestureMessage>(message);
            
            isHandDetected = data.hands_detected;
            
            if (isHandDetected)
            {
                if (enablePositionTracking && data.hand_position != null)
                {
                    // Log raw values from server
                    Debug.Log($"Raw position from server - X: {data.hand_position.x:F3}, Y: {data.hand_position.y:F3}, Z: {data.hand_position.z:F3}");

                    // Handle X and Y movement
                    float amplifiedX = -data.hand_position.x * sensitivityMultiplier;
                    float amplifiedY = -data.hand_position.y * sensitivityMultiplier;
                    
                    // Handle Z movement - keep original value but invert for natural movement
                    float normalizedZ = -data.hand_position.z;  // Invert Z for natural movement
                    
                    // Clamp all values
                    float clampedX = Mathf.Clamp(amplifiedX, -1f, 1f);
                    float clampedY = Mathf.Clamp(amplifiedY, -1f, 1f);
                    float clampedZ = Mathf.Clamp(normalizedZ, -1f, 1f);

                    // Calculate new position
                    float mappedZ = Mathf.Lerp(maxDepthOffset, minDepthOffset, (clampedZ + 1f) * 0.5f);
                    Vector3 newPosition = new Vector3(
                        initialPosition.x + (clampedX * horizontalBound),
                        initialPosition.y + (clampedY * verticalBound),
                        mappedZ
                    );
                    // Update target position
                    targetPosition = newPosition;

                    // Detailed debug logging
                    Debug.Log($"Processed values - X: {clampedX:F3}, Y: {clampedY:F3}, Z: {clampedZ:F3}");
                    Debug.Log($"Current position: {transform.position:F3}");
                    Debug.Log($"Target position: {targetPosition:F3}");
                    Debug.Log($"Depth movement: Current Z = {transform.position.z:F3}, Target Z = {targetPosition.z:F3}");
                }

                if (!string.IsNullOrEmpty(data.dynamic_gesture_type))
                {
                    HandleDynamicGesture(data.dynamic_gesture_type, data.dynamic_gesture_confidence);
                }
                else
                {
                    // Reset flower rotations when no gesture is detected
                    ResetFlowerRotations();
                }

                // Update finger gestures
                UpdateHandGesture(data);
                currentGesture = data.gesture_type;
            }
            else
            {
                // Reset flower rotations when hand is not detected
                ResetFlowerRotations();
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error processing message: {e.Message}");
        }
    }
    
    void UpdateHandGesture(HandGestureMessage data)
    {
        Debug.Log($"Received gesture: {data.gesture_type}");
        
        // Check for pinch gesture state change
        bool isPinching = (data.gesture_type == "PINCH");
        if (isPinching != wasPinching)  // Only handle when pinch state changes
        {
            if (isPinching)  // Pinch started
            {
                // Turn off day/night when pinch starts
                if (dayNightController != null)
                {
                    dayNightController.SetActive(false);
                    Debug.Log("Day Night cycle disabled");
                }
            }
            else  // Pinch ended
            {
                // Turn on day/night when pinch ends
                if (dayNightController != null)
                {
                    dayNightController.SetActive(true);
                    Debug.Log("Day Night cycle enabled");
                }
            }
            wasPinching = isPinching;
        }

        // Handle the gestures
        if (data.gesture_type == "GRAB")
        {
            if (!isGrabbing)
            {
                TryGrabNearestObject();
            }
            SetAllFingers(false);
        }
        else if (data.gesture_type == "OPEN_PALM")
        {
            if (isGrabbing && grabbedObject != null)
            {
                ReleaseObject();
            }
            SetAllFingers(true);
        }
        else if (data.gesture_type == "PINCH")
        {
            SetPinchGesture();  // Just handle finger animation
        }
        else
        {
            // Handle other gestures
            switch(data.gesture_type)
            {
                case "POINT":
                    SetPointGesture();
                    break;
                default:
                    if (data.fingerStates != null && data.fingerStates.Length == 5)
                    {
                        UpdateIndividualFingers(data.fingerStates);
                    }
                    else
                    {
                        SetAllFingers(true);
                    }
                    break;
            }
        }
    }

    void TryGrabNearestObject()
    {
        // Find all nearby interactable objects
        Collider[] hitColliders = Physics.OverlapSphere(transform.position, grabDistance);
        float closestDistance = float.MaxValue;
        GameObject closestObject = null;

        foreach (var hitCollider in hitColliders)
        {
            if (hitCollider.CompareTag("Interactable"))
            {
                float distance = Vector3.Distance(transform.position, hitCollider.transform.position);
                if (distance < closestDistance)
                {
                    closestDistance = distance;
                    closestObject = hitCollider.gameObject;
                }
            }
        }

        if (closestObject != null)
        {
            // Grab the object
            grabbedObject = closestObject;
            isGrabbing = true;
            
            // Store the initial offset between hand and object
            grabbedObjectOffset = grabbedObject.transform.position - transform.position;
            
            // Disable object's rigidbody physics while grabbed
            Rigidbody rb = grabbedObject.GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.isKinematic = true;
                rb.useGravity = false;
            }
            
            Debug.Log($"Grabbed object: {grabbedObject.name}");
        }
    }


    void ReleaseObject()
    {
        if (grabbedObject != null)
        {
            Rigidbody rb = grabbedObject.GetComponent<Rigidbody>();
            if (rb != null)
            {
                // First set the rotation to upright
                grabbedObject.transform.rotation = Quaternion.Euler(0f, 0f, 0f);
                
                if (flowerSnapPositions.ContainsKey(grabbedObject.name))
                {
                    Vector3 snapPosition = flowerSnapPositions[grabbedObject.name];
                    float distanceToSnap = Vector3.Distance(grabbedObject.transform.position, snapPosition);
                    
                    if (distanceToSnap < snapDistance)
                    {
                        grabbedObject.transform.position = snapPosition;
                        Debug.Log($"Snapped {grabbedObject.name} to position: {snapPosition}");
                        
                        // Only freeze Y position
                        rb.constraints = RigidbodyConstraints.FreezePositionY;
                    }
                    else
                    {
                        Vector3 frozenPosition = grabbedObject.transform.position;
                        frozenPosition.y = 6.13f;
                        grabbedObject.transform.position = frozenPosition;
                    }
                }

                rb.isKinematic = true;
                rb.useGravity = false;
            }
            grabbedObject = null;
        }
        isGrabbing = false;
    }


    void UpdateIndividualFingers(FingerState[] states)
    {
        if (states == null)
        {
            Debug.LogError("Received null finger states");
            return;
        }

        Debug.Log($"Updating {states.Length} fingers");
        
        for (int i = 0; i < fingers.Count && i < states.Length; i++)
        {
            var finger = fingers[i];
            var state = states[i];
            
            if (state == null)
            {
                Debug.LogError($"Null state for finger {i}");
                continue;
            }
            
            Debug.Log($"{finger.name}: extended={state.isExtended}, bend={state.bendAngle:F2}");
            
            foreach (var joint in finger.joints)
            {
                if (joint.joint != null)
                {
                    // More aggressive finger movement
                    float targetBend = state.isExtended ? 0f : 1f;
                    
                    Vector3 targetRotation = Vector3.Lerp(
                        joint.openRotation,
                        joint.closedRotation,
                        targetBend
                    );

                    // Modified thumb handling
                    if (finger.name == "Thumb" && !state.isExtended)
                    {
                        targetRotation += new Vector3(0, 0, 45f);
                    }

                    joint.joint.localRotation = Quaternion.Lerp(
                        joint.joint.localRotation,
                        Quaternion.Euler(targetRotation),
                        Time.deltaTime * gestureSpeed * 2f
                    );
                }
            }
        }
    }

    void SetAllFingers(bool open)
    {
        Debug.Log($"Setting all fingers to {(open ? "open" : "closed")}");
        foreach (var finger in fingers)
        {
            if (finger.name == "Thumb")
            {
                Debug.Log($"Thumb gesture: {(open ? "open" : "closed")}");
            }
            
            foreach (var joint in finger.joints)
            {
                Vector3 targetRotation = open ? joint.openRotation : joint.closedRotation;
                if (joint.joint != null)
                {
                    // Add extra rotation for thumb when closing
                    if (finger.name == "Thumb" && !open)
                    {
                        targetRotation.z += 45f; // Additional inward rotation for thumb
                    }
                    
                    joint.joint.localRotation = Quaternion.Lerp(
                        joint.joint.localRotation,
                        Quaternion.Euler(targetRotation),
                        Time.deltaTime * gestureSpeed
                    );
                    Debug.Log($"Setting {finger.name} joint rotation to {targetRotation}");
                }
                else
                {
                    Debug.LogError($"Null joint found in finger {finger.name}");
                }
            }
        }
    }

    void SetPinchGesture()
    {
        foreach (var finger in fingers)
        {
            bool shouldClose = finger.name == "Thumb" || finger.name == "Index";
            foreach (var joint in finger.joints)
            {
                Vector3 targetRotation = shouldClose ? 
                    Vector3.Lerp(joint.openRotation, joint.closedRotation, 0.7f) : 
                    joint.openRotation;

                joint.joint.localRotation = Quaternion.Lerp(
                    joint.joint.localRotation,
                    Quaternion.Euler(targetRotation),
                    Time.deltaTime * gestureSpeed
                );
            }
        }
    }

    void SetPointGesture()
    {
        foreach (var finger in fingers)
        {
            bool shouldBeOpen = finger.name == "Index";
            foreach (var joint in finger.joints)
            {
                Vector3 targetRotation = shouldBeOpen ? joint.openRotation : joint.closedRotation;
                joint.joint.localRotation = Quaternion.Lerp(
                    joint.joint.localRotation,
                    Quaternion.Euler(targetRotation),
                    Time.deltaTime * gestureSpeed
                );
            }
        }
    }

    void Update()
    {
        if (websocket != null)
        {
            websocket.DispatchMessageQueue();
        }

        // Handle reconnection logic
        if (!isConnected && !isReconnecting)
        {
            reconnectTimer += Time.deltaTime;
            if (reconnectTimer >= RECONNECT_INTERVAL)
            {
                reconnectTimer = 0f;
                Debug.Log("Attempting to reconnect...");
                ConnectToServer();
            }
        }

        if (isAnimating)
        {
            // Calculate rotation step
            float step = rotationSpeed * Time.deltaTime;
            
            if (returnTimer > 0)
            {
                // Moving to target angle
                currentAngle = Mathf.MoveTowards(currentAngle, targetAngle, step);
                if (Mathf.Approximately(currentAngle, targetAngle))
                {
                    returnTimer -= Time.deltaTime;
                }
            }
            else
            {
                // Return to zero
                currentAngle = Mathf.MoveTowards(currentAngle, 0f, step);
                if (Mathf.Approximately(currentAngle, 0f))
                {
                    isAnimating = false;
                }
            }

            // Apply rotation to all flowers
            GameObject[] flowers = { flowers_1, flowers_02, flowers_02_01, flowers_02_02 };
            foreach (GameObject flower in flowers)
            {
                if (flower != null)
                {
                    Vector3 currentRotation = flower.transform.rotation.eulerAngles;
                    flower.transform.rotation = Quaternion.Euler(currentRotation.x, currentRotation.y, currentAngle);
                }
            }
        }

        // Update hand position
        if (enablePositionTracking && isHandDetected)
        {
            Vector3 currentPos = transform.position;
            Vector3 smoothedPosition = Vector3.Lerp(
                currentPos,
                targetPosition,
                moveSpeed * Time.deltaTime
            );
            
            transform.position = smoothedPosition;
            
            // Update grabbed object position if we're holding one
            if (isGrabbing && grabbedObject != null)
            {
                Vector3 targetObjectPosition = transform.position + grabbedObjectOffset;
                grabbedObject.transform.position = Vector3.Lerp(
                    grabbedObject.transform.position,
                    targetObjectPosition,
                    grabSpeed * Time.deltaTime
                );
            }
            
            previousHandPosition = currentPos;
        }
    }

    private bool IsNearGround()
    {
        if (ground != null && grabbedObject != null && flowerSnapPositions.ContainsKey(grabbedObject.name))
        {
            Vector3 snapPosition = flowerSnapPositions[grabbedObject.name];
            float distance = Vector3.Distance(grabbedObject.transform.position, snapPosition);
            Debug.Log($"Distance to snap position: {distance}");
            return distance < snapDistance;
        }
        return false;
    }

    private async void OnApplicationQuit()
    {
        if (websocket != null)
        {
            await websocket.Close();
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Interactable"))
        {
            isColliding = true;
            Debug.Log("Hand entered cube trigger");
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("Interactable"))
        {
            isColliding = false;
            Debug.Log("Hand exited cube trigger");
        }
    }

    private void HandlePinchGesture()
    {
        if (dayNightController != null)
        {
            isDayNightEnabled = !isDayNightEnabled;  // Toggle the state
            dayNightController.SetActive(isDayNightEnabled);
            Debug.Log($"Day Night cycle is now {(isDayNightEnabled ? "enabled" : "disabled")}");
        }
        else
        {
            Debug.LogWarning("Day Night Controller reference is missing!");
        }
    }

}
// 3.621272
// 2.733984e-07
// 3.621272