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
    public float bendAngle;  
}

[System.Serializable]
public class HandGestureMessage : HandTrackingMessage
{
    public string gesture_type;
    public float gesture_confidence;
    public FingerState[] fingerStates;  
    public string dynamic_gesture_type;
    public float dynamic_gesture_confidence;
}

public class HandGestureController : MonoBehaviour
{
    WebSocket websocket;
    bool isConnected = false;

    [Header("Finger References")]
    public List<Finger> fingers;  

    [Header("Gesture Settings")]
    public float gestureSpeed = 8.0f;  
    public bool invertFingerRotation = false;  

     [Header("Position Settings")]
    public float moveSpeed = 5.0f;
    public float horizontalBound = 8.0f;  
    public float verticalBound = 6.0f;    
    public float depthBound = 4.0f;  
    public float minDepthOffset = -14.0f;  
    public float maxDepthOffset = -10.0f;  
    public float sensitivityMultiplier = 2.0f;  
    public bool enablePositionTracking = true;

    [Header("Flower Settings")]
    public Transform ground;  
    public float snapDistance = 2.0f;  
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
    public GameObject dayNightController;  
    private bool isDayNightEnabled = true; 
    private bool wasPinching = false;
    private float targetRotation = 0f;  
    //private float rotationSpeed = 10f;  
    //private bool isSwipeActive = false;  

    private bool isHandDetected = false;
    private float reconnectTimer = 0f;
    private bool isReconnecting = false;
    private readonly float RECONNECT_INTERVAL = 2f;
    private string currentGesture = "NONE";

    private Vector3 initialPosition;
    private Vector3 targetPosition;
    private Quaternion initialRotation;
    private float zPosition;  

    private GameObject grabbedObject;
    private bool isGrabbing = false;
    private Vector3 grabbedObjectOffset;
    private readonly float grabDistance = 1.0f;
    private readonly float grabSpeed = 10f; 
    private bool isAnimating = false; 
    private float currentAngle = 0f;   
    private float targetAngle = 0f;    
    private float returnDelay = 0.2f;  
    private float returnTimer = 0f;    
    private float rotationSpeed = 240f; 

    private BoxCollider handCollider;
    private bool isColliding = false;
    [Header("Interaction Settings")]
    public float pushForce = 5f;  
    private Vector3 previousHandPosition;  


    private void HandleDynamicGesture(string gestureType, float confidence)
    {
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
        zPosition = initialPosition.z;  
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
                   
                    Debug.Log($"Raw position from server - X: {data.hand_position.x:F3}, Y: {data.hand_position.y:F3}, Z: {data.hand_position.z:F3}");

                    float amplifiedX = -data.hand_position.x * sensitivityMultiplier;
                    float amplifiedY = -data.hand_position.y * sensitivityMultiplier;                 
                   
                    float normalizedZ = -data.hand_position.z; 
                    
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
                    ResetFlowerRotations();
                }

                UpdateHandGesture(data);
                currentGesture = data.gesture_type;
            }
            else
            {
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
        
        bool isPinching = (data.gesture_type == "PINCH");
        if (isPinching != wasPinching)  
        {
            if (isPinching)  // Pinch started
            {
                if (dayNightController != null)
                {
                    dayNightController.SetActive(false);
                    Debug.Log("Day Night cycle disabled");
                }
            }
            else  // Pinch ended
            {
                if (dayNightController != null)
                {
                    dayNightController.SetActive(true);
                    Debug.Log("Day Night cycle enabled");
                }
            }
            wasPinching = isPinching;
        }

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
            SetPinchGesture();  
        }
        else
        {
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
            grabbedObject = closestObject;
            isGrabbing = true;
            
            grabbedObjectOffset = grabbedObject.transform.position - transform.position;
            
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
                grabbedObject.transform.rotation = Quaternion.Euler(0f, 0f, 0f);
                
                if (flowerSnapPositions.ContainsKey(grabbedObject.name))
                {
                    Vector3 snapPosition = flowerSnapPositions[grabbedObject.name];
                    float distanceToSnap = Vector3.Distance(grabbedObject.transform.position, snapPosition);
                    
                    if (distanceToSnap < snapDistance)
                    {
                        grabbedObject.transform.position = snapPosition;
                        Debug.Log($"Snapped {grabbedObject.name} to position: {snapPosition}");
                        
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
                    float targetBend = state.isExtended ? 0f : 1f;
                    
                    Vector3 targetRotation = Vector3.Lerp(
                        joint.openRotation,
                        joint.closedRotation,
                        targetBend
                    );

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
                    if (finger.name == "Thumb" && !open)
                    {
                        targetRotation.z += 45f; 
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
            float step = rotationSpeed * Time.deltaTime;
            
            if (returnTimer > 0)
            {
                currentAngle = Mathf.MoveTowards(currentAngle, targetAngle, step);
                if (Mathf.Approximately(currentAngle, targetAngle))
                {
                    returnTimer -= Time.deltaTime;
                }
            }
            else
            {
                currentAngle = Mathf.MoveTowards(currentAngle, 0f, step);
                if (Mathf.Approximately(currentAngle, 0f))
                {
                    isAnimating = false;
                }
            }

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

        if (enablePositionTracking && isHandDetected)
        {
            Vector3 currentPos = transform.position;
            Vector3 smoothedPosition = Vector3.Lerp(
                currentPos,
                targetPosition,
                moveSpeed * Time.deltaTime
            );
            
            transform.position = smoothedPosition;
            
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
            isDayNightEnabled = !isDayNightEnabled;  
            dayNightController.SetActive(isDayNightEnabled);
            Debug.Log($"Day Night cycle is now {(isDayNightEnabled ? "enabled" : "disabled")}");
        }
        else
        {
            Debug.LogWarning("Day Night Controller reference is missing!");
        }
    }

}
