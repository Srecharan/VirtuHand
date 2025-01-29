using UnityEngine;
using NativeWebSocket;
using System;

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

public class EnhancedTestClient : MonoBehaviour
{
    WebSocket websocket;
    bool isConnected = false;
    
    public GameObject statusCube;
    public GameObject handIndicator;
    
    private readonly float xMultiplier = 30f;
    private readonly float yMultiplier = 30f;
    private readonly float zMultiplier = 50f;
    
    private Vector3 targetPosition;
    private bool isHandDetected = false;
    private Vector3 initialPosition;

    void Awake()
    {
        if (statusCube == null)
        {
            Debug.LogError("Status Cube is not assigned!");
            statusCube = GameObject.Find("StatusCube");
        }
        
        if (handIndicator == null)
        {
            Debug.LogError("Hand Indicator is not assigned!");
            handIndicator = GameObject.Find("HandIndicator");
        }

        if (handIndicator != null)
        {
            initialPosition = handIndicator.transform.position;
            Debug.Log($"Initial position set to: {initialPosition}");
        }
    }

    async void Start()
    {
        Debug.Log("Starting enhanced test client...");
        websocket = new WebSocket("ws://127.0.0.1:8765");
        
        websocket.OnOpen += () => {
            Debug.Log("Connected to hand tracking server!");
            isConnected = true;
            if (statusCube != null)
            {
                statusCube.GetComponent<Renderer>().material.color = Color.green;
            }
        };

        websocket.OnMessage += (bytes) => {
            var message = System.Text.Encoding.UTF8.GetString(bytes);
            ProcessHandTrackingData(message);
        };

        websocket.OnError += (e) => {
            Debug.LogError($"WebSocket error: {e}");
            if (statusCube != null)
            {
                statusCube.GetComponent<Renderer>().material.color = Color.red;
            }
        };

        websocket.OnClose += (e) => {
            Debug.Log($"WebSocket closed: {e}");
            isConnected = false;
            if (statusCube != null)
            {
                statusCube.GetComponent<Renderer>().material.color = Color.red;
            }
        };

        await websocket.Connect();
    }

    void ProcessHandTrackingData(string jsonData)
    {
        try
        {
            HandTrackingData data = JsonUtility.FromJson<HandTrackingData>(jsonData);
            
            if (data.type == "hand_tracking" && data.hands_detected && 
                data.hand_landmarks != null && data.hand_landmarks.Length > 0 && 
                data.hand_landmarks[0] != null && data.hand_landmarks[0].Length > 0)
            {
                isHandDetected = true;
                
                HandLandmark wrist = data.hand_landmarks[0][0];
                
                Vector3 newPosition = new Vector3(
                    (wrist.x - 0.5f) * xMultiplier,
                    (wrist.y - 0.5f) * yMultiplier,
                    (1.0f - wrist.z) * zMultiplier  // Invert Z for more intuitive movement
                );

                newPosition += initialPosition;
                
                targetPosition = newPosition;
                
                Debug.Log($"Hand detected - Raw: ({wrist.x:F3}, {wrist.y:F3}, {wrist.z:F3}) -> Target: {targetPosition}");
            }
            else
            {
                isHandDetected = false;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error processing hand tracking data: {e.Message}\nData: {jsonData}");
        }
    }

    void Update()
    {
        if (websocket != null)
        {
            #if !UNITY_WEBGL || UNITY_EDITOR
                websocket.DispatchMessageQueue();
            #endif
        }

        if (handIndicator != null && isHandDetected)
        {
            Vector3 currentPos = handIndicator.transform.position;
            Vector3 newPos = Vector3.Lerp(currentPos, targetPosition, Time.deltaTime * 15f);
            
            if (Vector3.Distance(currentPos, newPos) > 0.01f)
            {
                handIndicator.transform.position = newPos;
                Debug.Log($"Moving indicator - Current: {currentPos} -> New: {newPos}");
            }
        }
    }

    private async void OnApplicationQuit()
    {
        if (websocket != null && isConnected)
            await websocket.Close();
    }
}
