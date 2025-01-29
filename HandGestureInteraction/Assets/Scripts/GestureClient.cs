using UnityEngine;
using System;
using NativeWebSocket;
using System.Threading.Tasks;

[Serializable]
public class GestureData
{
    public string gesture_type;
    public float confidence;
    public string hand_side;
    public PalmPosition palm_position;
}

[Serializable]
public class PalmPosition
{
    public float x;
    public float y;
    public float z;
}

public class GestureClient : MonoBehaviour
{
    private WebSocket websocket;
    private readonly string serverUrl = "ws://127.0.0.1:8765";
    private bool isConnected = false;
    private const int MAX_RECONNECT_ATTEMPTS = 3;  
    private int reconnectAttempts = 0;            
    public GameObject grabCube;
    public GameObject pinchSphere;

    void Awake()
    {
        Debug.Log($"Initial Object References - GrabCube: {grabCube != null}, PinchSphere: {pinchSphere != null}");

        if (grabCube == null)
        {
            Debug.LogError("GrabCube reference is missing! Please assign it in the Inspector.");
            grabCube = GameObject.Find("GrabCube");
            if (grabCube != null)
                Debug.Log("Found GrabCube automatically!");
        }

        if (pinchSphere == null)
        {
            Debug.LogError("PinchSphere reference is missing! Please assign it in the Inspector.");
            pinchSphere = GameObject.Find("PinchSphere");
            if (pinchSphere != null)
                Debug.Log("Found PinchSphere automatically!");
        }
    }

    async void Start()
    {
        Debug.Log("Starting WebSocket client...");
        await ConnectToServer();
    }

    async Task ConnectToServer()
    {
        Debug.Log("Starting connection process...");
        while (reconnectAttempts < MAX_RECONNECT_ATTEMPTS && !isConnected)
        {
            try
            {
                Debug.Log($"Attempt {reconnectAttempts + 1} to connect to {serverUrl}...");
                
                if (websocket != null)
                {
                    await websocket.Close();
                    websocket = null;
                }
                
                websocket = new WebSocket(serverUrl);
                
                websocket.OnOpen += () =>
                {
                    isConnected = true;
                    Debug.Log("WebSocket connection established!");
                    
                    // Send a test message
                    var testMessage = System.Text.Encoding.UTF8.GetBytes("Unity client connected");
                    websocket.Send(testMessage);
                };

                websocket.OnMessage += (bytes) =>
                {
                    var message = System.Text.Encoding.UTF8.GetString(bytes);
                    Debug.Log($"Received message: {message}");
                    ProcessGestureData(message);
                };

                websocket.OnError += (e) =>
                {
                    Debug.LogError($"WebSocket Error: {e}");
                };

                websocket.OnClose += (e) =>
                {
                    isConnected = false;
                    Debug.Log($"Connection closed: {e}");
                };

                await websocket.Connect();
                await Task.Delay(2000);  // Wait longer for connection to establish
                
                if (isConnected)
                {
                    Debug.Log("Successfully connected to gesture server!");
                    break;
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"Connection attempt {reconnectAttempts + 1} failed: {e.Message}");
                await Task.Delay(1000);
            }
            
            reconnectAttempts++;
        }

        if (!isConnected)
        {
            Debug.LogError("Failed to connect after maximum attempts. Please check if the Python server is running.");
        }
    }


    void ProcessGestureData(string jsonData)
    {
        try
        {
            GestureData gestureData = JsonUtility.FromJson<GestureData>(jsonData);
            Debug.Log($"Received gesture: {gestureData.gesture_type} with confidence: {gestureData.confidence}");

            switch (gestureData.gesture_type)
            {
                case "OPEN_PALM":
                    HandleOpenPalm(gestureData);
                    break;
                case "PINCH":
                    HandlePinch(gestureData);
                    break;
                case "GRAB":
                    HandleGrab(gestureData);
                    break;
                case "POINT":
                    HandlePoint(gestureData);
                    break;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error processing gesture data: {e.Message}");
        }
    }

    private void HandleOpenPalm(GestureData data)
    {
        Debug.Log($"Open Palm detected! Hand: {data.hand_side}, Confidence: {data.confidence}");
    }

    private void HandlePinch(GestureData data)
    {
        if (pinchSphere != null)
        {
            float scale = Mathf.Abs(data.palm_position.z) * 0.1f;
            pinchSphere.transform.localScale = new Vector3(scale, scale, scale);
        }
    }

    private void HandleGrab(GestureData data)
    {
        if (grabCube != null)
        {
            Vector3 newPosition = new Vector3(
                data.palm_position.x * 10f,  
                data.palm_position.y * 10f + 2f,  
                Mathf.Abs(data.palm_position.z) * 10f + 2f  
            );

            Debug.Log($"Hand Position: ({data.palm_position.x}, {data.palm_position.y}, {data.palm_position.z})");
            Debug.Log($"Moving cube to: {newPosition}");

            grabCube.transform.position = Vector3.Lerp(
                grabCube.transform.position,
                newPosition,
                Time.deltaTime * 10f 
            );    
        }
        else
        {
            Debug.LogWarning("GrabCube reference is missing!");
        }
    }

    private void HandlePoint(GestureData data)
    {
        Debug.Log($"Point detected! Hand: {data.hand_side}, Confidence: {data.confidence}");
    }

    private async void OnApplicationQuit()
    {
        if (websocket != null && isConnected)
            await websocket.Close();
    }

    private void Update()
    {
        #if !UNITY_WEBGL || UNITY_EDITOR
            if (websocket != null)
                websocket.DispatchMessageQueue();
        #endif
    }
}
