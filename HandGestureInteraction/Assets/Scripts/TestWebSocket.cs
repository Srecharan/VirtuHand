using UnityEngine;
using NativeWebSocket;
using System;

public class TestWebSocket : MonoBehaviour
{
    WebSocket websocket;
    bool isConnected = false;
    
    // Reference to your cube that we'll use as a visual indicator
    public GameObject statusCube;

    async void Start()
    {
        Debug.Log("Starting camera test client...");
        websocket = new WebSocket("ws://127.0.0.1:8765");
        
        websocket.OnOpen += () => {
            Debug.Log("Connected to camera server!");
            isConnected = true;
        };

        websocket.OnMessage += (bytes) => {
            var message = System.Text.Encoding.UTF8.GetString(bytes);
            Debug.Log("Received camera data: " + message);
            
            try {
                // Parse the JSON message into our CameraStatus class
                CameraStatus status = JsonUtility.FromJson<CameraStatus>(message);
                
                // Update our visual indicator if we have one
                if (statusCube != null) {
                    // Get the cube's renderer and change its color
                    Renderer cubeRenderer = statusCube.GetComponent<Renderer>();
                    if (cubeRenderer != null) {
                        // Set to green if camera is active, red if not
                        cubeRenderer.material.color = status.camera_active ? Color.green : Color.red;
                    }
                }
                else {
                    Debug.LogWarning("Status cube not assigned!");
                }
            }
            catch (Exception e) {
                Debug.LogError($"Error processing message: {e.Message}\nMessage received: {message}");
            }
        };

        websocket.OnError += (e) => {
            Debug.LogError($"WebSocket error: {e}");
        };

        websocket.OnClose += (e) => {
            Debug.Log($"WebSocket closed: {e}");
            isConnected = false;
        };

        Debug.Log("Attempting to connect to server...");
        await websocket.Connect();
    }

    void Update()
    {
        if (websocket != null) {
            #if !UNITY_WEBGL || UNITY_EDITOR
                websocket.DispatchMessageQueue();
            #endif
        }
    }

    private async void OnApplicationQuit()
    {
        if (websocket != null && isConnected)
            await websocket.Close();
    }
}

// This class needs to match the structure of our JSON message from Python
[Serializable]
public class CameraStatus
{
    public string type;
    public bool camera_active;
    public int frame_width;
    public int frame_height;
}