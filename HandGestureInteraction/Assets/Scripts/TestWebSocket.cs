using UnityEngine;
using NativeWebSocket;
using System;

public class TestWebSocket : MonoBehaviour
{
    WebSocket websocket;
    bool isConnected = false;
    

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
                
                CameraStatus status = JsonUtility.FromJson<CameraStatus>(message);
                
               
                if (statusCube != null) {
                   
                    Renderer cubeRenderer = statusCube.GetComponent<Renderer>();
                    if (cubeRenderer != null) {
                       
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

[Serializable]
public class CameraStatus
{
    public string type;
    public bool camera_active;
    public int frame_width;
    public int frame_height;
}