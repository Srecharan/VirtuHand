using UnityEngine;
using NativeWebSocket;
using System;

[System.Serializable]
public class HandPosition
{
    public float x;
    public float y;
    public float z;
}

[System.Serializable]
public class HandTrackingMessage
{
    public string type;
    public bool hands_detected;
    public HandPosition hand_position;
}

public class DebugTestClient : MonoBehaviour
{
    WebSocket websocket;
    bool isConnected = false;
    
    public GameObject handIndicator;
    private bool isHandDetected = false;
    private float reconnectTimer = 0f;
    private bool isReconnecting = false;

    // Movement parameters
    [Header("Movement Settings")]
    public float moveSpeed = 5.0f;  // Smoothing factor
    public float horizontalBound = 5.0f;  // X-axis movement bound
    public float verticalBound = 3.0f;    // Y-axis movement bound
    public float depthBound = 4.0f;       // Z-axis movement bound
    
    private Vector3 initialPosition;
    private Vector3 targetPosition;

    private readonly float RECONNECT_INTERVAL = 2f;
    
    void Start()
    {
        Debug.Log("Debug Test Client initialized");
        InitializeObject();
        ConnectToServer();
    }

    void InitializeObject()
    {
        if (handIndicator == null)
        {
            Debug.LogError("HandIndicator not assigned!");
            return;
        }

        initialPosition = handIndicator.transform.position;
        targetPosition = initialPosition;
        Debug.Log("Object initialized at position: " + initialPosition);
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
            HandTrackingMessage data = JsonUtility.FromJson<HandTrackingMessage>(message);
            
            isHandDetected = data.hands_detected;
            
            if (isHandDetected && data.hand_position != null)
            {
                // Update target position based on hand position
                // Note: Z is now being used for depth movement
                targetPosition = new Vector3(
                    initialPosition.x + (data.hand_position.x * horizontalBound),
                    initialPosition.y + (data.hand_position.y * verticalBound),
                    initialPosition.z + (data.hand_position.z * depthBound)
                );
                
                Debug.Log($"Hand position - X: {data.hand_position.x:F2}, Y: {data.hand_position.y:F2}, Z: {data.hand_position.z:F2}");
                Debug.Log($"Target position: {targetPosition}");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error processing message: {e.Message}");
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

        // Smooth movement towards target position
        if (handIndicator != null && isHandDetected)
        {
            handIndicator.transform.position = Vector3.Lerp(
                handIndicator.transform.position,
                targetPosition,
                moveSpeed * Time.deltaTime
            );
        }
    }

    private async void OnApplicationQuit()
    {
        if (websocket != null)
        {
            await websocket.Close();
        }
    }

    // Optional: Add visual debugging
    void OnDrawGizmos()
    {
        if (handIndicator != null)
        {
            // Draw movement bounds
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireCube(
                initialPosition,
                new Vector3(horizontalBound * 2, verticalBound * 2, depthBound * 2)
            );
        }
    }
}