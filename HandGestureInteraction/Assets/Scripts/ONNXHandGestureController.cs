using UnityEngine;
using Unity.Barracuda;
using Intel.RealSense;
using System.Collections.Generic;
using System;

[System.Serializable]
public class PalmDetectionSettings
{
    public NNModel model;
    public string inputName = "input";  
    public string classificatorsOutputName = "classificators";
    public string regressorsOutputName = "regressors";
    public int inputWidth = 192;  
    public int inputHeight = 192;
}

[System.Serializable]
public class HandLandmarkSettings
{
    public NNModel model;
    public string inputName = "input_1";  
    public string landmarkOutputName = "ld_21_3d";
    public string flagOutputName = "output_handflag";
    public int inputWidth = 224;  
    public int inputHeight = 224; 
}

public class ONNXHandGestureController : MonoBehaviour
{
    [Header("RealSense Settings")]
    private Pipeline pipeline;
    private Config config;
    private Colorizer colorizer;
    private Texture2D inputTexture;
    private byte[] rawImagePixels;

    [Header("ONNX Model Settings")]
    [SerializeField] private PalmDetectionSettings palmDetection = new PalmDetectionSettings();
    [SerializeField] private HandLandmarkSettings handLandmark = new HandLandmarkSettings();
    private IWorker palmDetectionWorker;
    private IWorker handLandmarkWorker;

    [Header("Hand Model References")]
    public Transform handModel;
    public List<Transform> fingerJoints;

    [Header("Processing Settings")]
    public int inputWidth = 224;
    public int inputHeight = 224;
    public float confidenceThreshold = 0.5f;

    [Header("Debug Visualization")]
    public bool showCameraFeed = true;
    public float displayScale = 0.5f;
    private Texture2D debugTexture;
    private Rect? debugPalmDetection = null;
    private float debugPalmConfidence = 0f;

    private void Start()
    {
        InitializeRealSense();
        InitializeONNXModels();
    }

    private void InitializeRealSense()
    {
        try
        {
            pipeline = new Pipeline();
            config = new Config();
            colorizer = new Colorizer();

            config.EnableStream(Stream.Color, 640, 480, Format.Rgb8, 30);
            config.EnableStream(Stream.Depth, 640, 480, Format.Z16, 30);

            Debug.Log("Starting RealSense pipeline...");
            pipeline.Start(config);
            Debug.Log("RealSense pipeline started successfully!");

            inputTexture = new Texture2D(640, 480, TextureFormat.RGB24, false);
            rawImagePixels = new byte[640 * 480 * 3];

            Debug.Log("Camera initialized with resolution: 640x480");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to initialize RealSense: {e.Message}");
            Debug.LogException(e);
        }
    }


    private void InitializeONNXModels()
    {
        try
        {
            if (palmDetection.model == null || handLandmark.model == null)
            {
                Debug.LogError("Models not assigned in inspector!");
                return;
            }

            var palmRuntime = ModelLoader.Load(palmDetection.model);
            var handRuntime = ModelLoader.Load(handLandmark.model);
            
            palmDetectionWorker = WorkerFactory.CreateWorker(palmRuntime);
            handLandmarkWorker = WorkerFactory.CreateWorker(handRuntime);

            Debug.Log("=== Palm Detection Model ===");
            Debug.Log("Expected input shape: [1, 3, 192, 192]");
            Debug.Log("Model inputs: " + string.Join(", ", palmRuntime.inputs));
            Debug.Log("Model outputs: " + string.Join(", ", palmRuntime.outputs));
            
            Debug.Log("\n=== Hand Landmark Model ===");
            Debug.Log("Expected input shape: [1, 3, 224, 224]");
            Debug.Log("Model inputs: " + string.Join(", ", handRuntime.inputs));
            Debug.Log("Model outputs: " + string.Join(", ", handRuntime.outputs));

            Debug.Log("ONNX models initialized successfully");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to initialize ONNX models: {e.Message}");
            Debug.LogException(e);
        }
    }

    private void Update()
    {
        if (pipeline == null || handLandmarkWorker == null) return;  // Updated check

        try
        {
            using (var frames = pipeline.WaitForFrames())
            {
                using (var colorFrame = frames.ColorFrame)
                {
                    if (colorFrame == null)
                    {
                        Debug.LogWarning("No color frame received");
                        return;
                    }

                    ProcessFrame(colorFrame);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error processing frame: {e.Message}");
        }
    }

    private void ProcessFrame(VideoFrame colorFrame)
    {
        try
        {
            colorFrame.CopyTo(rawImagePixels);
            inputTexture.LoadRawTextureData(rawImagePixels);
            inputTexture.Apply();

            using (var palmInput = PreprocessImage(inputTexture, true))
            {
                if (palmInput == null)
                {
                    Debug.LogError("Failed to preprocess image for palm detection");
                    return;
                }

                palmDetectionWorker.Execute(palmInput);
                var palmDetections = ProcessPalmDetections(palmDetectionWorker);
                Debug.Log($"Detected {palmDetections.Count} palms");

                if (palmDetections.Count > 0)
                {
                    using (var handInput = PreprocessImage(inputTexture, false))
                    {
                        if (handInput == null)
                        {
                            Debug.LogError("Failed to preprocess image for hand landmarks");
                            return;
                        }

                        handLandmarkWorker.Execute(handInput);
                        var landmarksTensor = handLandmarkWorker.PeekOutput(handLandmark.landmarkOutputName);
                        var handFlagTensor = handLandmarkWorker.PeekOutput(handLandmark.flagOutputName);
                        
                        if (handFlagTensor != null && landmarksTensor != null)
                        {
                            float flag = handFlagTensor.AsFloats()[0];
                            if (flag > confidenceThreshold)
                            {
                                ProcessHandLandmarks(landmarksTensor);
                            }
                        }
                    }
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error in ProcessFrame: {e.Message}\n{e.StackTrace}");
        }
    }

    private List<Rect> ProcessPalmDetections(IWorker worker)
    {
        var detections = new List<Rect>();
        try
        {
            var classificators = worker.PeekOutput(palmDetection.classificatorsOutputName);
            var regressors = worker.PeekOutput(palmDetection.regressorsOutputName);
            
            if (classificators == null || regressors == null) return detections;

            float[] scores = classificators.AsFloats();
            float[] boxes = regressors.AsFloats();
            
            for (int i = 0; i < scores.Length; i++)
            {
                if (scores[i] > confidenceThreshold)
                {
                    int boxIdx = i * 4;
                    float x = boxes[boxIdx];
                    float y = boxes[boxIdx + 1];
                    float w = boxes[boxIdx + 2];
                    float h = boxes[boxIdx + 3];
                    
                    detections.Add(new Rect(x, y, w, h));
                    debugPalmDetection = new Rect(x, y, w, h);
                    debugPalmConfidence = scores[i];
                    
                    Debug.Log($"Palm detected: Score={scores[i]:F2} at ({x:F2}, {y:F2}) size={w:F2}x{h:F2}");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error processing palm detections: {e.Message}");
        }
        return detections;
    }

    private Tensor PreprocessImage(Texture2D texture, bool forPalmDetection)
    {
        try 
        {
            int targetWidth = forPalmDetection ? palmDetection.inputWidth : handLandmark.inputWidth;
            int targetHeight = forPalmDetection ? palmDetection.inputHeight : handLandmark.inputHeight;

            RenderTexture rt = RenderTexture.GetTemporary(targetWidth, targetHeight, 0, RenderTextureFormat.ARGB32);
            rt.filterMode = FilterMode.Bilinear;
            Graphics.Blit(texture, rt);
            
            Tensor tensor = new Tensor(1, targetHeight, targetWidth, 3);

            RenderTexture.active = rt;
            Texture2D resizedTexture = new Texture2D(targetWidth, targetHeight, TextureFormat.RGB24, false);
            resizedTexture.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
            resizedTexture.Apply();

            Color32[] pixels = resizedTexture.GetPixels32();
            float[] tensorData = tensor.data.Download(tensor.shape);
            
            for (int i = 0; i < pixels.Length; i++)
            {
                int baseIdx = i * 3;
                Color32 color = pixels[i];

                tensorData[baseIdx + 0] = (color.r / 255f - 0.5f) * 2.0f;
                tensorData[baseIdx + 1] = (color.g / 255f - 0.5f) * 2.0f;
                tensorData[baseIdx + 2] = (color.b / 255f - 0.5f) * 2.0f;
            }
            
            tensor.data.Upload(tensorData, tensor.shape);
            
            RenderTexture.ReleaseTemporary(rt);
            Destroy(resizedTexture);
            
            return tensor;
        }
        catch (Exception e)
        {
            Debug.LogError($"Error in PreprocessImage: {e.Message}\n{e.StackTrace}");
            return null;
        }
    }

    private void ProcessHandLandmarks(Tensor output)
    {
        float[] landmarks = output.AsFloats();

        for (int i = 0; i < fingerJoints.Count; i++)
        {
            if (i * 3 + 2 >= landmarks.Length) break;

            Vector3 position = new Vector3(
                landmarks[i * 3],
                landmarks[i * 3 + 1],
                landmarks[i * 3 + 2]
            );

            if (fingerJoints[i] != null)
            {
                position = ConvertToUnitySpace(position);
                fingerJoints[i].localPosition = position;
                
                if (i > 0 && i < fingerJoints.Count - 1)
                {
                    Vector3 toNext = fingerJoints[i + 1].position - fingerJoints[i].position;
                    Vector3 toPrev = fingerJoints[i - 1].position - fingerJoints[i].position;
                    Quaternion rotation = Quaternion.LookRotation(toNext, Vector3.Cross(toNext, toPrev));
                    fingerJoints[i].rotation = rotation;
                }
            }
        }
    }

    private Vector3 ConvertToUnitySpace(Vector3 position)
    {
        return new Vector3(
            -position.x,
            position.y,
            position.z
        );
    }

    private void OnDestroy()
    {
        pipeline?.Stop();
        pipeline?.Dispose();
        colorizer?.Dispose();
        config?.Dispose();
        palmDetectionWorker?.Dispose();
        handLandmarkWorker?.Dispose();
    }

    private void OnGUI()
    {
        if (showCameraFeed && inputTexture != null)
        {
            float width = Screen.width * displayScale;
            float height = width * (inputTexture.height / (float)inputTexture.width);
            GUI.DrawTexture(new Rect(0, 0, width, height), inputTexture);

            GUI.Label(new Rect(10, height + 10, 300, 20), 
                $"Camera: {(pipeline != null ? "Connected" : "Not Connected")}");
            GUI.Label(new Rect(10, height + 30, 300, 20), 
                $"Resolution: {inputTexture.width}x{inputTexture.height}");
            GUI.Label(new Rect(10, height + 50, 300, 20), 
                $"Frame Processing: Active");
        }
    }

    private Color[] FlipTextureVertically(Color[] pixels, int width, int height)
    {
        Color[] flipped = new Color[pixels.Length];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                flipped[(height - 1 - y) * width + x] = pixels[y * width + x];
            }
        }
        return flipped;
    }

    private void DrawDebugInfo()
    {
        if (!showCameraFeed || inputTexture == null) return;

        float width = Screen.width * displayScale;
        float height = width * (inputTexture.height / (float)inputTexture.width);

        GUI.DrawTexture(new Rect(0, 0, width, height), inputTexture);

        if (debugPalmDetection.HasValue)
        {
            Rect scaledRect = new Rect(
                debugPalmDetection.Value.x * width,
                debugPalmDetection.Value.y * height,
                debugPalmDetection.Value.width * width,
                debugPalmDetection.Value.height * height
            );

            DrawRect(scaledRect, Color.green, 2);
            string confidenceText = $"Confidence: {debugPalmConfidence:F2}";
            GUI.Label(new Rect(scaledRect.x, scaledRect.y - 20, 200, 20), confidenceText);
        }

        float y = height + 10;
        GUI.Label(new Rect(10, y, 300, 20), 
            $"Camera: {(pipeline != null ? "Connected" : "Not Connected")}");
        GUI.Label(new Rect(10, y + 20, 300, 20), 
            $"Palm Detection: {(palmDetectionWorker != null ? "Running" : "Not Running")}");
        GUI.Label(new Rect(10, y + 40, 300, 20), 
            $"Hand Landmark: {(handLandmarkWorker != null ? "Running" : "Not Running")}");
        
        if (fingerJoints != null && fingerJoints.Count > 0)
        {
            DrawLandmarks(width, height);
        }
    }

    private void DrawRect(Rect position, Color color, float thickness)
    {
        Texture2D tex = new Texture2D(1, 1);
        tex.SetPixel(0, 0, color);
        tex.Apply();

        GUI.DrawTexture(new Rect(position.x, position.y, position.width, thickness), tex);
        GUI.DrawTexture(new Rect(position.x, position.y, thickness, position.height), tex);
        GUI.DrawTexture(new Rect(position.x + position.width - thickness, position.y, thickness, position.height), tex);
        GUI.DrawTexture(new Rect(position.x, position.y + position.height - thickness, position.width, thickness), tex);

        Destroy(tex);
    }

    private void DrawLandmarks(float width, float height)
    {
        for (int i = 0; i < fingerJoints.Count; i++)
        {
            if (fingerJoints[i] != null)
            {
                Vector3 screenPos = Camera.main.WorldToScreenPoint(fingerJoints[i].position);
                if (screenPos.z > 0) 
                {
                    float x = screenPos.x * (width / Screen.width);
                    float y = height - (screenPos.y * (height / Screen.height));
                    DrawPoint(new Vector2(x, y), Color.yellow, 5);
                }
            }
        }
    }

    private void DrawPoint(Vector2 position, Color color, float size)
    {
        Texture2D tex = new Texture2D(1, 1);
        tex.SetPixel(0, 0, color);
        tex.Apply();
        GUI.DrawTexture(new Rect(position.x - size/2, position.y - size/2, size, size), tex);
        Destroy(tex);
    }

}
