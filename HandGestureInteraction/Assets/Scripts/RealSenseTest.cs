using UnityEngine;
using Intel.RealSense;
using System.Collections.Generic;
using System;

public class RealSenseTest : MonoBehaviour
{
    private Pipeline pipeline;
    private Config config;
    private Colorizer colorizer;
    private Texture2D colorTexture;
    private Texture2D depthTexture;
    private byte[] rawColorPixels;
    private byte[] rawDepthPixels;

    // Debug info
    private float frameTimer = 0f;
    private int frameCount = 0;
    private float currentFPS = 0f;
    private bool isStreaming = false;

    [Header("Stream Configuration")]
    public int colorWidth = 640;
    public int colorHeight = 480;
    public int depthWidth = 640;
    public int depthHeight = 480;
    public int frameRate = 30;

    [Header("Display Settings")]
    public bool showColorFeed = true;
    public bool showDepthFeed = true;
    public float displayScale = 0.5f;

    void Start()
    {
        try
        {
            // Initialize RealSense pipeline
            pipeline = new Pipeline();
            config = new Config();
            colorizer = new Colorizer();

            // Configure streams
            config.EnableStream(Stream.Color, colorWidth, colorHeight, Format.Rgb8, frameRate);
            config.EnableStream(Stream.Depth, depthWidth, depthHeight, Format.Z16, frameRate);

            Debug.Log("Starting RealSense pipeline...");
            pipeline.Start(config);
            Debug.Log("RealSense pipeline started successfully!");
            isStreaming = true;

            // Initialize textures
            colorTexture = new Texture2D(colorWidth, colorHeight, TextureFormat.RGB24, false);
            depthTexture = new Texture2D(depthWidth, depthHeight, TextureFormat.RGB24, false);
            rawColorPixels = new byte[colorWidth * colorHeight * 3];
            rawDepthPixels = new byte[depthWidth * depthHeight * 3];
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to initialize RealSense: {e.Message}");
            isStreaming = false;
        }
    }

    void Update()
    {
        if (!isStreaming || pipeline == null) return;

        try
        {
            using (var frames = pipeline.WaitForFrames())
            {
                // Process color frame
                using (var colorFrame = frames.ColorFrame)
                {
                    if (colorFrame != null && showColorFeed)
                    {
                        colorFrame.CopyTo(rawColorPixels);
                        colorTexture.LoadRawTextureData(rawColorPixels);
                        colorTexture.Apply();
                    }
                }

                // Process depth frame
                using (var depthFrame = frames.DepthFrame)
                {
                    if (depthFrame != null && showDepthFeed)
                    {
                        using (var colorizedDepth = colorizer.Process<VideoFrame>(depthFrame))
                        {
                            if (colorizedDepth != null)
                            {
                                colorizedDepth.CopyTo(rawDepthPixels);
                                depthTexture.LoadRawTextureData(rawDepthPixels);
                                depthTexture.Apply();
                            }
                        }
                    }
                }

                // Update frame counter
                frameCount++;
                frameTimer += Time.deltaTime;
                if (frameTimer >= 1.0f)
                {
                    currentFPS = frameCount / frameTimer;
                    frameCount = 0;
                    frameTimer = 0f;
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error processing frame: {e.Message}");
        }
    }

    void OnGUI()
    {
        if (!isStreaming)
        {
            GUI.Label(new Rect(10, 10, 300, 20), "RealSense camera not streaming!");
            return;
        }

        float scaledWidth = colorWidth * displayScale;
        float scaledHeight = colorHeight * displayScale;
        
        // Display FPS
        GUI.Label(new Rect(10, 10, 200, 20), $"FPS: {currentFPS:F1}");

        // Display color feed
        if (showColorFeed && colorTexture != null)
        {
            GUI.DrawTexture(new Rect(10, 40, scaledWidth, scaledHeight), colorTexture);
            GUI.Label(new Rect(10, 50 + scaledHeight, 200, 20), "Color Feed");
        }

        // Display depth feed
        if (showDepthFeed && depthTexture != null)
        {
            GUI.DrawTexture(new Rect(20 + scaledWidth, 40, scaledWidth, scaledHeight), depthTexture);
            GUI.Label(new Rect(20 + scaledWidth, 50 + scaledHeight, 200, 20), "Depth Feed");
        }
    }

    void OnDestroy()
    {
        if (pipeline != null)
        {
            pipeline.Stop();
            pipeline.Dispose();
        }

        if (colorizer != null)
        {
            colorizer.Dispose();
        }

        if (config != null)
        {
            config.Dispose();
        }
    }
}