using UnityEngine;
using Unity.Barracuda;
using System.IO;

public class NeuralNetworkCreator : MonoBehaviour
{
    void Start()
    {
        // Define paths
        string palmModelPath = Path.Combine(Application.dataPath, "Models/palm_detection_full_inf_post_192x192.onnx");
        string handModelPath = Path.Combine(Application.dataPath, "Models/hand_landmark_sparse_Nx3x224x224.onnx");
        
        // Read the ONNX files as bytes
        byte[] palmModelBytes = File.ReadAllBytes(palmModelPath);
        byte[] handModelBytes = File.ReadAllBytes(handModelPath);

        // Create the NNModel assets
        string palmAssetPath = "Assets/NeuralNetworks/PalmDetectionNet.asset";
        string landmarkAssetPath = "Assets/NeuralNetworks/HandLandmarkNet.asset";

#if UNITY_EDITOR
        // Create the NNModel instances
        var palmNNModel = ScriptableObject.CreateInstance<NNModel>();
        var palmModelData = ScriptableObject.CreateInstance<NNModelData>();
        palmModelData.Value = palmModelBytes;
        palmNNModel.modelData = palmModelData;

        var landmarkNNModel = ScriptableObject.CreateInstance<NNModel>();
        var landmarkModelData = ScriptableObject.CreateInstance<NNModelData>();
        landmarkModelData.Value = handModelBytes;
        landmarkNNModel.modelData = landmarkModelData;

        // Create directories if they don't exist
        string dir = Path.GetDirectoryName(palmAssetPath);
        if (!Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }

        // Save the assets
        UnityEditor.AssetDatabase.CreateAsset(palmModelData, "Assets/NeuralNetworks/PalmDetectionData.asset");
        UnityEditor.AssetDatabase.CreateAsset(palmNNModel, palmAssetPath);
        UnityEditor.AssetDatabase.CreateAsset(landmarkModelData, "Assets/NeuralNetworks/HandLandmarkData.asset");
        UnityEditor.AssetDatabase.CreateAsset(landmarkNNModel, landmarkAssetPath);
        UnityEditor.AssetDatabase.SaveAssets();
        UnityEditor.AssetDatabase.Refresh();
        
        Debug.Log("Neural Network assets created successfully!");
#endif
    }
}
