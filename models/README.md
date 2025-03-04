# Model Directory

This directory contains trained models for the 3D Model Generator.

## Contents

- `classifier.pth`: The model for classifying point clouds
- `generator.pth`: The model for generating point clouds from text descriptions
- `tokenizer.json`: Tokenizer vocabulary for text processing
- `shape_types.json`: List of supported shape types

## Adding Reference Models

To upload real models for reference, add your 3D model files (.obj, .fbx, .stl, etc.) to the `reference_models` directory. 
These will be used for training and as references for generation.

Preferred formats:
- Wavefront OBJ (.obj)
- FBX (.fbx)
- STL (.stl)
- GLTF/GLB (.gltf, .glb)

The system will automatically convert these formats to point clouds during training.
