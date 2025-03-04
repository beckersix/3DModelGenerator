# Reference Models Directory

Place your 3D model files in this directory to be used as references for training and generation.

## Supported Formats

- Wavefront OBJ (.obj)
- FBX (.fbx)
- STL (.stl)
- GLTF/GLB (.gltf, .glb)

## How it works

1. Models placed in this directory will be processed when you run the application in training mode
2. Each model will be converted to a point cloud representation
3. The system will learn from these models and use them as references for future generations

## Best practices

- Use models with clean topology
- Include a variety of shapes to improve generation quality
- Name your files descriptively (e.g., "chair_modern.obj", "table_round.fbx")
- Aim for models with 1,000-10,000 polygons for best results
