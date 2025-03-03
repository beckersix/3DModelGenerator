# 3D Mesh Model Loader for Point Cloud Conversion

## Installation Requirements

To use the 3D mesh model loader, you'll need to install the following dependencies:

```bash
pip install trimesh numpy matplotlib
```

For additional mesh format support, install these optional dependencies:

```bash
# For FBX support
pip install pyrender

# For GLTF/GLB support
pip install pyglet

# For additional importers
pip install meshio
```

## Supported Formats

The mesh loader supports the following 3D model formats:
- OBJ (.obj)
- FBX (.fbx) - requires pyrender
- STL (.stl)
- PLY (.ply)
- GLTF/GLB (.gltf, .glb) - requires pyglet
- And more formats supported by trimesh

## Usage

1. Place your 3D model files in a directory
2. Run the application in interactive mode:
   ```
   python main.py
   ```
3. Choose option 5: "Load 3D mesh models and convert to point clouds"
4. Follow the prompts to specify:
   - Directory containing your 3D models
   - Number of points to sample per model
   - Sampling method (surface or vertex)
   - Noise level
   - Output directory for saving point clouds

## How It Works

The loader operates in these steps:
1. Scans a directory for 3D model files
2. Loads each file using the trimesh library
3. Converts meshes to point clouds using either:
   - Surface sampling: samples points on the model's surface
   - Vertex sampling: uses the model's vertices directly
4. Normalizes coordinates and applies an optional noise level
5. Allows you to preview and save the resulting point clouds


# 3D Shape AI Project Directory Structure

## Directory Structure

The application now uses a standardized directory structure for storing models and data:

```
3DModelGenerator/
├── models/                   # Default directory for trained models
│   ├── classifier.pth        # Trained classifier model
│   ├── generator.pth         # Trained generator model
│   ├── tokenizer.json        # Tokenizer vocabulary
│   └── shape_types.json      # Available shape types
│
├── training_data/            # Training data for models
│   ├── pointcloud_*.npy      # Individual point cloud files
│   └── metadata.json         # Training data metadata
│
├── converted_models/         # Default directory for converted 3D models
│   └── *.npy                 # Converted point clouds
│
├── main.py                   # Main application entry point
├── models.py                 # Neural network model definitions
├── shape_generation.py       # Shape generation functions
├── data_utils.py             # Dataset and data handling utilities
├── shape_ai.py               # Main API combining components
├── training.py               # Training functions
└── fbx_loader.py             # 3D mesh model loader
```

## Automatic Directory Management

The application now:

1. Automatically creates these directories if they don't exist
2. Saves models to the default `models/` directory unless specified otherwise
3. Uses consistent paths for loading models in interactive mode
4. Saves converted 3D models to `converted_models/` by default

## Command-Line Options

You can still specify custom directories when needed:

```bash
# Use custom directories
python main.py --mode train --data_dir custom_data --model_dir custom_models

# Use default directories
python main.py --mode train
```

## Locating Your Files

- **Trained Models**: Located in the `models/` directory
- **Converted 3D Models**: Located in the `converted_models/` directory
- **Training Data**: Located in the `training_data/` directory

All paths are relative to the application's main directory, making it easy to find your files regardless of where you run the application from.

## Troubleshooting

- **Import errors**: Make sure you have installed trimesh and the appropriate format-specific dependencies
- **Loading failures**: Some complex models may not load correctly; try a different format if available
- **Memory issues**: Very large models may require reducing the number of points