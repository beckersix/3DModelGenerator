# 3D Model Generator

A powerful AI-based tool for generating, classifying, and manipulating 3D shapes using natural language commands.

## Installation Requirements

To use the 3D Model Generator, you'll need to install the following dependencies:

```bash
pip install torch numpy matplotlib trimesh tqdm
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

1. Run the application in interactive mode:
   ```
   python main-app.py
   ```
2. Enter natural language commands to generate shapes, such as:
   - "Create a cube"
   - "Generate a tall cylindrical tower"
   - "Make a hollow sphere with radius 5"

## Directory Structure

The application uses a standardized directory structure:

```
3DModelGenerator/
├── models/                   # Trained models
│   ├── classifier.pth        # Trained classifier model
│   ├── generator.pth         # Trained generator model
│   ├── tokenizer.json        # Tokenizer vocabulary
│   └── shape_types.json      # Available shape types
│
├── reference_models/         # Place your 3D model files here for training
│   └── *.obj, *.fbx, etc.    # Your 3D models for reference
│
├── training_data/            # Training data for models
│   ├── pointcloud_*.npy      # Individual point cloud files
│   └── metadata.json         # Training data metadata
│
├── data/                     # Processed data
│   └── *.npy                 # Processed point clouds
│
├── main-app.py               # Main application entry point
├── models.py                 # Neural network model definitions
├── shape_generation.py       # Shape generation functions
├── data_utils.py             # Dataset and data handling utilities
├── shape_ai.py               # Main AI engine combining components
├── command_interpreter.py    # Natural language command processor
├── point_cloud_processor.py  # Point cloud processing utilities
└── fbx_loader.py             # 3D mesh model loader
```

## Training with Your Own Models

To train the system with your own 3D models:

1. Place your 3D model files (.obj, .fbx, .stl, etc.) in the `reference_models/` directory
2. Run the application in training mode:
   ```
   python main-app.py --mode train
   ```
3. The system will:
   - Convert your 3D models to point clouds
   - Train the classifier and generator models
   - Save the trained models to the `models/` directory

## Model Generation

The system generates 3D models in these ways:

1. **Basic Shape Generation**: Creates geometric primitives based on parameters from your text
2. **AI-Based Generation**: Uses trained models to generate complex shapes based on descriptions
3. **Adaptive Generation**: Creates variations of existing models by adapting them to your specifications

## Troubleshooting

- **No point cloud generated**: Make sure your shape description contains parameters or a recognized shape type
- **Loading failures**: Some complex models may not load correctly; try a different format if available
- **Memory issues**: Very large models may require reducing the number of points 

## Adding More Training Data

For best results when generating models:

- Add diverse 3D models to the `reference_models/` directory
- Include models of various categories (furniture, vehicles, architecture, etc.)
- Use models with clean topology and reasonable polygon counts
- Name files descriptively to help with classification