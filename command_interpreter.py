"""
Simple natural language command interpreter for 3D Shape AI.
"""
import re
import random
import torch

class CommandInterpreter:
    def __init__(self):
        """Initialize the command interpreter with pattern matching rules."""
        # Define shape types and their synonyms
        self.shape_types = {
            "cube": ["cube", "box", "square", "block", "dice", "cubic"],
            "rectangular_prism": ["rectangular prism", "rectangle", "brick", "rectangular box", "cuboid", "block"],
            "sphere": ["sphere", "ball", "orb", "globe", "round", "spherical"],
            "cylinder": ["cylinder", "tube", "pipe", "cylindrical", "can", "drum"],
            "pyramid": ["pyramid", "triangular pyramid", "tetrahedron", "triangular", "egyptian"],
            "torus": ["torus", "donut", "ring", "doughnut", "hoop", "toroid", "circular"],
            "cone": ["cone", "funnel", "conical", "ice cream cone", "triangular", "pointed"],
            "ellipsoid": ["ellipsoid", "egg", "oval", "elliptical", "elongated sphere"],
            "capsule": ["capsule", "pill", "tablet", "medicine", "rounded cylinder"],
            "star": ["star", "starburst", "stellated", "starfish", "asterisk", "five-pointed"],
            "helix": ["helix", "spiral", "coil", "spring", "corkscrew", "dna"]
        }
        
        # Define properties and their synonyms
        self.properties = {
            "size": ["small", "tiny", "little", "medium", "average", "large", "big", "huge", "enormous", "gigantic"],
            "width": ["narrow", "thin", "wide", "broad", "thick"],
            "height": ["short", "tall", "low", "high"],
            "depth": ["shallow", "deep"],
            "rotation": ["rotated", "tilted", "angled", "straight", "upright", "sideways"],
            "noise": ["smooth", "rough", "noisy", "perfect", "irregular", "precise", "exact", "bumpy"]
        }
        
        # Define materials and their attributes
        self.materials = [
            "wooden", "metal", "plastic", "glass", "stone", "marble", "rubber", 
            "transparent", "concrete", "cardboard", "paper", "cloth", "ceramic"
        ]
        
        # Define colors
        self.colors = [
            "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", 
            "black", "white", "gray", "silver", "gold", "bronze", "copper"
        ]
        
        # Command patterns for creating shapes
        self.creation_patterns = [
            r"(create|make|generate|give me|show me|build|produce|render)(?: a| an)? (.*)",
            r"(new|another) (.*)",
            r"i want(?: a| an)? (.*)",
            r"(?:can|could) (?:you|i) (?:have|get|make|create|see|generate)(?: a| an)? (.*)",
            r"(.*)\bshape\b"
        ]
        
        # Command patterns for classification
        self.classification_patterns = [
            r"(classify|identify|what is|what's|recognize|determine|analyze)(?: this| the)? (.*)",
            r"what( kind of| type of)? shape is (.*)"
        ]
        
        # Command patterns for model conversion
        self.conversion_patterns = [
            r"(convert|transform|change|load|process|import)(?: the| a| an)? (.*)(?: model| file| mesh| object)?",
            r"(import|load|read|open|use)(?: the| a| an)? (.*)(?: model| file| mesh)?",
            r"(scan|folder|directory|path)(?: of| with| containing)? (.*)"
        ]
        
        # Command patterns for help and information
        self.help_patterns = [
            r"(help|assist|support|guide|instructions)",
            r"(how|what) (do|to|can) (i|you) (do|use|make)",
            r"what (commands|options|features|abilities|functions) (do you|can i) (have|use)"
        ]
        
        # Command patterns for quitting
        self.quit_patterns = [
            r"(quit|exit|leave|close|end|stop|terminate)",
            r"(i('m| am) done)",
            r"(goodbye|bye|farewell|adios)"
        ]
        
        # Add adaptive generation patterns
        self.adaptive_patterns = [
            r"(?:create|generate|make)\s+(?:a\s+)?(?:new\s+)?(?:3d\s+)?model\s+(?:of\s+)?(?:a\s+)?(.+?)(?:\s+like|similar to|based on)\s+(.+)",
            r"(?:create|generate|make)\s+(?:a\s+)?(?:new\s+)?variation\s+of\s+(.+)",
            r"adapt\s+(?:the\s+)?model\s+(.+)\s+to\s+(?:create|make|generate)\s+(.+)",
            r"learn\s+from\s+(.+)\s+(?:and|to)\s+(?:create|generate|make)\s+(.+)"
        ]

    def _find_shape_type(self, text):
        """
        Identify the shape type mentioned in the text.
        
        Args:
            text: Input text to search for shape mentions
            
        Returns:
            The canonical shape type name or None if not found
        """
        text = text.lower()
        for canonical, synonyms in self.shape_types.items():
            for synonym in synonyms:
                if synonym in text:
                    return canonical
        return None
    
    def _extract_properties(self, text):
        """
        Extract size, material, color, and other properties from text.
        
        Args:
            text: Input text to search for properties
            
        Returns:
            Dictionary of extracted properties
        """
        text = text.lower()
        properties = {}
        
        # Check for materials
        for material in self.materials:
            if material in text:
                properties["material"] = material
                break
        
        # Check for colors
        for color in self.colors:
            if color in text:
                properties["color"] = color
                break
        
        # Check for sizes and other properties
        for prop, values in self.properties.items():
            for value in values:
                if value in text:
                    properties[prop] = value
                    break
        
        # Check for numeric parameters
        # Size
        size_match = re.search(r"size(?:\s*=\s*|\s+of\s+|\s+is\s+|\s+)(\d+\.?\d*)", text)
        if size_match:
            properties["size"] = float(size_match.group(1))
        
        # Width
        width_match = re.search(r"width(?:\s*=\s*|\s+of\s+|\s+is\s+|\s+)(\d+\.?\d*)", text)
        if width_match:
            properties["width"] = float(width_match.group(1))
        
        # Height
        height_match = re.search(r"height(?:\s*=\s*|\s+of\s+|\s+is\s+|\s+)(\d+\.?\d*)", text)
        if height_match:
            properties["height"] = float(height_match.group(1))
        
        # Radius
        radius_match = re.search(r"radius(?:\s*=\s*|\s+of\s+|\s+is\s+|\s+)(\d+\.?\d*)", text)
        if radius_match:
            properties["radius"] = float(radius_match.group(1))
        
        return properties
    
    def _extract_numeric_params(self, text, shape_type):
        """
        Extract numeric parameters specific to a shape type.
        
        Args:
            text: Input text to search for parameters
            shape_type: The type of shape to get parameters for
            
        Returns:
            Dictionary of parameter names and values
        """
        params = {}
        text = text.lower()
        
        # Common parameters
        size_match = re.search(r"size(?:\s*=\s*|\s+of\s+|\s+is\s+|\s+)(\d+\.?\d*)", text)
        if size_match:
            params["size"] = float(size_match.group(1))
        
        # Process based on shape type
        if shape_type == "cube":
            # Cube has a single size parameter
            if "size" not in params:
                # Look for size descriptions and convert to approximate values
                if "small" in text or "tiny" in text:
                    params["size"] = 0.5
                elif "large" in text or "big" in text:
                    params["size"] = 1.5
                else:
                    params["size"] = 1.0
        
        elif shape_type == "rectangular_prism":
            # Check for specific dimensions
            width_match = re.search(r"width(?:\s*=\s*|\s+of\s+|\s+is\s+|\s+)(\d+\.?\d*)", text)
            height_match = re.search(r"height(?:\s*=\s*|\s+of\s+|\s+is\s+|\s+)(\d+\.?\d*)", text)
            depth_match = re.search(r"depth(?:\s*=\s*|\s+of\s+|\s+is\s+|\s+)(\d+\.?\d*)", text)
            
            if width_match:
                params["size_x"] = float(width_match.group(1))
            elif "wide" in text:
                params["size_x"] = 2.0
            elif "narrow" in text:
                params["size_x"] = 0.7
                
            if height_match:
                params["size_y"] = float(height_match.group(1))
            elif "tall" in text:
                params["size_y"] = 2.0
            elif "short" in text:
                params["size_y"] = 0.7
                
            if depth_match:
                params["size_z"] = float(depth_match.group(1))
            elif "deep" in text:
                params["size_z"] = 2.0
            elif "shallow" in text:
                params["size_z"] = 0.5
        
        elif shape_type in ["sphere", "cylinder", "cone", "capsule"]:
            # Check for radius
            radius_match = re.search(r"radius(?:\s*=\s*|\s+of\s+|\s+is\s+|\s+)(\d+\.?\d*)", text)
            if radius_match:
                params["radius"] = float(radius_match.group(1))
            elif "small" in text or "tiny" in text:
                params["radius"] = 0.5
            elif "large" in text or "big" in text:
                params["radius"] = 1.5
            
            # For cylinder, cone, and capsule, also check height
            if shape_type in ["cylinder", "cone", "capsule"]:
                height_match = re.search(r"height(?:\s*=\s*|\s+of\s+|\s+is\s+|\s+)(\d+\.?\d*)", text)
                if height_match:
                    params["height"] = float(height_match.group(1))
                elif "tall" in text:
                    params["height"] = 2.5
                elif "short" in text:
                    params["height"] = 1.0
        
        elif shape_type == "torus":
            # Check for major and minor radius
            major_match = re.search(r"major[_\s]radius(?:\s*=\s*|\s+of\s+|\s+is\s+|\s+)(\d+\.?\d*)", text)
            minor_match = re.search(r"minor[_\s]radius(?:\s*=\s*|\s+of\s+|\s+is\s+|\s+)(\d+\.?\d*)", text)
            
            if major_match:
                params["major_radius"] = float(major_match.group(1))
            elif "large" in text:
                params["major_radius"] = 1.5
            
            if minor_match:
                params["minor_radius"] = float(minor_match.group(1))
            elif "thick" in text:
                params["minor_radius"] = 0.5
            elif "thin" in text:
                params["minor_radius"] = 0.2
        
        # Add noise parameter
        if "smooth" in text:
            params["noise"] = 0.01
        elif "rough" in text or "bumpy" in text:
            params["noise"] = 0.1
        
        return params
    
    def _convert_to_shape_parameters(self, shape_type, properties):
        """
        Convert general properties to shape-specific parameters.
        
        Args:
            shape_type: The type of shape
            properties: Dictionary of properties
            
        Returns:
            Dictionary of parameters for the specific shape
        """
        params = {"num_points": 1024}
        
        # Copy any numeric parameters already present
        for key in ["size", "radius", "height", "width", "major_radius", "minor_radius"]:
            if key in properties and isinstance(properties[key], (int, float)):
                params[key] = properties[key]
        
        # Convert size descriptions to parameters
        if "size" in properties and isinstance(properties["size"], str):
            size_map = {
                "tiny": 0.3,
                "small": 0.7,
                "medium": 1.0,
                "large": 1.5,
                "huge": 2.0,
                "enormous": 2.5,
                "gigantic": 3.0
            }
            if properties["size"] in size_map:
                if shape_type in ["cube", "pyramid"]:
                    params["size"] = size_map[properties["size"]]
                elif shape_type in ["sphere", "cylinder", "cone", "capsule"]:
                    params["radius"] = size_map[properties["size"]] / 2
                elif shape_type == "torus":
                    params["major_radius"] = size_map[properties["size"]] / 2
        
        # Handle other properties
        if "height" in properties and isinstance(properties["height"], str):
            height_map = {
                "short": 0.7,
                "tall": 2.0,
                "low": 0.5,
                "high": 2.5
            }
            if properties["height"] in height_map and shape_type in ["cylinder", "cone", "capsule", "pyramid"]:
                params["height"] = height_map[properties["height"]]
        
        if "width" in properties and isinstance(properties["width"], str):
            width_map = {
                "narrow": 0.7,
                "thin": 0.5,
                "wide": 1.8,
                "broad": 2.0,
                "thick": 1.5
            }
            if properties["width"] in width_map:
                if shape_type == "rectangular_prism":
                    params["size_x"] = width_map[properties["width"]]
                elif shape_type == "torus":
                    params["minor_radius"] = width_map[properties["width"]] / 4
        
        # Set noise level based on descriptors
        if "noise" in properties:
            if properties["noise"] == "smooth" or properties["noise"] == "perfect":
                params["noise"] = 0.01
            elif properties["noise"] == "rough" or properties["noise"] == "irregular":
                params["noise"] = 0.08
            elif properties["noise"] == "bumpy":
                params["noise"] = 0.12
        
        return params
    
    def interpret_command(self, text):
        """
        Interpret a natural language command and convert it to an action.
        
        Args:
            text: User's text input
            
        Returns:
            Dict with command type and parameters
        """
        text = text.lower().strip()
        
        # Check for quit commands
        for pattern in self.quit_patterns:
            if re.search(pattern, text):
                return {"command": "quit"}
        
        # Check for help commands
        for pattern in self.help_patterns:
            if re.search(pattern, text):
                return {"command": "help"}
        
        # Check for creation commands
        for pattern in self.creation_patterns:
            match = re.search(pattern, text)
            if match:
                # Try to extract the shape description from the second group
                description = match.group(2) if len(match.groups()) > 1 else match.group(1)
                
                # Find the shape type
                shape_type = self._find_shape_type(description)
                
                if shape_type:
                    # Extract general properties
                    properties = self._extract_properties(description)
                    
                    # Get shape-specific parameters
                    params = self._convert_to_shape_parameters(shape_type, properties)
                    
                    # Add shape type and return creation command
                    return {
                        "command": "create",
                        "shape_type": shape_type,
                        "parameters": params,
                        "description": description
                    }
                else:
                    # No specific shape identified, treat as text-to-shape
                    return {
                        "command": "generate",
                        "description": description
                    }
        
        # Check for classification commands
        for pattern in self.classification_patterns:
            match = re.search(pattern, text)
            if match:
                # Try to extract what needs to be classified
                description = match.group(2) if len(match.groups()) > 1 else ""
                return {
                    "command": "classify",
                    "description": description
                }
        
        # Check for conversion commands
        for pattern in self.conversion_patterns:
            match = re.search(pattern, text)
            if match:
                # Extract path or directory info
                path_info = match.group(2) if len(match.groups()) > 1 else ""
                return {
                    "command": "convert",
                    "path_info": path_info
                }
        
        # Check for adaptive generation commands
        for pattern in self.adaptive_patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) == 2:
                    target_desc, reference_model = match.groups()
                    return {
                        'command': 'generate',
                        'type': 'adaptive',
                        'target_description': target_desc.strip(),
                        'reference_model': reference_model.strip(),
                        'variations': self._extract_variations(text)
                    }
                else:
                    model_path = match.group(1)
                    return {
                        'command': 'generate',
                        'type': 'variation',
                        'model_path': model_path.strip(),
                        'variations': self._extract_variations(text)
                    }
        
        # If no patterns matched but shape is mentioned, default to creation
        shape_type = self._find_shape_type(text)
        if shape_type:
            properties = self._extract_properties(text)
            params = self._convert_to_shape_parameters(shape_type, properties)
            return {
                "command": "create",
                "shape_type": shape_type,
                "parameters": params,
                "description": text
            }
        
        # Default interpretation - assume it's a text-to-shape command
        return {
            "command": "generate",
            "description": text
        }
    
    def _extract_variations(self, text):
        """Extract number of variations from text"""
        variations = 1  # default
        
        # Look for explicit number of variations
        var_match = re.search(r'(\d+)\s+variations?', text)
        if var_match:
            variations = int(var_match.group(1))
        
        return variations
    
    def interpret_and_generate(self, command, shape_ai):
        """
        Interpret user command and generate appropriate 3D shape using ShapeAI.
        
        Args:
            command: User's natural language command
            shape_ai: Instance of ShapeAI for generation
            
        Returns:
            tuple: (generated_points, shape_type, properties)
        """
        command = command.lower().strip()
        
        # Handle single word shape commands directly
        if command in ["cube", "sphere", "cylinder", "pyramid", "torus", "cone"]:
            print(f"Generating a {command}...")
            try:
                generated_points = shape_ai.generate(command)
                if isinstance(generated_points, torch.Tensor):
                    return generated_points, command, {}
                return torch.tensor(generated_points), command, {}
            except Exception as e:
                print(f"Error generating {command}: {e}")
                return None, None, None
        
        # Try to match creation patterns
        creation_command = None
        for pattern in self.creation_patterns:
            match = re.search(pattern, command)
            if match:
                if len(match.groups()) > 1:
                    creation_command = match.group(2)
                else:
                    creation_command = match.group(1)
                break
        
        # If no creation pattern matched, check if it contains a shape name directly
        if not creation_command:
            shape_type = self._find_shape_type(command)
            if shape_type:
                creation_command = command
            else:
                return None, None, None
            
        # Extract shape type and properties
        shape_type = self._find_shape_type(creation_command)
        properties = self._extract_properties(creation_command)
        
        # If no specific shape type found, use the entire command for generation
        if not shape_type:
            generation_prompt = creation_command
        else:
            # Convert properties to parameters suitable for the shape
            params = self._convert_to_shape_parameters(shape_type, properties)
            
            # Format the generation prompt with parameters
            param_str = " ".join(f"{k}={v}" for k, v in params.items() if k != "num_points")
            generation_prompt = f"{shape_type} {param_str}".strip()
        
        try:
            # Generate point cloud using ShapeAI
            print(f"Generating from prompt: {generation_prompt}")
            generated_points = shape_ai.generate(generation_prompt)
            
            # Convert to tensor if it's not already
            if not isinstance(generated_points, torch.Tensor):
                generated_points = torch.tensor(generated_points, dtype=torch.float32)
                
            return generated_points, shape_type or "custom", properties
        except Exception as e:
            print(f"Error generating shape: {e}")
            return None, None, None

    def generate_response(self, command_result):
        """
        Generate a natural language response to the command.
        
        Args:
            command_result: Dictionary with the result of executing a command
            
        Returns:
            A natural language response
        """
        responses = {
            "create": [
                "Creating a {shape_type} with the specified parameters.",
                "Here's the {shape_type} you asked for.",
                "I've generated a {shape_type} based on your description.",
                "Your {shape_type} has been created successfully."
            ],
            "generate": [
                "Generating a shape based on your description: '{description}'",
                "Creating what you described: '{description}'",
                "Transforming your description into a 3D model: '{description}'",
                "Generating a point cloud from: '{description}'"
            ],
            "classify": [
                "Analyzing the shape...",
                "Let me identify this shape for you.",
                "Classifying the shape...",
                "I'll determine what shape this is."
            ],
            "convert": [
                "Converting 3D models from the specified location.",
                "Processing the mesh files you specified.",
                "Reading and converting the 3D models.",
                "Transforming mesh models into point clouds."
            ],
            "help": [
                "Here's what I can do:\n"
                "- Create specific shapes like 'make a large blue cube'\n"
                "- Generate shapes from descriptions like 'a tall cylindrical tower'\n"
                "- Classify existing point clouds\n"
                "- Convert 3D mesh models (FBX, OBJ, etc.) to point clouds",
                
                "You can ask me to:\n"
                "- Create basic shapes (cube, sphere, cylinder, etc.)\n"
                "- Generate shapes from text descriptions\n"
                "- Analyze and classify shapes\n"
                "- Convert 3D models from files",
                
                "Try commands like:\n"
                "- 'Create a small red cube'\n"
                "- 'Generate a tall hollow cylinder'\n"
                "- 'Convert the 3D models in my models folder'\n"
                "- 'What shape is this?'"
            ],
            "quit": [
                "Goodbye! Come back soon.",
                "Exiting the application. Have a nice day!",
                "Shutting down. Thanks for using 3D Shape AI!",
                "Farewell! I hope you enjoyed creating shapes."
            ],
            "error": [
                "I'm sorry, I couldn't understand that command.",
                "I'm not sure what you meant. Try a different command?",
                "I didn't recognize that instruction. Need help? Type 'help'.",
                "I couldn't process that. Please try rephrasing your request."
            ]
        }
        
        command = command_result.get("command", "error")
        
        if command in responses:
            response = random.choice(responses[command])
            
            # Format response with command details
            if command == "create" and "shape_type" in command_result:
                response = response.format(shape_type=command_result["shape_type"])
            elif command == "generate" and "description" in command_result:
                response = response.format(description=command_result["description"])
            
            return response
        else:
            return random.choice(responses["error"])
