"""
Functions for generating different 3D shapes as point clouds.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate a point cloud for a cube
def generate_cube_points(num_points=1024, size=1.0, noise=0.05):
    """Generate a point cloud for a cube"""
    points = []
    
    # Generate points on each face of the cube
    points_per_face = num_points // 6
    
    for dim in range(3):  # x, y, z dimensions
        for sign in [-1, 1]:  # negative and positive faces
            # Create a grid of points on this face
            grid_size = int(np.sqrt(points_per_face))
            for i in range(grid_size):
                for j in range(grid_size):
                    point = [0, 0, 0]
                    point[dim] = sign * size / 2
                    
                    # Calculate coordinates for the other two dimensions
                    other_dims = [d for d in range(3) if d != dim]
                    point[other_dims[0]] = (i / grid_size - 0.5) * size
                    point[other_dims[1]] = (j / grid_size - 0.5) * size
                    
                    # Add some noise
                    noise_vector = np.random.normal(0, noise, 3)
                    point = [p + n for p, n in zip(point, noise_vector)]
                    
                    points.append(point)
    
    # Shuffle and trim to exact number of points
    np.random.shuffle(points)
    points = points[:num_points]
    
    return np.array(points, dtype=np.float32)

# Function to generate a point cloud for a rectangular prism
def generate_rect_prism_points(num_points=1024, size_x=1.0, size_y=2.0, size_z=0.5, noise=0.05):
    """Generate a point cloud for a rectangular prism"""
    points = []
    sizes = [size_x, size_y, size_z]
    
    # Similar to cube, but with different dimensions
    points_per_face = num_points // 6
    
    for dim in range(3):  # x, y, z dimensions
        for sign in [-1, 1]:  # negative and positive faces
            # Create a grid of points on this face
            grid_size = int(np.sqrt(points_per_face))
            for i in range(grid_size):
                for j in range(grid_size):
                    point = [0, 0, 0]
                    point[dim] = sign * sizes[dim] / 2
                    
                    # Calculate coordinates for the other two dimensions
                    other_dims = [d for d in range(3) if d != dim]
                    point[other_dims[0]] = (i / grid_size - 0.5) * sizes[other_dims[0]]
                    point[other_dims[1]] = (j / grid_size - 0.5) * sizes[other_dims[1]]
                    
                    # Add some noise
                    noise_vector = np.random.normal(0, noise, 3)
                    point = [p + n for p, n in zip(point, noise_vector)]
                    
                    points.append(point)
    
    # Shuffle and trim to exact number of points
    np.random.shuffle(points)
    points = points[:num_points]
    
    return np.array(points, dtype=np.float32)

# Function to generate a point cloud for a sphere
def generate_sphere_points(num_points=1024, radius=1.0, noise=0.05):
    """Generate a point cloud for a sphere"""
    points = []
    
    for _ in range(num_points):
        # Generate random point on unit sphere
        vec = np.random.randn(3)
        vec = vec / np.linalg.norm(vec) * radius
        
        # Add some noise
        noise_vector = np.random.normal(0, noise, 3)
        vec = vec + noise_vector
        
        points.append(vec)
    
    return np.array(points, dtype=np.float32)

# Function to generate a point cloud for a cylinder
def generate_cylinder_points(num_points=1024, radius=1.0, height=2.0, noise=0.05):
    """Generate a point cloud for a cylinder"""
    points = []
    
    # Points for the circular ends
    points_per_end = num_points // 3
    for end in [-1, 1]:  # Bottom and top
        for _ in range(points_per_end):
            # Random angle and radius
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius)
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = end * height / 2
            
            # Add some noise
            noise_vector = np.random.normal(0, noise, 3)
            point = [x + noise_vector[0], y + noise_vector[1], z + noise_vector[2]]
            
            points.append(point)
    
    # Points for the curved surface
    points_for_side = num_points - 2 * points_per_end
    for _ in range(points_for_side):
        theta = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(-height/2, height/2)
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        # Add some noise
        noise_vector = np.random.normal(0, noise, 3)
        point = [x + noise_vector[0], y + noise_vector[1], z + noise_vector[2]]
        
        points.append(point)
    
    # Shuffle all points
    np.random.shuffle(points)
    points = points[:num_points]
    
    return np.array(points, dtype=np.float32)

# Function to generate a point cloud for a pyramid
def generate_pyramid_points(num_points=1024, base_size=1.0, height=1.5, noise=0.05):
    """Generate a point cloud for a pyramid"""
    points = []
    
    # Base points (square)
    points_for_base = num_points // 2
    grid_size = int(np.sqrt(points_for_base))
    for i in range(grid_size):
        for j in range(grid_size):
            x = (i / grid_size - 0.5) * base_size
            y = (j / grid_size - 0.5) * base_size
            z = -height / 2
            
            # Add some noise
            noise_vector = np.random.normal(0, noise, 3)
            point = [x + noise_vector[0], y + noise_vector[1], z + noise_vector[2]]
            
            points.append(point)
    
    # Side points (triangular faces)
    points_for_sides = num_points - len(points)
    points_per_side = points_for_sides // 4
    
    # Define the apex point
    apex = [0, 0, height/2]
    
    # Define the base corners
    base_corners = [
        [-base_size/2, -base_size/2, -height/2],
        [base_size/2, -base_size/2, -height/2],
        [base_size/2, base_size/2, -height/2],
        [-base_size/2, base_size/2, -height/2]
    ]
    
    # Generate points for each triangular face
    for i in range(4):
        corner1 = base_corners[i]
        corner2 = base_corners[(i+1) % 4]
        
        for _ in range(points_per_side):
            # Random barycentric coordinates
            a = np.random.uniform(0, 1)
            b = np.random.uniform(0, 1-a)
            c = 1 - a - b
            
            # Generate point as weighted average of three vertices
            x = a * apex[0] + b * corner1[0] + c * corner2[0]
            y = a * apex[1] + b * corner1[1] + c * corner2[1]
            z = a * apex[2] + b * corner1[2] + c * corner2[2]
            
            # Add some noise
            noise_vector = np.random.normal(0, noise, 3)
            point = [x + noise_vector[0], y + noise_vector[1], z + noise_vector[2]]
            
            points.append(point)
    
    # Shuffle and trim to exact number of points
    np.random.shuffle(points)
    points = points[:num_points]
    
    return np.array(points, dtype=np.float32)

# Function to generate a point cloud for a torus
def generate_torus_points(num_points=1024, major_radius=1.0, minor_radius=0.3, noise=0.05):
    """Generate a point cloud for a torus"""
    points = []
    
    for _ in range(num_points):
        # Random angles
        theta = np.random.uniform(0, 2 * np.pi)  # Around the major circle
        phi = np.random.uniform(0, 2 * np.pi)    # Around the minor circle
        
        # Convert to cartesian coordinates
        x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
        y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
        z = minor_radius * np.sin(phi)
        
        # Add some noise
        noise_vector = np.random.normal(0, noise, 3)
        point = [x + noise_vector[0], y + noise_vector[1], z + noise_vector[2]]
        
        points.append(point)
    
    return np.array(points, dtype=np.float32)

# Function to generate a point cloud for a cone
def generate_cone_points(num_points=1024, radius=1.0, height=2.0, noise=0.05):
    """Generate a point cloud for a cone"""
    points = []
    
    # Points for the circular base
    points_for_base = num_points // 2
    for _ in range(points_for_base):
        # Random angle and radius (for the base)
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, radius)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = -height / 2
        
        # Add some noise
        noise_vector = np.random.normal(0, noise, 3)
        point = [x + noise_vector[0], y + noise_vector[1], z + noise_vector[2]]
        
        points.append(point)
    
    # Points for the side surface (triangular surface)
    points_for_side = num_points - points_for_base
    for _ in range(points_for_side):
        # Random angle and height
        theta = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(-height/2, height/2)
        
        # Radius decreases linearly from base to top
        r_factor = (height/2 - z) / height
        r = radius * r_factor
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Add some noise
        noise_vector = np.random.normal(0, noise, 3)
        point = [x + noise_vector[0], y + noise_vector[1], z + noise_vector[2]]
        
        points.append(point)
    
    # Shuffle all points
    np.random.shuffle(points)
    points = points[:num_points]
    
    return np.array(points, dtype=np.float32)

# Function to generate a point cloud for an ellipsoid
def generate_ellipsoid_points(num_points=1024, radius_x=1.0, radius_y=0.7, radius_z=0.5, noise=0.05):
    """Generate a point cloud for an ellipsoid"""
    points = []
    
    for _ in range(num_points):
        # Generate random point on unit sphere
        u = np.random.uniform(0, 1)
        v = np.random.uniform(0, 1)
        
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        
        # Convert to cartesian coordinates and scale by radii
        x = radius_x * np.sin(phi) * np.cos(theta)
        y = radius_y * np.sin(phi) * np.sin(theta)
        z = radius_z * np.cos(phi)
        
        # Add some noise
        noise_vector = np.random.normal(0, noise, 3)
        point = [x + noise_vector[0], y + noise_vector[1], z + noise_vector[2]]
        
        points.append(point)
    
    return np.array(points, dtype=np.float32)

# Function to generate a point cloud for a capsule
def generate_capsule_points(num_points=1024, radius=0.5, height=2.0, noise=0.05):
    """Generate a point cloud for a capsule (cylinder with hemispherical caps)"""
    points = []
    
    # Calculate the total height including hemispheres
    total_height = height + 2 * radius
    
    # Points for the cylindrical body
    points_for_cyl = num_points // 2
    for _ in range(points_for_cyl):
        # Random angle and height for cylinder
        theta = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(-height/2, height/2)
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        # Add some noise
        noise_vector = np.random.normal(0, noise, 3)
        point = [x + noise_vector[0], y + noise_vector[1], z + noise_vector[2]]
        
        points.append(point)
    
    # Points for the two hemispherical caps
    points_for_caps = num_points - points_for_cyl
    points_per_cap = points_for_caps // 2
    
    # Top hemisphere
    for _ in range(points_per_cap):
        # Random point on unit hemisphere
        u = np.random.uniform(0, 1)
        v = np.random.uniform(0, 1)
        
        theta = 2 * np.pi * u
        phi = np.arccos(v)  # Only top half
        
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = height/2 + radius * np.cos(phi)  # Offset by height/2
        
        # Add some noise
        noise_vector = np.random.normal(0, noise, 3)
        point = [x + noise_vector[0], y + noise_vector[1], z + noise_vector[2]]
        
        points.append(point)
    
    # Bottom hemisphere
    for _ in range(points_for_caps - points_per_cap):
        # Random point on unit hemisphere
        u = np.random.uniform(0, 1)
        v = np.random.uniform(0, 1)
        
        theta = 2 * np.pi * u
        phi = np.arccos(v)  # Only top half, but we'll invert it
        
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = -height/2 - radius * np.cos(phi)  # Negative offset and z-coordinate
        
        # Add some noise
        noise_vector = np.random.normal(0, noise, 3)
        point = [x + noise_vector[0], y + noise_vector[1], z + noise_vector[2]]
        
        points.append(point)
    
    # Shuffle all points
    np.random.shuffle(points)
    points = points[:num_points]
    
    return np.array(points, dtype=np.float32)

# Function to determine if a point is inside a polygon
def is_point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray-casting algorithm"""
    x, y = point
    inside = False
    
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        if intersect:
            inside = not inside
        j = i
    
    return inside

# Function to generate a point cloud for a star shape
def generate_star_points(num_points=1024, outer_radius=1.0, inner_radius=0.4, points=5, height=0.3, noise=0.05):
    """Generate a point cloud for a star shape"""
    cloud_points = []
    
    # Generate the star in 2D first (top and bottom faces)
    angles = np.linspace(0, 2 * np.pi, 2 * points, endpoint=False)
    
    # Define the star vertices
    star_vertices = []
    for i, angle in enumerate(angles):
        # Alternate between outer and inner radius
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        star_vertices.append((x, y))
    
    # Close the polygon
    star_vertices.append(star_vertices[0])
    
    # Generate points for top and bottom faces
    points_per_face = num_points // 2
    
    for face_z in [height/2, -height/2]:
        # Use rejection sampling to generate points inside the star
        face_points = 0
        while face_points < points_per_face // 2:
            # Generate random point in the bounding box
            x = np.random.uniform(-outer_radius, outer_radius)
            y = np.random.uniform(-outer_radius, outer_radius)
            
            # Check if it's inside the star using point-in-polygon test
            inside = is_point_in_polygon((x, y), star_vertices)
            
            if inside:
                # Add noise
                noise_vector = np.random.normal(0, noise, 3)
                point = [x + noise_vector[0], y + noise_vector[1], face_z + noise_vector[2]]
                cloud_points.append(point)
                face_points += 1
    
    # Generate points for the side surfaces
    remaining_points = num_points - len(cloud_points)
    
    if remaining_points > 0:
        # Generate points connecting vertices between top and bottom
        for i in range(len(star_vertices) - 1):
            x1, y1 = star_vertices[i]
            points_for_side = remaining_points // (len(star_vertices) - 1)
            
            for _ in range(points_for_side):
                # Random height between top and bottom
                z = np.random.uniform(-height/2, height/2)
                
                # Add noise
                noise_vector = np.random.normal(0, noise, 3)
                point = [x1 + noise_vector[0], y1 + noise_vector[1], z + noise_vector[2]]
                cloud_points.append(point)
    
    # Shuffle all points
    np.random.shuffle(cloud_points)
    cloud_points = cloud_points[:num_points]
    
    return np.array(cloud_points, dtype=np.float32)

# Function to generate a point cloud for a helix
def generate_helix_points(num_points=1024, radius=1.0, pitch=0.3, turns=5, thickness=0.2, noise=0.05):
    """Generate a point cloud for a helix"""
    points = []
    
    # Calculate parameters
    height = pitch * turns
    
    # Generate points along the helix curve
    t_values = np.linspace(0, turns * 2 * np.pi, num_points)
    
    for t in t_values:
        # Calculate helix center point
        cx = radius * np.cos(t)
        cy = radius * np.sin(t)
        cz = (t / (2 * np.pi)) * pitch - height/2
        
        # Add random offset within thickness
        offset_r = np.random.uniform(0, thickness)
        offset_angle = np.random.uniform(0, 2 * np.pi)
        
        # Calculate offset
        offset_x = offset_r * np.cos(offset_angle)
        offset_y = offset_r * np.sin(offset_angle)
        
        # Final point
        x = cx + offset_x
        y = cy + offset_y
        z = cz
        
        # Add some noise
        noise_vector = np.random.normal(0, noise, 3)
        point = [x + noise_vector[0], y + noise_vector[1], z + noise_vector[2]]
        
        points.append(point)
    
    # Shuffle all points
    np.random.shuffle(points)
    points = points[:num_points]
    
    return np.array(points, dtype=np.float32)

# Main function to generate point clouds of various types
def generate_point_cloud(shape_name, num_points=1024, **kwargs):
    """Generate a point cloud based on the specified shape
    
    Args:
        shape_name: String identifier for the shape
        num_points: Number of points to generate
        **kwargs: Additional shape-specific parameters
    
    Returns:
        Numpy array of point cloud coordinates
    """
    # Set default noise level
    noise = kwargs.get('noise', 0.05)
    
    # Generate point cloud based on shape name
    if shape_name.lower() == 'cube':
        size = kwargs.get('size', 1.0)
        return generate_cube_points(num_points, size, noise)
    
    elif shape_name.lower() == 'rectangular_prism' or shape_name.lower() == 'rect_prism':
        size_x = kwargs.get('size_x', 1.0)
        size_y = kwargs.get('size_y', 2.0)
        size_z = kwargs.get('size_z', 0.5)
        return generate_rect_prism_points(num_points, size_x, size_y, size_z, noise)
    
    elif shape_name.lower() == 'sphere':
        radius = kwargs.get('radius', 1.0)
        return generate_sphere_points(num_points, radius, noise)
    
    elif shape_name.lower() == 'cylinder':
        radius = kwargs.get('radius', 1.0)
        height = kwargs.get('height', 2.0)
        return generate_cylinder_points(num_points, radius, height, noise)
    
    elif shape_name.lower() == 'pyramid':
        base_size = kwargs.get('base_size', 1.0)
        height = kwargs.get('height', 1.5)
        return generate_pyramid_points(num_points, base_size, height, noise)
    
    elif shape_name.lower() == 'torus':
        major_radius = kwargs.get('major_radius', 1.0)
        minor_radius = kwargs.get('minor_radius', 0.3)
        return generate_torus_points(num_points, major_radius, minor_radius, noise)
    
    else:
        raise ValueError(f"Unknown shape: {shape_name}")

# Function to generate custom shapes beyond the standard ones
def generate_custom_shape(shape_type, num_points=1024, **params):
    """Generate a custom shape point cloud beyond the predefined types
    
    Args:
        shape_type: String identifier for the shape type
        num_points: Number of points to generate
        **params: Shape-specific parameters
    
    Returns:
        Numpy array of point cloud coordinates
    """
    # Set default noise level
    noise = params.get('noise', 0.05)
    
    # Standard shapes (handled by the original function)
    standard_shapes = ["cube", "rectangular_prism", "rect_prism", "sphere", 
                      "cylinder", "pyramid", "torus"]
    
    if shape_type.lower() in standard_shapes:
        return generate_point_cloud(shape_type, num_points, **params)
    
    # Novel shape types
    elif shape_type.lower() == "cone":
        radius = params.get('radius', 1.0)
        height = params.get('height', 2.0)
        return generate_cone_points(num_points, radius, height, noise)
    
    elif shape_type.lower() == "ellipsoid":
        radius_x = params.get('radius_x', 1.0)
        radius_y = params.get('radius_y', 0.7)
        radius_z = params.get('radius_z', 0.5)
        return generate_ellipsoid_points(num_points, radius_x, radius_y, radius_z, noise)
    
    elif shape_type.lower() == "capsule":
        radius = params.get('radius', 0.5)
        height = params.get('height', 2.0)
        return generate_capsule_points(num_points, radius, height, noise)
    
    elif shape_type.lower() == "star":
        outer_radius = params.get('outer_radius', 1.0)
        inner_radius = params.get('inner_radius', 0.4)
        points = params.get('points', 5)
        height = params.get('height', 0.3)
        return generate_star_points(num_points, outer_radius, inner_radius, points, height, noise)
    
    elif shape_type.lower() == "helix":
        radius = params.get('radius', 1.0)
        pitch = params.get('pitch', 0.3)
        turns = params.get('turns', 5)
        thickness = params.get('thickness', 0.2)
        return generate_helix_points(num_points, radius, pitch, turns, thickness, noise)
    
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")

# Function to visualize a point cloud
def visualize_point_cloud(points, title="Point Cloud"):
    """Visualize a point cloud in 3D
    
    Args:
        points: Numpy array of shape (num_points, 3)
        title: Title for the plot
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set aspect ratio
    max_range = np.max([
        np.max(points[:, 0]) - np.min(points[:, 0]),
        np.max(points[:, 1]) - np.min(points[:, 1]),
        np.max(points[:, 2]) - np.min(points[:, 2])
    ])
    
    mid_x = (np.max(points[:, 0]) + np.min(points[:, 0])) * 0.5
    mid_y = (np.max(points[:, 1]) + np.min(points[:, 1])) * 0.5
    mid_z = (np.max(points[:, 2]) + np.min(points[:, 2])) * 0.5
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    plt.show()