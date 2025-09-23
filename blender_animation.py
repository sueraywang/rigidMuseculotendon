import bpy
import os

def read_file_and_setup_scene(file_path):
    # Ensure the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Clear existing box objects
    for obj in bpy.data.objects:
        if obj.name.startswith("Box") or obj.name.startswith("Cube"):
            bpy.data.objects.remove(obj, do_unlink=True)

    # Read the file contents
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract number of boxes and frames (ignore other characters in the lines)
    num_boxes = int(''.join(filter(str.isdigit, lines[0])))
    num_sims = int(''.join(filter(str.isdigit, lines[2])))
    num_frames = int(''.join(filter(str.isdigit, lines[3])))

    # Prepare the data for the boxes
    box_data = []
    for line in lines[4:]:
        parts = line.strip().split()
        for i in range(num_boxes):
            for j in range(num_sims):
                rotation = tuple([float(parts[i*num_sims*7+j*7+6]),float(parts[i*num_sims*7+j*7+3]),float(parts[i*num_sims*7+j*7+4]),float(parts[i*num_sims*7+j*7+5])])  # Quaternion (w, x, y, z)
                #position = tuple(map(lambda x: float(x) / 20, parts[i*num_sims*7+j*7+4:i*num_sims*7+j*7+7]))  # Position (x, y, z)
                position = tuple([(float(parts[i*num_sims*7+j*7]) - (j % 16) * 20 )/ 20, (float(parts[i*num_sims*7+j*7+1]) + (j / 16) * 20) / 20, float(parts[i*num_sims*7+j*7+2]) / 20])  # Position (x, y, z)
                box_data.append((rotation, position))

    # Ensure the data matches expectations
    if len(box_data) != num_boxes * num_sims *num_frames:
        print("Error: Mismatch between expected and provided data.")
        return

    # Define a set of 7 colors from MATLAB plot colors
    colors = [
        (0.0, 0.4470, 0.7410, 1.0),  # MATLAB Blue
        (0.8500, 0.3250, 0.0980, 1.0),  # MATLAB Red
        (0.9290, 0.6940, 0.1250, 1.0),  # MATLAB Yellow
        (0.4940, 0.1840, 0.5560, 1.0),  # MATLAB Purple
        (0.4660, 0.6740, 0.1880, 1.0),  # MATLAB Green
        (0.3010, 0.7450, 0.9330, 1.0),  # MATLAB Cyan
        (0.6350, 0.0780, 0.1840, 1.0)   # MATLAB Dark Red
    ]

    # Assign a material to the box
    material = bpy.data.materials.get("High Gloss Plastic")    
    if material is None:
        print("Material 'High Gloss Plastic' not found.")
    else:
        for i in range(len(colors)):
            new_material = bpy.data.materials.get(f"High Gloss Plastic - Color {i+1}") 
            if new_material is None:
                new_material = material.copy()
                new_material.name = f"High Gloss Plastic - Color {i+1}"
                # Set the box color using the 7 MATLAB plot colors
                color = colors[i % len(colors)]
                if new_material.use_nodes:
                    nodes = new_material.node_tree.nodes
                    shader_node = nodes.get("Group.016")
                    if shader_node:
                        shader_node.inputs[0].default_value = color  # Base Color
        
    # Create boxes
    boxes = []
    for i in range(num_boxes):
        box_i = []
        shapes = lines[1].strip().split()
        box_scale = (float(shapes[i*4+3]) / 40, float(shapes[i*4+4]) / 40, float(shapes[i*4+5]) / 40)
        for j in range(num_sims):
            bpy.ops.mesh.primitive_cube_add()
            box = bpy.context.object
            box.name = f"Box_{i}_{j}"
            box.scale = box_scale  # Set the edge length to 0.1
            box_i.append(box)
            box_material = bpy.data.materials.get(f"High Gloss Plastic - Color {i % len(colors)+1}") 
            if box.data.materials:
                box.data.materials[0] = box_material
            else:
                box.data.materials.append(box_material)
        boxes.append(box_i)
            
    # Animate the boxes
    frame_index = 0
    for frame in range(num_frames):
        for i, box_i in enumerate(boxes):
            for j, box in enumerate(box_i):
                rotation, position = box_data[frame_index*num_boxes*num_sims + i*num_sims + j]

                # Set position and rotation
                box.location = position
                box.rotation_mode = 'QUATERNION'
                box.rotation_quaternion = rotation

                # Insert keyframes
                box.keyframe_insert(data_path="location", frame=frame + 1)
                box.keyframe_insert(data_path="rotation_quaternion", frame=frame + 1)
        frame_index += 1

# Replace this path with the actual path to your text file
file_path = "Results/Body_States.txt"
read_file_and_setup_scene(file_path)
