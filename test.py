import warp as wp
import subprocess
import os

from DataTypes import *
from Model import Model
from Shape import Box
from utils import SIM_NUM, DEVICE
from Muscle import MuscleParams

# Global muscle parameters
muscle_params = MuscleParams()

def launch_blender_with_script(blender_script_path, data_file_path=None):
    """
    Launch Blender GUI directly and run the animation script
    """
    # Common Blender installation paths
    blender_paths = [
        "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe",  # Windows
        "C:/Program Files/Blender Foundation/Blender 4.1/blender.exe",
        "C:/Program Files/Blender Foundation/Blender 4.0/blender.exe",
        "/Applications/Blender.app/Contents/MacOS/Blender",  # macOS
        "/usr/bin/blender",  # Linux
        "blender"  # If blender is in PATH
    ]
    
    blender_exe = None
    for path in blender_paths:
        if os.path.exists(path):
            blender_exe = path
            break
    
    if not blender_exe:
        # Try to find blender in PATH
        try:
            subprocess.run(["blender", "--version"], capture_output=True, check=True)
            blender_exe = "blender"
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ERROR: Blender not found. Please:")
            print("1. Install Blender")
            print("2. Add Blender to your system PATH, or")
            print("3. Update the blender_paths list with your Blender installation path")
            return False
    
    try:
        # Launch Blender GUI directly with the script
        cmd = [
            blender_exe,
            "--python", blender_script_path  # Execute the Python script in GUI mode
        ]
        
        print(f"Launching Blender GUI with script: {blender_script_path}")
        print("Command:", " ".join(cmd))
        
        # Launch Blender (completely detached from parent process)
        if os.name == 'nt':  # Windows
            # On Windows, use DETACHED_PROCESS to fully detach
            subprocess.Popen(cmd, creationflags=subprocess.DETACHED_PROCESS)
        else:  # Linux/macOS
            # On Unix systems, use start_new_session to detach
            subprocess.Popen(cmd, start_new_session=True)
        
        print("Blender GUI launched successfully!") 
        return True
            
    except Exception as e:
        print(f"Error launching Blender: {e}")
        return False


def run_simulation_and_visualize(scene_id, blender_script_path="blender_animation.py"):
    """
    Run the simulation and automatically launch Blender for visualization
    """
    print(f"Running simulation for scene {scene_id}...")
    
    # Run the simulation
    runScene(scene_id)
    print("Simulation completed!")
    
    # Check if the required data file exists
    body_states_file = "Results/Body_States.txt"
    if not os.path.exists(body_states_file):
        print(f"Warning: {body_states_file} not found. Make sure your simulation generates this file.")
        return False
    
    # Launch Blender with the animation script
    print("Launching Blender for visualization...")
    success = launch_blender_with_script(blender_script_path, body_states_file)
    
    if success:
        print("Success! Blender should now be running with your animation.")
        print("In Blender, press SPACE to play the animation.")
    else:
        print("Failed to launch Blender automatically.")
        print(f"You can manually run: blender --python {blender_script_path}")
    
    return success

def runScene(sceneId):
    if(sceneId==0):
        with wp.ScopedDevice(DEVICE.CPU):
            # alloc and launch on "cpu"
            sides = Vec3(0.1, 0.1, 0.1)
            box_shape = Box(sides)

        with wp.ScopedDevice(DEVICE.GPU):
            model = Model()
            model.gravity = Vec3(0.0, 0.0, -9.8)
            model.steps = 300
            body1 = model.addBody(box_shape, 0.0)
            model.setBodyInitialState(body1, Transform(Vec3(0.0, 0.0, 2.1), Quat(0.0, 0.0, 0.0, 1.0)), Vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            body2 = model.addBody(box_shape, 100.0)
            model.setBodyInitialState(body2, Transform(Vec3(0.0, 0.0, 2.0), Quat(0.0, 0.0, 0.0, 1.0)), Vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            bodies = [body1, body2]
            points = [Vec3(0.0, 0.0, 0.0), Vec3(-1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0)]
            model.addMuscle(bodies, points, muscle_params)
            model.init()
            model.step(model.steps)
            model.saveResults("Results\\")
        
# Run the simulation and launch Blender
if __name__ == "__main__":
    # You can change the scene_id here
    scene_id = 0
    blender_script_path = "blender_animation.py"  # Path to your Blender script
    
    run_simulation_and_visualize(scene_id, blender_script_path)