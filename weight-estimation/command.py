#food-sam
#python FoodSAM/panoptic.py --img_path images/real-image.jpeg --output outputs

#instant-mesh
#python run.py configs/instant-mesh-large.yaml examples/chicken.jpg --save_video --output_path outputs/instant-mesh-chicken
import os
import subprocess

def run_instant_mesh(image_path):
        command = [
        "python",
        "InstantMesh/run.py",
        "InstantMesh/configs/instant-mesh-large.yaml",
        "segments/apple-and-burger/apple_0.png",
        "--save_video",
        "--output_path",
        "mesh-outputs/test"
    ]
    
    try:
        subprocess.run(command, check=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error occurred while running the command:", e)

if __name__ == "__main__":

    folder_path = 'InstantMesh/segments/apple-and-burger'  # Replace with your folder path

    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path):
            run_instant_mesh(full_path)