import subprocess

def run_instant_mesh():
    command = [
        "python", "FoodSAM/panoptic.py",
        "--img_path", "images/apple-and-burger.jpeg",
        "--output", "outputs"
    ]
    
    try:
        subprocess.run(command, check=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error occurred while running the command:", e)

if __name__ == "__main__":
    run_instant_mesh()

#python FoodSAM/panoptic.py --img_path images/apple-and-burger.jpeg --output outputs