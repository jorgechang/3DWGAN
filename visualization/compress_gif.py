import subprocess
import os

# Set the path to the folder with the .gif files
folder_paths = ["chair","plane"]

# Loop through all the .gif files in the folder
for folder in folder_paths:
    for filename in os.listdir(folder):
        if filename.endswith(".gif"):
            input_path = os.path.join(folder,filename)
            output_path = os.path.join(folder,f"{filename}")
            # Construct the command to run gibsicle on the file
            command = [
                "gifsicle",
                "-O3",
                "--colors=256",
                input_path,
                "-o",
                output_path,
            ]

            # Run the command in a subprocess
            subprocess.run(command)
