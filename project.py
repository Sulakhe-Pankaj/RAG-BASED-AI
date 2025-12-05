import whisper
import os
import subprocess
files = os.listdir("Videos")
for file in files:
    tutorial_number = file.split("_")[0]
    print(tutorial_number)
    file_name = file.split("_")[-1].split(".")[0]
    print(tutorial_number, file_name)
    subprocess.run(['ffmpeg', '-i', f"videos/{file}", f"audios/{tutorial_number}_{file_name}.mp3"])
