import os
import shutil

folder = "/Users/johnny/Workspace/AAI-521-IN4/Project/AAI-521-Group2_Project/data/raw/leaf beetle"
trash = "/Users/johnny/Workspace/AAI-521-IN4/Project/AAI-521-Group2_Project/data/raw/leaf beetle/unannotated"

os.makedirs(trash, exist_ok=True)

jpg_files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
xml_bases = {f.replace(".xml", "") for f in os.listdir(folder) if f.endswith(".xml")}

for jpg in jpg_files:
    basename = jpg.replace(".jpg", "")
    if basename not in xml_bases:
        src = os.path.join(folder, jpg)
        dst = os.path.join(trash, jpg)
        print("Moving:", src)
        shutil.move(src, dst)

