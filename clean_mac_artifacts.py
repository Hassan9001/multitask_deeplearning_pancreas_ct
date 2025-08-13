import sys
import os
import shutil
#if this file is in parent directory, run this in commmand line: 
    # """ python ../clean_mac_artifacts.py """
#if its in the same directory, 
    # # """ python ./clean_mac_artifacts.py """
    # OR can run this in notebook: """
    # import clean_mac_artifacts
    # clean_mac_artifacts.mac_sux('./')
    # """
def mac_sux(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Delete .DS_Store files
        for file in filenames:
            if file == ".DS_Store":
                try:
                    os.remove(os.path.join(dirpath, file))
                    print(f"Deleted file: {os.path.join(dirpath, file)}\n")
                except Exception as e:
                    print(f"Error deleting {file}: {e}\n")
        # Delete __MACOSX folders
        for dirname in dirnames:
            if dirname == "__MACOSX":
                try:
                    shutil.rmtree(os.path.join(dirpath, dirname))
                    print(f"Deleted folder: {os.path.join(dirpath, dirname)}\n")
                except Exception as e:
                    print(f"Error deleting folder {dirname}: {e}\n")
# This makes it runnable from the command line
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else './'
    mac_sux(path)