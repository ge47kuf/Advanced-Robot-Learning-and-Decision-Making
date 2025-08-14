import os
import sys

# Get the directory of the current Python file
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the PYTHONPATH
sys.path.append(os.path.dirname(current_file_dir))
