#!/bin/sh

# Navigate to your script directory (adjust as necessary)
cd src

# Define the command to activate the virtual environment
ACTIVATE_VENV="source ../myenv/bin/activate"

# Run task_1_main.py in a new Terminal window
osascript -e 'tell app "Terminal"
    do script "cd '$(PWD)' && '"$ACTIVATE_VENV"' && python3 task_1_main.py"
end tell'

# Run task_2_main.py in a new Terminal window
osascript -e 'tell app "Terminal"
    do script "cd '$(PWD)' && '"$ACTIVATE_VENV"' && python3 task_2_main.py"
end tell'

# Run task_3_main.py in a new Terminal window
osascript -e 'tell app "Terminal"
    do script "cd '$(PWD)' && '"$ACTIVATE_VENV"' && python3 task_3_main.py"
end tell'
