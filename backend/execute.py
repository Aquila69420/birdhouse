import subprocess
import os

def execute_command():
    try:
        # Set environment variables
        env_vars = os.environ.copy()
        env_vars["TASK_ID"] = "12"
        env_vars["FLOCK_API_KEY"] = "8ZX738S09CE2KX1V1WOO3R7A2XRRQJEA"
        env_vars["HF_TOKEN"] = "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"
        env_vars["CUDA_VISIBLE_DEVICES"] = "0"
        env_vars["HF_USERNAME"] = "atharva-pawar"

        # Command to execute
        command = "python backend/full_automation.py"

        # Use subprocess to execute the command and capture output
        result = subprocess.run(command, shell=True, text=True, capture_output=True, env=env_vars)

        # Send the output and error (if any) as JSON response or print to console
        print("Output:", result.stdout)
        print("Error:", result.stderr)

    except Exception as e:
        print("Exception:", str(e))

execute_command()
