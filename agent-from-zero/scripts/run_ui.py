import subprocess
import sys


if __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", "src/agent_from_zero/ui/streamlit_app.py"]
        )
    )
