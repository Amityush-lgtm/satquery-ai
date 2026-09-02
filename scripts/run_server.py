import os
import sys
from pathlib import Path
import uvicorn
from dotenv import load_dotenv

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

load_dotenv()

def main():
    # Render and other cloud platforms provide a PORT environment variable.
    # If it exists, we must bind to 0.0.0.0 instead of 127.0.0.1.
    env_port = os.getenv("PORT")
    if env_port:
        host = os.getenv("SATQUERY_HOST", "0.0.0.0")
        port = int(env_port)
        reload = False
    else:
        host = os.getenv("SATQUERY_HOST", "127.0.0.1")
        port = int(os.getenv("SATQUERY_PORT", 8000))
        reload = True

    print(f"Starting SatQuery AI Server on http://{host}:{port} ...")
    uvicorn.run("satquery.api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
