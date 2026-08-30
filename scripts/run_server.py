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
    host = os.getenv("SATQUERY_HOST", "127.0.0.1")
    port = int(os.getenv("SATQUERY_PORT", 8000))
    print(f"Starting SatQuery AI Server on http://{host}:{port} ...")
    uvicorn.run("satquery.api.main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
