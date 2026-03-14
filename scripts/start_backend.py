
import uvicorn
import os
import sys

# Add project root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

if __name__ == "__main__":
    uvicorn.run("services.api.main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["services", "src"])
