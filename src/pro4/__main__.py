import sys
from train import main as train_main
from inference import app as inference_app
import uvicorn

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_main()
    else:
        uvicorn.run(inference_app, host="127.0.0.1", port=8000)
