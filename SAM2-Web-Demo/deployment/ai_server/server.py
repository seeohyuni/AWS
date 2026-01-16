import os
import sys
import traceback

# SAM2 레포지토리 경로 추가 및 하위 모듈 경로 보강
SAM2_REPO_DIR = "/home/ec2-user/sam2"
sys.path.append(SAM2_REPO_DIR)

# Fix for OMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import torch
import numpy as np
from PIL import Image
import io
import uvicorn

# sam2 임포트 지연실행 (sys.path 적용 후)
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"Failed to import sam2: {e}")

app = FastAPI()

# --- Configuration ---
# SAM2.1 Hiera Large 모델 경로
CHECKPOINT_PATH = "/home/ec2-user/ai_server/checkpoints/sam2.1_hiera_large.pt"
# build_sam2 내부의 hydra가 config 폴더를 찾을 수 있도록 상대 경로 설정
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Global Predictor
predictor = None

@app.on_event("startup")
def load_model():
    global predictor
    print(f"Working Directory: {os.getcwd()}")
    try:
        # 작업 디렉토리를 레포지토리 루트로 변경하여 hydra가 configs/를 찾게 함
        os.chdir(SAM2_REPO_DIR)
        print(f"Changed to Directory: {os.getcwd()}")
        
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
            
        print(f"Loading SAM2 model with: {MODEL_CFG}")
        sam2_model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device="cpu")
        predictor = SAM2ImagePredictor(sam2_model)
        print("✅ Model Loaded Successfully!")
    except Exception as e:
        print(f"❌ FATAL ERROR LOADING MODEL: {e}")
        traceback.print_exc()

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    global predictor
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Read Image
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGBA")
        image_np = np.array(image_pil.convert("RGB"))

        # Inference
        predictor.set_image(image_np)
        h, w = image_np.shape[:2]
        input_point = np.array([[w // 2, h // 2]])
        input_label = np.array([1])
        
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        # Create Cutout
        best_mask = masks[0] 
        alpha_mask = (best_mask * 255).astype(np.uint8)
        alpha_image = Image.fromarray(alpha_mask, mode='L')
        result_pil = image_pil.convert("RGBA")
        result_pil.putalpha(alpha_image)

        # Return as PNG
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run on all interfaces
    uvicorn.run(app, host="0.0.0.0", port=8000)
