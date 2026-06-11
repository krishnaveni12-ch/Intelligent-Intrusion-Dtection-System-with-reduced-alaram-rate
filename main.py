from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# 1. Load the Model and Scaler you downloaded from Colab
try:
    model = pickle.load(open('dt_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    # These are the 11 features your model expects based on your Colab correlation analysis
    features = ['id', 'sttl', 'ct_state_ttl', 'dload', 'ct_dst_sport_ltm', 
                'dmean', 'rate', 'swin', 'dwin', 'ct_src_dport_ltm', 'ct_dst_src_ltm']
except Exception as e:
    print(f"Error loading models: {e}")

# Define the data structure for incoming requests
class NetworkTraffic(BaseModel):
    id: float
    sttl: float
    ct_state_ttl: float
    dload: float
    ct_dst_sport_ltm: float
    dmean: float
    rate: float
    swin: float
    dwin: float
    ct_src_dport_ltm: float
    ct_dst_src_ltm: float

@app.post("/predict")
async def predict(traffic: NetworkTraffic):
    try:
        input_dict = traffic.dict()
        missing = [f for f in features if f not in input_dict]
        if missing:
            raise HTTPException(status_code=422, detail={
                'error': 'Missing required fields',
                'missing': missing
            })

        input_data = pd.DataFrame([input_dict], columns=features)

        # The model is trained on these 11 features; scaler expects full 40 features
        # from the original dataset, so we skip scaling here to avoid feature mismatch.
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data).max()

        return {
            "prediction": "Attack" if int(prediction) == 1 else "Normal",
            "confidence": round(float(probability) * 100, 2),
            "status_code": 200
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)