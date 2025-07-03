import sagemaker
import boto3
from sagemaker.huggingface import HuggingFacePredictor

# Get the endpoint name (you'll need to save this from deployment)
endpoint_name = "huggingface-pytorch-inference-neuronx-m-2025-07-03-02-04-22-948"  # or specify manually

# Create predictor from existing endpoint
predictor = HuggingFacePredictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker.Session()
)

inputs = ["Hyperfine spiked 30% in the past day due to a new software release", "GE stock price droped 20% last quarter due to waining demand for appliances", "Linkedin subscriptions up 30%"]

outputs = []



# Make predictions
for phrase in inputs:
    result = predictor.predict({
        "inputs": phrase
    })
    result[0]["phrase"] = phrase
    outputs.append(result)

for pred in outputs: 
    print(f'Phrase: {pred[0]["phrase"]}, Score: {pred[0]["score"]}, Label: {pred[0]["label"]}')


