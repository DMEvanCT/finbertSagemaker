import boto3
import json
import time
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

def create_sagemaker_execution_role():
    """
    Create a SageMaker execution role with proper trust policy and permissions
    """
    iam = boto3.client('iam')
    
    role_name = 'SageMakerExecutionRole'
    
    # Trust policy allowing SageMaker to assume this role
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        print(f"Creating IAM role: {role_name}")
        
        # Create the role
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='SageMaker execution role for model deployment',
            Path='/'
        )
        
        role_arn = response['Role']['Arn']
        print(f"✓ Created role: {role_arn}")
        
        # Wait a moment for role to be available
        time.sleep(10)
        
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Role {role_name} already exists, using existing role")
        response = iam.get_role(RoleName=role_name)
        role_arn = response['Role']['Arn']
    
    # Attach necessary managed policies
    required_policies = [
        'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
        'arn:aws:iam::aws:policy/AmazonS3FullAccess',
        'arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly'
    ]
    
    for policy_arn in required_policies:
        try:
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy_arn
            )
            print(f"✓ Attached policy: {policy_arn}")
        except iam.exceptions.LimitExceededException:
            print(f"Policy already attached: {policy_arn}")
        except Exception as e:
            print(f"Warning: Could not attach policy {policy_arn}: {e}")
    
    return role_arn

def deploy_huggingface_model(role_arn):
    """
    Deploy the HuggingFace model to SageMaker endpoint
    """
    print("\n" + "="*50)
    print("DEPLOYING HUGGINGFACE MODEL")
    print("="*50)
    
    # Hub Model configuration
    hub = {
        'HF_MODEL_ID': 'ProsusAI/finbert',
        'HF_TASK': 'text-classification',
        'HF_OPTIMUM_BATCH_SIZE': '1',
        'HF_OPTIMUM_SEQUENCE_LENGTH': '512',
    }
    
    print(f"Model ID: {hub['HF_MODEL_ID']}")
    print(f"Task: {hub['HF_TASK']}")
    
    # Create Hugging Face Model Class
    huggingface_model = HuggingFaceModel(
        transformers_version='4.43.2',
        pytorch_version='2.1.2',
        py_version='py310',
        env=hub,
        role=role_arn,
    )
    
    # Let SageMaker know that we compile on startup
    huggingface_model._is_compiled_model = True
    
    print("Deploying model to SageMaker endpoint...")
    print("⚠️  This may take 5-10 minutes...")
    
    # Deploy model to SageMaker Inference
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type='ml.inf2.xlarge',
        endpoint_name=None  # Will auto-generate unique name
    )
    
    print(f"✓ Model deployed successfully!")
    print(f"✓ Endpoint name: {predictor.endpoint_name}")
    
    return predictor

def test_endpoint(predictor):
    """
    Test the deployed endpoint with sample financial texts
    """
    print("\n" + "="*50)
    print("TESTING ENDPOINT")
    print("="*50)
    
    # Test samples
    test_texts = [
        "The company's quarterly earnings exceeded expectations significantly.",
        "Market volatility increased due to economic uncertainty.",
        "Strong revenue growth driven by innovative product launches.",
        "Major layoffs announced affecting 15% of workforce.",
        "Stock price surged after positive analyst upgrade."
    ]
    
    print("Testing with sample financial texts:")
    
    for i, text in enumerate(test_texts, 1):
        try:
            result = predictor.predict({"inputs": text})
            
            # Extract prediction details
            if isinstance(result, list) and len(result) > 0:
                prediction = result[0]
                label = prediction.get('label', 'Unknown')
                score = prediction.get('score', 0.0)
                
                print(f"\n{i}. Text: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
                print(f"   Sentiment: {label} (confidence: {score:.3f})")
            else:
                print(f"\n{i}. Text: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
                print(f"   Result: {result}")
                
        except Exception as e:
            print(f"\n{i}. Error testing text: {e}")

def main():
    """
    Main function to orchestrate the entire deployment process
    """
    print("🚀 SAGEMAKER HUGGINGFACE DEPLOYMENT SCRIPT")
    print("="*60)
    
    try:
        # Step 1: Create SageMaker execution role
        print("Step 1: Creating SageMaker execution role...")
        role_arn = create_sagemaker_execution_role()
        
        # Step 2: Deploy HuggingFace model
        print(f"\nStep 2: Deploying HuggingFace model with role: {role_arn}")
        predictor = deploy_huggingface_model(role_arn)
        
        # Step 3: Test the endpoint
        print("\nStep 3: Testing the deployed endpoint...")
        test_endpoint(predictor)
        
        # Step 4: Provide connection information
        print("\n" + "="*50)
        print("DEPLOYMENT COMPLETE!")
        print("="*50)
        print(f"✓ Endpoint Name: {predictor.endpoint_name}")
        print(f"✓ Role ARN: {role_arn}")
        print("\n📝 To connect to this endpoint later, use:")
        print(f"""
from sagemaker.huggingface import HuggingFacePredictor

predictor = HuggingFacePredictor(
    endpoint_name='{predictor.endpoint_name}',
    sagemaker_session=sagemaker.Session()
)

result = predictor.predict({{"inputs": "Your text here"}})
print(result)
        """)
        
        print("\n⚠️  Remember to delete the endpoint when done to avoid charges:")
        print("predictor.delete_endpoint()")
        
        return predictor
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        print("Please check your AWS credentials and permissions.")
        raise

if __name__ == "__main__":
    # Run the deployment
    predictor = main()
    
    # Optionally keep the predictor object for immediate use
    print(f"\n🎉 Predictor object available as 'predictor'")
    print(f"Endpoint: {predictor.endpoint_name}")
