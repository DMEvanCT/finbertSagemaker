# finbertSagemaker

Deploy and use the [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) financial sentiment analysis model on AWS SageMaker using Hugging Face.

## Overview

This repository provides scripts to:
- **Set up AWS IAM roles for SageMaker deployment**
- **Deploy FinBERT as a Hugging Face model to a SageMaker endpoint**
- **Invoke the deployed endpoint for financial sentiment classification**

## Contents

- `finbert.py`: Automates IAM role creation and deploys the FinBERT model to SageMaker.
- `finbert-invoke.py`: Example script for invoking the SageMaker endpoint with sample financial phrases.

> **Note:** This project uses the Hugging Face `transformers` and `sagemaker` Python libraries.

## Getting Started

### Requirements

- Python 3.10+
- AWS Account with SageMaker permissions
- [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [sagemaker](https://sagemaker.readthedocs.io/en/stable/)
- AWS CLI configured with appropriate credentials

### Installation

Install dependencies:

```bash
pip install boto3 sagemaker
```

Configure your AWS CLI if you haven't already:

```bash
aws configure
```

### SageMaker Role Setup & Model Deployment

Run the deployment script to:
1. Create a SageMaker execution role (if it doesn't exist)
2. Deploy the FinBERT model to a SageMaker endpoint

```bash
python finbert.py
```

This script will output the name of the SageMaker endpoint once deployment is complete.

### Invoking the Endpoint

Edit the `endpoint_name` variable in `finbert-invoke.py` to match your deployed endpoint. Then run:

```bash
python finbert-invoke.py
```

This will send sample financial phrases to the endpoint and print out their sentiment classification.

## Example Output

```
Phrase: Hyperfine spiked 30% in the past day due to a new software release, Score: 0.98, Label: positive
Phrase: GE stock price droped 20% last quarter due to waining demand for appliances, Score: 0.95, Label: negative
Phrase: Linkedin subscriptions up 30%, Score: 0.92, Label: positive
```

## References

- [FinBERT on Hugging Face](https://huggingface.co/ProsusAI/finbert)
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
- [Hugging Face SageMaker Integration](https://huggingface.co/docs/sagemaker/main/en/overview)

## License

Specify your project license here.

---

*This README was generated based on available code and may be incomplete. [See more on GitHub.](https://github.com/DMEvanCT/finbertSagemaker/search?q=)*