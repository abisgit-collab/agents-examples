# Prescription extraction agent

This example uses an Amazon Nova vision model through AWS Bedrock to extract structured data from a prescription PDF or image. It is an extraction aid for a human reviewer, not a diagnostic or dispensing system.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
aws configure
# Optional: choose a different Nova model or AWS region.
export NOVA_MODEL_ID="amazon.nova-lite-v1:0"
export AWS_DEFAULT_REGION="us-east-1"
```

## Run

```bash
python prescription_agent.py ./prescription.jpg
python prescription_agent.py ./prescription.pdf --model amazon.nova-pro-v1:0
```

The agent returns patient, prescriber, pharmacy, medication, and document-date fields, plus a confidence score and review reasons. Unreadable or absent values are returned as `null`; the agent is instructed not to guess. `needs_review` is set when the document is ambiguous or no medication can be extracted.

Prescriptions contain sensitive health information. Use approved storage and access controls, avoid committing documents or extracted output, and require a qualified human to verify every result before it is used.
