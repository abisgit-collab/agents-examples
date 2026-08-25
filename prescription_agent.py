"""Extract structured, reviewable data from a prescription image or PDF."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import boto3


EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "patient": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": ["string", "null"]},
                "date_of_birth": {"type": ["string", "null"]},
            },
            "required": ["name", "date_of_birth"],
        },
        "prescriber": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": ["string", "null"]},
                "license_number": {"type": ["string", "null"]},
                "phone": {"type": ["string", "null"]},
            },
            "required": ["name", "license_number", "phone"],
        },
        "pharmacy": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": ["string", "null"]},
                "phone": {"type": ["string", "null"]},
            },
            "required": ["name", "phone"],
        },
        "medications": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "strength": {"type": ["string", "null"]},
                    "form": {"type": ["string", "null"]},
                    "directions": {"type": ["string", "null"]},
                    "quantity": {"type": ["string", "null"]},
                    "refills": {"type": ["string", "null"]},
                },
                "required": [
                    "name",
                    "strength",
                    "form",
                    "directions",
                    "quantity",
                    "refills",
                ],
            },
        },
        "document_date": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "needs_review": {"type": "boolean"},
        "review_reasons": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "patient",
        "prescriber",
        "pharmacy",
        "medications",
        "document_date",
        "confidence",
        "needs_review",
        "review_reasons",
    ],
}

SYSTEM_PROMPT = """You extract data from prescription documents for a human reviewer.

Transcribe only what is visible in the supplied document. Never infer a medicine,
dose, patient, or instruction from context. Use null for a missing or unreadable
field, and preserve the document's wording for directions. Dates should be copied
as written. Set needs_review to true when handwriting, image quality, ambiguity,
or a missing safety-relevant field could affect a pharmacist or clinician's review.
List each reason in review_reasons. This is extraction only, not medical advice.
Return JSON matching the supplied schema.
"""


class PrescriptionAgent:
    """Vision-enabled prescription extraction agent."""

    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        self.client = client or boto3.client("bedrock-runtime", region_name="us-east-1")
        self.model = model or os.getenv("NOVA_MODEL_ID", "amazon.nova-pro-v1:0")

    def extract(self, document: str | Path) -> dict[str, Any]:
        path = Path(document)
        if not path.is_file():
            raise FileNotFoundError(f"Prescription document not found: {path}")
        if path.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            raise ValueError("Supported documents are PDF, PNG, JPG, JPEG, WEBP, and GIF")

        file_bytes = path.read_bytes()
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            document_part = {
                "document": {
                    "format": "pdf",
                    "name": path.stem,
                    "source": {"bytes": file_bytes},
                }
            }
        else:
            image_format = "jpeg" if suffix in {".jpg", ".jpeg"} else suffix[1:]
            document_part = {
                "image": {"format": image_format, "source": {"bytes": file_bytes}}
            }

        response = self.client.converse(
            modelId=self.model,
            system=[
                {
                    "text": f"{SYSTEM_PROMPT}\nJSON schema:\n{json.dumps(EXTRACTION_SCHEMA)}"
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"text": "Extract the prescription details from this document."},
                        document_part,
                    ],
                }
            ],
            inferenceConfig={"temperature": 0},
        )
        response_text = "".join(
            block.get("text", "")
            for block in response["output"]["message"]["content"]
        )
        result = json.loads(response_text.strip().removeprefix("```json").removesuffix("```").strip())
        self._validate_result(result)
        return result

    @staticmethod
    def _validate_result(result: Any) -> None:
        required = set(EXTRACTION_SCHEMA["required"])
        if not isinstance(result, dict) or not required.issubset(result):
            raise ValueError("Model response did not match the prescription extraction schema")
        if not isinstance(result["medications"], list) or not result["medications"]:
            result["needs_review"] = True
            result["review_reasons"].append("No medication could be confidently extracted")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract data from a prescription document")
    parser.add_argument("document", type=Path, help="Path to a prescription PDF or image")
    parser.add_argument(
        "--model",
        help="Bedrock model ID (defaults to NOVA_MODEL_ID or amazon.nova-pro-v1:0)",
    )
    args = parser.parse_args()

    extraction = PrescriptionAgent(model=args.model).extract(args.document)
    print(json.dumps(extraction, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()