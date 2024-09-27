"""
title: Presidio PII Redaction Pipeline
author: justinh-rahb
date: 2024-07-07
version: 0.1.0
license: MIT
description: A pipeline for redacting personally identifiable information (PII) using the Presidio library.
requirements: presidio-analyzer, presidio-anonymizer
"""

import os
from typing import List, Optional

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from pydantic import BaseModel


class Pipeline:
    class Valves(BaseModel):
        ENABLE_PII_FILTER: bool = os.getenv("ENABLE_PII_FILTER", "false").lower() == "true"
        pipelines: List[str] = os.getenv("PII_FILTER_PIPELINES", "*").split(";")
        priority: int = 0
        entities_to_redact: List[str] = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "IP_ADDRESS", "US_PASSPORT", "LOCATION", "DATE_TIME", "NRP", "MEDICAL_LICENSE", "URL"]
        language: str = "en"

    def __init__(self):
        self.type = "filter"
        self.id = "presidio_pii_filter"
        self.name = "Presidio PII Filter"

        self.valves = self.Valves()
        if not self.valves.ENABLE_PII_FILTER:
            self.valves.pipelines = []

        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    def redact_pii(self, text: str) -> str:
        results = self.analyzer.analyze(
            text=text,
            language=self.valves.language,
            entities=self.valves.entities_to_redact,
        )

        anonymized_text = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={"DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})},
        )

        return anonymized_text.text

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"pipe:{__name__}")

        messages = body.get("messages", [])
        for message in messages:
            if message.get("role") == "user":
                print(f"Before PII filter: {message['content']}")
                message["content"] = self.redact_pii(message["content"])
                print(f"After PII filter: {message['content']}")

        return body
