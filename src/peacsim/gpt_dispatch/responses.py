"""Direct OpenAI Responses API transport with strict output and no tools."""
import json
import os
import subprocess

from .decisions import INSTRUCTIONS, dumps


class ResponsesPolicy:
    method = "openai-responses"

    def __init__(self, *, model, effort, schema, max_output_tokens):
        self.model = model
        self.effort = effort
        self.schema = schema
        self.max_output_tokens = max_output_tokens

    def decide(self, request, *, timeout_seconds):
        receipt = dict(model=self.model, response_id=None, raw_output=None, usage=None)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key or "\r" in api_key or "\n" in api_key:
            return dict(**receipt, error="Set OPENAI_API_KEY for Responses API access", retryable=False)
        payload = dict(
            model=self.model, instructions=INSTRUCTIONS, input=dumps(request),
            tools=[], tool_choice="none", store=False, max_output_tokens=self.max_output_tokens,
            text=dict(format=dict(type="json_schema", name="flexdc_decision", strict=True, schema=self.schema)),
        )
        if self.effort is not None:
            payload["reasoning"] = dict(effort=self.effort)
        # curl's config is supplied over stdin so the key and observation stay
        # out of process arguments. --disable prevents loading ~/.curlrc.
        config = (
            f'header = {dumps("Authorization: Bearer " + api_key)}\n'
            f'data-binary = {dumps(dumps(payload))}\n'
        )
        try:
            result = subprocess.run(
                ["curl", "--disable", "--silent", "--show-error", "--fail-with-body",
                 "--header", "Content-Type: application/json", "--config", "-",
                 "https://api.openai.com/v1/responses"],
                input=config, text=True, encoding="utf-8", capture_output=True, timeout=timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            # TimeoutExpired may include subprocess arguments. They contain no
            # key, but the fixed message keeps transport diagnostics compact.
            return dict(**receipt, error=f"Responses transport failed: {type(exc).__name__}")
        try:
            response = json.loads(result.stdout)
        except ValueError:
            return dict(**receipt, error=f"Responses transport returned no JSON (curl exit {result.returncode})")
        if response.get("error"):
            # API error messages can echo a partially masked credential.
            code = response["error"].get("code") or response["error"].get("type")
            return dict(**receipt, error=f"Responses API error: {code}",
                        retryable=code in {"rate_limit_exceeded", "server_error", "overloaded"})
        if result.returncode:
            return dict(**receipt, error=f"Responses transport failed (curl exit {result.returncode})")
        receipt.update(model=response.get("model") or self.model, response_id=response.get("id"),
                       usage=response.get("usage"))
        texts, refusals = [], []
        for item in response.get("output", ()):
            if item["type"] == "reasoning":
                continue
            if item["type"] != "message" or item["role"] != "assistant":
                return dict(**receipt, error="Responses returned an unexpected non-message item", retryable=False)
            for part in item["content"]:
                if part["type"] == "output_text":
                    texts.append(part["text"])
                elif part["type"] == "refusal":
                    refusals.append(part["refusal"])
        receipt["raw_output"] = "".join(texts) or "\n".join(refusals) or None
        if response.get("status") != "completed" or refusals or not texts or not receipt["response_id"]:
            return dict(**receipt, error=f"Responses returned no complete decision (status={response.get('status')})")
        return receipt
