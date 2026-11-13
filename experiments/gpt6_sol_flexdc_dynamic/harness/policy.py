"""Model-selected local queries, followed by the unchanged six-dimensional action."""
import json
from time import monotonic

from peacsim.gpt_dispatch.codex import CodexPolicy
from peacsim.gpt_dispatch.decisions import dumps
from .primitives import TOOLS, invoke


def choice_schema(action_schema):
    query = dict(type="object", additionalProperties=False, required=["query"], properties={
        "query": dict(type="array", items=dict(type="string", enum=list(TOOLS)), minItems=1, maxItems=2)})
    return dict(type="object", additionalProperties=False, required=["response"], properties={
        "response": {"anyOf": [action_schema, query]}})


class HarnessPolicy(CodexPolicy):
    def __init__(self, protocol, renderer, *, force_smoke_query=False, **kwargs):
        super().__init__(**kwargs)
        self.renderer = renderer
        self.cohort = frozenset(protocol["evaluation_cohort"]["job_ids"])
        self.target = protocol["completion_target"]
        self.wall_budget = protocol["wall_budget_seconds"]
        self.token_threshold = protocol["reported_total_tokens_stop_threshold"]
        self.max_attempts = protocol["max_model_attempts"]
        self.thread_steps = protocol["thread_steps"]
        self.started = None
        self.calls = self.tokens = 0
        self._thread_segment = 0
        self._action_schema = self.schema
        self._choice_schema = choice_schema(self.schema)
        self._force_smoke_query = force_smoke_query
        self._queried_at = None

    def _turn(self, message, deadline, schema):
        self.schema = schema
        try:
            self.calls += 1
            result = CodexPolicy.decide(self, message, timeout_seconds=self._remaining(deadline))
            self.tokens += (result.get("usage") or {}).get("totalTokens", 0)
            return result
        finally:
            self.schema = self._action_schema

    def _has_budget(self):
        return self.tokens < self.token_threshold and self.calls < self.max_attempts

    def decide(self, request, *, timeout_seconds):
        if self.started is None:
            self.started = monotonic()
        deadline = min(self.started + self.wall_budget, monotonic() + timeout_seconds)
        if deadline <= monotonic() or not self._has_budget():
            return dict(error="Experiment inference budget exhausted", retryable=False,
                        model=self.model, raw_output=None, usage=None)
        payload = self.renderer(request, self.cohort)
        t = payload["t"]
        segment = t // self.thread_steps if self.thread_steps else 0
        if segment != self._thread_segment:
            self.close()
            self._thread_segment = segment
            payload["operator_notices"] = [
                self.renderer.objective,
                "Continuation of the same simulation. Use the current state and included real execution feedback.",
                *payload.get("operator_notices", ()),
            ]
        elif segment and t % self.thread_steps == 0:
            # Preserve the objective across validation retries on a new thread.
            payload["operator_notices"] = [self.renderer.objective, *payload.get("operator_notices", ())]
        if self._force_smoke_query and t == 0:
            payload["smoke_check"] = "For this interface test only, query both tools before choosing an action."
        subturns, calls = [], []
        try:
            if self._queried_at == t:
                result = self._turn(payload, deadline, self._action_schema)
                subturns.append(dict(input=payload, **result))
            else:
                choice = self._turn(payload, deadline, self._choice_schema)
                subturns.append(dict(input=payload, **choice))
                result = choice
                if "error" not in choice:
                    response = json.loads(choice["raw_output"])["response"]
                    if not isinstance(response, dict):
                        raise ValueError("Expected a structured action or query object.")
                    if "query" in response:
                        selected = response["query"]
                        if (set(response) != {"query"} or not isinstance(selected, list) or
                                not 1 <= len(selected) <= 2 or len(set(selected)) != len(selected)):
                            raise ValueError("Choose one or both distinct allowlisted tools.")
                        self._queried_at = t
                        for name in selected:
                            outcome = invoke(name, {}, request, self.cohort, self.target)
                            calls.append(dict(t=t, name=name, arguments={}, requested_by_turn=choice["turn_id"],
                                              thread_id=choice["thread_id"], result=outcome))
                        followup = dict(t=t, tool_results=calls,
                                        instruction="Use these results as optional evidence. Submit the six-dimensional action now.")
                        if not self._has_budget():
                            raise TimeoutError("Inference budget exhausted after query")
                        result = self._turn(followup, deadline, self._action_schema)
                        subturns.append(dict(input=followup, **result))
                        if result.get("thread_id") != choice.get("thread_id"):
                            raise RuntimeError("Tool result and action must remain in the same Codex thread.")
                    else:
                        result = dict(choice, raw_output=dumps(response))
        except (ValueError, KeyError, TypeError, TimeoutError, RuntimeError) as exc:
            result = dict(model=self.model, raw_output=None, error=f"{type(exc).__name__}: {exc}",
                          retryable=not isinstance(exc, (TimeoutError, RuntimeError)))
        usage = {}
        for turn in subturns:
            for key, value in (turn.get("usage") or {}).items():
                if type(value) is int:
                    usage[key] = usage.get(key, 0) + value
        return dict(result, usage=usage or None, model_input=payload, model_turns=subturns, tool_calls=calls)
