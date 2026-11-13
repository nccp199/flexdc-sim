"""The text-only policy contract and local validation."""
import json


INSTRUCTIONS = """You are the FlexDC dispatch policy for one simulated data center.
The input is JSON. Use the supplied public state and observed feedback to choose
one complete action for the current physical step. Return only the JSON object
defined by the output schema. 'explanation' is a short decision rationale.
'analysis' is a short operational note, such as the main trade-off or uncertainty.
Use the native action coordinate order. Give a value for every coordinate; use
zero for a statically inactive coordinate. A positive storage value discharges,
a negative value charges. The environment can limit the realized battery power.
IT budget and task preferences reach AQA at the end of this step. AQA handles
placement and per-node caps, whose actuation is delayed. The elapsed interval
is metered using the IT load at its start. Commands and actual load are distinct.
Prioritize the supplied objective, respect current device limits, and use only
observations and prior execution feedback. A validation error means that the
previous proposal was rejected and the simulator has not advanced.
"""


def dumps(value, **kwargs):
    return json.dumps(value, allow_nan=False, ensure_ascii=False, **kwargs)


def decision_schema(action_spec):
    vectors = {
        agent: {"type": "array", "items": {"type": "number"},
                "minItems": size, "maxItems": size}
        for agent, size in action_spec["dimensions"].items()
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "explanation", "analysis"],
        "properties": {
            "action": {
                "type": "object",
                "properties": vectors,
                "required": list(vectors),
                "additionalProperties": False,
            },
            "explanation": {"type": "string"},
            "analysis": {"type": "string"},
        },
    }


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _reject_constant(value):
    raise ValueError(f"{value} is not a finite JSON number")


def validate_decision(raw, env):
    """Validate the JSON envelope, then defer shape/range/mapping to FlexDC."""
    decision = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(decision, dict) or set(decision) != {"action", "explanation", "analysis"}:
        raise ValueError("expected exactly action, explanation and analysis")
    if not all(isinstance(decision[key], str) for key in ("explanation", "analysis")):
        raise ValueError("explanation and analysis must be strings")
    action = decision["action"]
    if not isinstance(action, dict) or set(action) != {"compute", "storage"}:
        raise ValueError("action must contain exactly compute and storage")
    if not all(isinstance(vector, list) for vector in action.values()):
        raise ValueError("compute and storage must be JSON arrays")
    try:
        decoded = env.decode_actions(action["compute"], action["storage"])
    except OverflowError as exc:
        raise ValueError("action numbers must be finite") from exc
    return decision, decoded.to_record()


class MockPolicy:
    """A deterministic transport fixture for exercising validation and execution."""

    method = "mock"
    model = "mock-fixture"

    def decide(self, request, *, timeout_seconds):
        spec = request["contract"]["action_spec"]
        k = len(spec["task_type_order"])
        storage = 0.8 if request["step"] % 2 else -0.4
        action = {"compute": [0.5] * k + [0.65], "storage": [storage]}
        for agent, mask in spec["active_dimensions"].items():
            action[agent] = [value if active else 0.0 for value, active in zip(action[agent], mask)]
        return {
            "raw_output": dumps(dict(action=action, explanation="Deterministic test fixture.",
                                     analysis="Alternating battery requests exercise native limits.")),
            "model": self.model,
            "usage": None,
        }
