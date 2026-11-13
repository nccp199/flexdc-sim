"""Two read-only calculations over the current dispatch observation."""
from collections import Counter
from heapq import heapify, heappop, heappush
from math import ceil, fsum


def _projection(state, profiles, cohort, horizon, *, full_speed):
    """Project occupied slots, then test arrived waiting jobs at full speed."""
    t = state["simulation_seconds"]
    nodes = {r[1]: r for r in state["nodes"]["rows"] if r[1] is not None}
    slots = [0] * state["servers"]["idle"]
    running, waiting, cohort_finishes = [], [], []
    for job, kind, status, arrival, start, slack, work, *_ in state["unfinished_tasks"]["rows"]:
        assert arrival <= t
        p = profiles[kind]
        assert p["required_nodes"] == 1
        if status == "waiting":
            waiting.append((job, kind, slack, ceil(work)))
            continue
        node = nodes[job]
        lo, hi = p["min_node_power_w"], p["max_node_power_w"]
        fraction = max(0., min(1., (node[2] - lo) / (hi - lo)))
        ratio = p["min_run_seconds"] / p["max_run_seconds"]
        rate = 1. if full_speed else ratio + (1. - ratio)*fraction
        finish = max(1, ceil(work/rate - 1e-9)) + int(start == t)
        running.append((job, kind, finish))
        slots.append(finish)
        if job in cohort:
            cohort_finishes.append((job, finish, slack))
    first_slot = min(slots)
    # Estimate service, never emit an action. This optimistic admission test
    # ignores native type quotas and assumes each waiting job then runs at max.
    priority = set(state["operating_conditions"]["priority_types"])
    waiting.sort(key=lambda r: (r[0] not in cohort, r[1] not in priority, r[3], r[2], r[0]))
    heapify(slots)
    completed_running = Counter(kind for _, kind, finish in running if finish <= horizon)
    completed_waiting = Counter()
    individually_possible = sum(max(1, first_slot) + 1 + duration <= horizon for _, _, _, duration in waiting)
    for job, kind, slack, duration in waiting:
        # Admission is at a future boundary (even for an idle slot); a newly
        # started task then skips one interval of progress.
        finish = max(1, heappop(slots)) + 1 + duration
        heappush(slots, finish)
        if finish <= horizon:
            completed_waiting[kind] += 1
        if job in cohort:
            cohort_finishes.append((job, finish, slack))
    latest = max((finish for _, finish, _ in cohort_finishes), default=None)
    return dict(
        running_finish_by_type=[completed_running[k] for k in profiles],
        waiting_finish_by_type=[completed_waiting[k] for k in profiles],
        waiting_individually_possible_ignoring_contention=individually_possible,
        first_slot_in_s=first_slot, batch_latest_finish_t=t+latest if latest is not None else None,
        batch_at_risk_ids=[job for job, finish, slack in cohort_finishes if finish > min(slack, horizon)],
    )


def delivery_risk(request, cohort, target):
    state = request["observation"]
    profiles = {p["type_id"]: p for p in request["contract"]["task_types"]}
    horizon = request["experiment_steps_remaining"]
    done = state["task_counts"]["completed"]
    cases = {}
    for name, fast in (("running_at_current_power", False), ("all_at_full_power", True)):
        result = _projection(state, profiles, cohort, horizon, full_speed=fast)
        projected = done + sum(result["running_finish_by_type"]) + sum(result["waiting_finish_by_type"])
        result.update(arrived_only_projected_total=projected, gap_vs_target=max(0, target-projected) if target is not None else None)
        cases[name] = result
    return dict(t=state["simulation_seconds"], remaining_s=horizon, completed=done,
                target=target, waiting=state["task_counts"]["waiting"], cases=cases,
                assumption="Arrived jobs only; no future arrivals. Waiting jobs use estimated free slots and full speed; "
                           "native AQA quotas, future power and arrivals may change outcomes. A gap is not proof of infeasibility.")


def _envelope(it, *, factor, pv, limit, charge, discharge):
    net = factor*it - pv
    required = max(0., net-limit) if limit is not None else 0.
    upper_charge = charge if limit is None else max(0., min(charge, limit-net))
    return dict(it_kw=round(it, 6), standby_import_kw=round(max(0., net), 6),
                min_discharge_kw_for_cap=round(required, 6), max_charge_kw_under_cap=round(upper_charge, 6),
                max_discharge_kw_without_export=round(max(0., min(discharge, net)), 6),
                feasible_at_this_it=required <= discharge+1e-9)


def grid_feasibility(request):
    s, contract = request["observation"], request["contract"]
    signal = s["signals_observed_now"]
    pv = signal["pv_available_kw"]
    limit = s["operating_conditions"]["grid_limit_kw"]
    factor = 1 + contract["cooling_power_ratio"]
    limits = s["battery"]["device_limits_for_step_kw"]
    charge, discharge = limits["charge_kw"], limits["discharge_kw"]
    current = s["servers"]["actual_it_kw"]
    pending = fsum(n[2] if n[3] is None else n[3] for n in s["nodes"]["rows"])/1000
    profiles = {p["type_id"]: p for p in contract["task_types"]}
    jobs = {r[0]: r[1] for r in s["unfinished_tasks"]["rows"]}
    minimum = fsum(n[2] if n[1] is None else profiles[jobs[n[1]]]["min_node_power_w"]
                   for n in s["nodes"]["rows"])/1000
    kwargs = dict(factor=factor, pv=pv, limit=limit, charge=charge, discharge=discharge)
    storage = s["battery"]
    return dict(
        t=s["simulation_seconds"], current_price_cny_per_kwh=signal["price_per_kwh"], pv_kw=pv,
        import_cap_kw=limit, battery_soc=storage["soc"], device_charge_kw=charge, device_discharge_kw=discharge,
        storage_coordinate_at_nameplate_kw=dict(charge=contract["battery"]["max_charge_power_kw"],
                                               discharge=contract["battery"]["max_discharge_power_kw"]),
        pv_only_it_kw=round(pv/factor, 6), current=_envelope(current, **kwargs),
        pending_same_conditions=_envelope(pending, **kwargs),
        physical_it_ceiling_kw_with_bess=(round((limit+pv+discharge)/factor, 6) if limit is not None else None),
        physical_it_ceiling_kw_without_bess=(round((limit+pv)/factor, 6) if limit is not None else None),
        minimum_current_nodes_it_kw=round(minimum, 6),
        feasible_with_minimum_node_power=_envelope(minimum, **kwargs)["feasible_at_this_it"],
        assumption="Current envelope is for this metered second. Pending uses present commands and signals, "
                   "excluding releases and starts. Ceilings bound actual IT, not the submitted delayed budget.")


TOOLS = {
    "delivery_risk": "Estimate completion and deadline risk from arrived jobs only under current-power and full-speed assumptions. No actions.",
    "grid_feasibility": "Compute this second's feasible BESS/import range and a conditional pending-power envelope. No actions.",
}
TOOL_SPECS = [
    dict(type="function", name=name, description=description,
         inputSchema={"type":"object", "properties":{}, "additionalProperties":False}, deferLoading=False)
    for name, description in TOOLS.items()
]


def invoke(name, arguments, request, cohort, target):
    if name not in TOOLS or arguments != {}:
        raise ValueError("Expected an allowlisted tool and an empty JSON object.")
    if name == "delivery_risk":
        return delivery_risk(request, cohort, target)
    return grid_feasibility(request)
