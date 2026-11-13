"""The online observation, event notification and feedback used in the Harness experiment."""
from collections import deque


EVENT_CONTEXT = """
输入额外提供 effective（当前状态生效的并网硬上限 kW，null 表示当前无限制，以及已生效的优先类型编号和受影响的已到达未完成任务 ID）。operator_notices 只报告当前已生效的新通知。上一秒的 executed_conditions 记录真实执行和保护结果。临时上限作用于实际并网进口。原生即时保护先减少充电，再补充可用放电，仍不足时在节点最小功率内削减当期计算。下一期的实际 IT 仍受原有 AQA 执行时序影响。优先类型的等待任务在原生可激活节点额度内优先入场，同类保持 FIFO，不抢占已运行任务。你仍自主提交相同的六维动作。
"""


def compact_request(request, cohort):
    state = request["observation"]
    signals = state["signals_observed_now"]
    battery = state["battery"]
    raw = state["rl_raw_summary"]
    tasks = state["unfinished_tasks"]["rows"]
    profiles = request["contract"]["task_types"]
    reserve = {p["type_id"]: [] for p in profiles}
    for row in tasks:
        reserve[row[1]].append(row[5] - row[6] / profiles[row[1]]["required_nodes"])
    queues = []
    for j, values in reserve.items():
        prefix = f"type_{j}."
        queues.append([int(raw[prefix + "waiting_count"]), int(raw[prefix + "running_count"]),
                       round(raw[prefix + "waiting_work_node_seconds"] +
                             raw[prefix + "running_work_node_seconds"], 1),
                       round(min(values), 1) if values else None])
    batch = [row for row in tasks if row[0] in cohort]
    waiting = [row[5] for row in batch if row[2] == "waiting"]
    running = [row[7] for row in batch if row[2] == "running"]
    previous = request["recent_decisions"]
    cost, last = 0, None
    if previous:
        recent = previous[-1]
        actual = recent["execution"]
        power, = actual["power_intervals"]
        bess, = actual["battery_execution"]
        action = actual["submitted_action"]
        cost = recent["cumulative"]["grid_purchase_cost"]["amount"]
        last = [action["compute"], action["storage"],
                round(actual["it_actuation"]["aqa_received_budget_kw"], 3),
                round(power["actual_it_kw"], 3), round(power["actual_grid_import_kw"], 3),
                round(bess["executed"]["charge_power_kw"], 3), round(bess["executed"]["discharge_power_kw"], 3),
                len(actual["task_events"]["started_job_ids"]), len(actual["task_events"]["completed_job_ids"]),
                round(power["grid_purchase_cost"]["amount"], 8), bess["reasons"]]
    it = state["servers"]["actual_it_kw"]
    limits = battery["device_limits_for_step_kw"]
    return dict(
        t=state["simulation_seconds"],
        signal=[round(signals[k], 4) for k in ("pv_available_kw", "price_per_kwh", "grid_target_kw")],
        plant=[round(it, 3), round(it + raw["common.pending_it_delta_kw"], 3),
               state["servers"]["busy"], round(battery["soc"], 7),
               round(limits["charge_kw"], 4), round(limits["discharge_kw"], 4)],
        q=queues, batch=[len(waiting), len(running), len(cohort) - len(batch),
                         round(min(running), 4) if running else None, min(waiting, default=None)],
        total=[state["task_counts"]["completed"], round(cost, 8)],
        last=last, validation_error=request["validation_feedback"],
    )


def effective(state):
    return state["operating_conditions"]


def model_payload(request, cohort, objective):
    model_input = compact_request(request, cohort)
    now = request["observation"]
    conditions = effective(now)
    model_input["effective"] = conditions
    previous = request["recent_decisions"]
    notices = []
    if not previous:
        notices.append(objective)
    else:
        execution = previous[-1]["execution"]
        past = execution["operating_conditions_executed"][-1]
        model_input["executed_conditions"] = past
        # Prices come from the just-metered interval, not a future event list.
        price = now["signals_observed_now"]["price_per_kwh"]
        old_price = execution["power_intervals"][-1]["price_per_kwh"]
        if price != old_price:
            notices.append(f"当前电价已变为 {price:g} CNY/kWh。请根据当前价格、服务需求与设备状态调整。")
        limit = conditions["grid_limit_kw"]
        if limit != past["grid_limit_kw"]:
            notices.append("临时并网购电功率上限已解除。" if limit is None else
                           f"临时需求响应现已生效，实际并网购电功率必须不高于 {limit:g} kW，同时尽可能保障重要任务按时完成。")
        types = conditions["priority_types"]
        if types != past["priority_types"]:
            name = request["contract"]["task_types"][types[0]]["name"] if types else None
            notices.append(f"从现在起，任务类型 {types[0]}（{name}）具有更高执行优先级。请优先保障该类任务的处理，同时尽可能降低对其他任务的影响。"
                           if types else "临时任务优先级已恢复。")
    if notices:
        model_input["operator_notices"] = notices
    return model_input


class OnlinePayload:
    """Retain only elapsed execution results and expose no scenario object."""
    def __init__(self, objective):
        self.objective = objective
        self.history = deque(maxlen=8)

    def __call__(self, request, cohort):
        payload = model_payload(request, cohort, self.objective)
        t = payload["t"]
        if t and (not self.history or self.history[-1][0] < t-1):
            last = request["recent_decisions"][-1]["execution"]
            power, = last["power_intervals"]
            counts = last["task_counts"]
            assert power["start_seconds"] == t-1 and power["end_seconds"] == t
            self.history.append([t-1, counts["completed"], counts["waiting"],
                                 len(last["task_events"]["completed_job_ids"]),
                                 round(power["actual_it_kw"], 3), round(power["actual_grid_import_kw"], 3)])
        raw = request["observation"]["rl_raw_summary"]
        payload["time_left"] = request["experiment_steps_remaining"]
        payload["work_by_state"] = [
            [round(raw[f"type_{j}.{state}_work_node_seconds"], 1) for state in ("waiting", "running")]
            for j in range(len(payload["q"]))]
        payload["recent_execution"] = list(self.history)
        return payload
