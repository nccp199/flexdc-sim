# FlexDC 云端 Harness 实验

使用 CPython 3.10–3.13。从仓库根目录执行以下命令，云主机需要能运行 Codex CLI 并访问支持 `gpt-6.1-sol` 的 ChatGPT 账户。应用服务接口曾在 Codex CLI 0.167.0-alpha.3.2 上验证。将 Codex home 保存在仓库目录之外，使用 ChatGPT 登录。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-cloud.txt

export FLEXDC_CODEX_HOME="$HOME/.flexdc-codex"
mkdir -p -m 700 "$FLEXDC_CODEX_HOME"
cp -n experiments/gpt6_sol_flexdc_dynamic/harness/codex.config.example.toml "$FLEXDC_CODEX_HOME/config.toml"
CODEX_HOME="$FLEXDC_CODEX_HOME" codex login --device-auth
CODEX_HOME="$FLEXDC_CODEX_HOME" codex login status
```

已登录时可复用这个独立目录。每步模型上下文仅含在线数据与已发生事件。模型自行选择是否查询交付风险或并网可行域，查询结果回到同一个模型线程，由模型提交动作。

```sh
PYTHONPATH=src .venv/bin/python -m experiments.gpt6_sol_flexdc_dynamic.harness.run \
  --mode codex --steps 3600 --codex-home "$FLEXDC_CODEX_HOME" \
  --step-timeout 90 --wall-budget-seconds 43200 --retries 1 \
  --max-model-turns 10800 --token-stop-threshold 576000000 --thread-steps 100 \
  --output "cloud-runs/harness-$(date -u +%Y%m%dT%H%M%SZ)"
```

- `--steps` 可在 1–7,200 之间设置。物理步长固定 1 秒。动态场景、任务及调度种子保持 101，原生场景仍有 7,200 秒，运行在所选物理步停止。事件使用上一轮配置。100 秒起，低价倍率结束且 Bloom 类优先恢复，价格回到原生信号。固定批次仍使用原生第 100 秒截止。
- `--target-completed N` 可显式设置窗口完工目标。100 步默认沿用 B2 的 90 项，其余长度默认不指定数值。只更新指导中的评价窗口和相关完工目标，不改物理模型、事件或奖励。
- 默认每步 90 秒、1 次重试。整轮墙钟默认 12 秒乘物理步数，默认轮次上限为物理步数乘 `(2 + retries)`，默认 token 停止阈值为物理步数乘 160,000。这些数值是实验上限。实际费用、速率限制和模型可用性由账户决定。Codex 的 reported tokens 包含缓存输入。token 阈值在下一轮调用前检查，单轮可越过阈值。预算耗尽或真实调用失败会记录错误并退出。
- 长窗口默认每 100 个物理步启动一个新 Codex 会话，保留物理状态、整轮预算、当前指导、上一秒真实执行反馈及最近八秒汇总。原语查询和该步动作仍在同一会话完成。`--thread-steps 0` 可在整个窗口使用同一会话，长运行会受到服务端上下文限制。
- 全部记录写入新建的 `cloud-runs/.../episode_000000`。其中 `experiment_protocol.json` 保存运行配置，`steps.jsonl` 保存原始模型输出、实际动作、两种原语调用、执行反馈与模型线程标识。真实入口不会自动回退到 mock。

正式运行前可执行以下检查。mock 以确定性动作验证本地 3,600 步与事件账本，无需模型。smoke 是两步真实模型接口测试。

```sh
PYTHONPATH=src .venv/bin/python -m experiments.gpt6_sol_flexdc_dynamic.harness.run \
  --mode mock --steps 3600 --output cloud-runs/mock-3600

PYTHONPATH=src .venv/bin/python -m experiments.gpt6_sol_flexdc_dynamic.harness.run \
  --mode smoke --steps 2 --codex-home "$FLEXDC_CODEX_HOME" \
  --output cloud-runs/codex-smoke
```
