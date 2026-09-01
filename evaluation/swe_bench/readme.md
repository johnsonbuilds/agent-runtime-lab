**跟 TB 一模一样，只换 `-d`**。TB2 的 `-d terminal-bench/terminal-bench-2` 是注册表数据集，SWE 没有现成的，所以我把 Verified 转成了**本地数据集目录**（harbor 0.21 实证：`-d` 直接收本地目录，自动发现子目录任务，无需 dataset.toml）。

**跑完它自己 verify 吗？——是。** Harbor 在 agent 阶段结束后自动跑每个任务的 `tests/test.sh`，脚本写 `/logs/verifier/reward.txt`（1=resolved），result.json 里直接读 reward。我给 10 个任务各配了官方 eval 镜像（swebench/sweb.eval.x86_64.<id>）+ 按 repo 分流的真 verifier（pytest 族 8 个 / django runtests / sympy bin/test），判 F2P 全过 + P2P 全过才给 1。
**你机器上三步**（文件都在刚发的 swe_bench/ 里，按原路径拷进 repo）：
```bash
# 0) 先拉镜像（首次，验证 latest tag 存在）
for id in $(python3 -c "import json;print(' '.join(json.load(open('evaluation/swe_bench/smoke-10.json'))['instance_ids']))"); do docker pull swebench/sweb.eval.x86_64.$(echo $id | sed 's/__/_1776_/'):latest; done

# 1) 两臂（建议先加 -l 1 单任务试跑一条）
AGENT_RUNTIME_HARNESS=code-v1  uv run harbor run -d evaluation/swe_bench/dataset-smoke10 --agent agent_runtime.integrations.harbor:HarborAgent
AGENT_RUNTIME_HARNESS=meta-v12 uv run harbor run -d evaluation/swe_bench/dataset-smoke10 --agent agent_runtime.integrations.harbor:HarborAgent
```
若某镜像 pull 404，换 `build_swe_dataset.py --image-tag <tag>` 重建即可。数据集也能用 `scripts/build_swe_dataset.py` 随时重生成（确定性，输入=smoke-10.json+Verified parquet）。