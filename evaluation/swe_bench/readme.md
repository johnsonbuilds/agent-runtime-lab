# SWE-bench Evaluation

## 概述

基于 SWE-bench Verified 数据集的代码修复评估。Harbor 在 agent 阶段结束后自动运行 `tests/test.sh` 验证，reward=1 表示 resolved（F2P 全过 + P2P 全过）。

## 快速开始

### 1. 拉取 Docker 镜像（首次）

```bash
for id in $(python3 -c "import json;print(' '.join(json.load(open('evaluation/swe_bench/smoke-10.json'))['instance_ids']))"); do
  docker pull swebench/sweb.eval.x86_64.$(echo $id | sed 's/__/_1776_/'):latest
done
```

### 2. 运行单个任务（测试）

```bash
# 运行单个任务（使用 code-v1 harness，包含文件编辑工具）
uv run harbor run -p evaluation/swe_bench/dataset-smoke10/astropy__astropy-14309 \
  --agent agent_runtime.integrations.harbor:HarborAgent \
  --config harnesses/code-v1.yaml -y

# 查看结果
cat jobs/<job-id>/result.json
cat jobs/<job-id>/<task-id>/verifier/rewards.json
```

### 3. 运行全部任务

```bash
# code-v1 臂（包含 run_command, read_file, apply_patch）
AGENT_RUNTIME_HARNESS=code-v1 uv run harbor run \
  -p evaluation/swe_bench/dataset-smoke10 \
  --agent agent_runtime.integrations.harbor:HarborAgent

# meta-v12 臂（更多工具）
AGENT_RUNTIME_HARNESS=meta-v12 uv run harbor run \
  -p evaluation/swe_bench/dataset-smoke10 \
  --agent agent_runtime.integrations.harbor:HarborAgent
```

### 4. 限制任务数量

```bash
# 只跑前 3 个任务
uv run harbor run -p evaluation/swe_bench/dataset-smoke10 \
  --agent agent_runtime.integrations.harbor:HarborAgent \
  --config harnesses/code-v1.yaml -l 3
```

## 任务列表（smoke-10）

| Instance ID | Repo | 
|-------------|------|
| astropy__astropy-14309 | astropy/astropy |
| django__django-13449 | django/django |
| sphinx-doc__sphinx-11510 | sphinx-doc/sphinx |
| pylint-dev__pylint-6903 | pylint-dev/pylint |
| psf__requests-1921 | psf/requests |
| matplotlib__matplotlib-26342 | matplotlib/matplotlib |
| pytest-dev__pytest-5809 | pytest-dev/pytest |
| scikit-learn__scikit-learn-14053 | scikit-learn/scikit-learn |
| pydata__xarray-6938 | pydata/xarray |
| sympy__sympy-19495 | sympy/sympy |

## 目录结构

```
evaluation/swe_bench/
├── readme.md              # 本文件
├── smoke-10.json          # 10 个任务 ID 列表
├── dev-50.json            # 50 个开发任务
├── build_swe_dataset.py   # 构建数据集脚本
├── select_swe_tasks.py    # 选择任务脚本
└── dataset-smoke10/       # 任务目录
    └── <task-id>/
        ├── task.toml          # 任务配置（Docker 镜像、超时）
        ├── instruction.md     # 任务描述（GitHub issue）
        ├── solution/solve.sh  # 参考解答（apply gold patch）
        ├── tests/
        │   ├── test.sh        # 验证脚本
        │   ├── test_patch.diff
        │   └── gold.patch
        └── environment/README.md
```

## Harness 选择

SWE-bench 任务需要文件编辑工具才能修复代码。推荐使用以下 harness：

| Harness | 工具 | 适用场景 |
|---------|------|----------|
| `baseline-v0` | run_command | 只读分析 |
| `code-v1` | run_command, read_file, apply_patch | **推荐** - 代码修复 |
| `files-v1` | run_command, write_file, read_file, list_dir | 文件操作 |
| `meta-v1` | run_command, write_file, read_file, list_dir, edit_file, apply_patch, grep_search, glob_files | 完整工具集 |

**运行命令示例：**
```bash
# 使用环境变量
AGENT_RUNTIME_HARNESS=code-v1 uv run harbor run ...

# 或使用 --config 参数
uv run harbor run ... --config harnesses/code-v1.yaml
```

## 注意事项

### Docker 镜像

- 镜像名格式：`swebench/sweb.eval.x86_64.<repo_id>:latest`
- Docker Hub 使用 `_1776_` 替代 `__`（如 `astropy_1776_astropy-14309`）
- 每个镜像包含对应 repo 的特定 commit 和依赖环境
- 镜像内有 `testbed` conda 环境（Python + pytest + 依赖）

### 验证机制

- Agent 完成后，Harbor 自动运行 `tests/test.sh`
- 脚本会先 `export PATH=/opt/miniconda3/envs/testbed/bin:$PATH` 使用正确的 Python 环境
- 验证结果写入 `/logs/verifier/reward.txt`（1=resolved, 0=未解决）
- 详细结果在 `/logs/verifier/rewards.json`

### 超时设置

- Agent 超时：3600s（1小时）
- Verifier 超时：1800s（30分钟）
- 环境构建超时：600s（10分钟）

### 常见问题

**Q: 镜像 pull 404？**
A: 检查镜像名是否使用 `_1776_` 格式，或用 `build_swe_dataset.py --image-tag <tag>` 重建。

**Q: 测试报 `No module named pytest`？**
A: 确保 `test.sh` 中有 `export PATH=/opt/miniconda3/envs/testbed/bin:$PATH`。

**Q: numpy 兼容性错误？**
A: 使用 `testbed` conda 环境（numpy 1.25.2），不要用 base 环境。

### 重新生成数据集

```bash
uv run python evaluation/swe_bench/build_swe_dataset.py \
  --list evaluation/swe_bench/smoke-10.json \
  --out evaluation/swe_bench/dataset-smoke10 \
  --parquet <path-to-parquet> \
  --image-tag latest
```

需要 `pyarrow` 读取 Verified parquet 文件。
