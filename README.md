# pPred 使用说明

## 模型运行

### 获取代码

```bash
git clone https://github.com/pengyaping21-gif/pPred-GNN.git
cd pPred-GNN
```

### 下载 checkpoint

首次运行前，从 [v0.2.0 Release](https://github.com/pengyaping21-gif/pPred-GNN/releases/tag/v0.2.0) 下载权重包，并在项目根目录解压：

```bash
mkdir -p .cache/checkpoints
curl -fL --retry 3 \
  https://github.com/pengyaping21-gif/pPred-GNN/releases/download/v0.2.0/ppred-checkpoints-v0.2.0.tar.gz \
  -o .cache/checkpoints/ppred-checkpoints-v0.2.0.tar.gz
curl -fL --retry 3 \
  https://github.com/pengyaping21-gif/pPred-GNN/releases/download/v0.2.0/ppred-checkpoints-v0.2.0.tar.gz.sha256 \
  -o .cache/checkpoints/ppred-checkpoints-v0.2.0.tar.gz.sha256
(cd .cache/checkpoints && sha256sum -c ppred-checkpoints-v0.2.0.tar.gz.sha256)
tar -xzf .cache/checkpoints/ppred-checkpoints-v0.2.0.tar.gz
```

解压后，三份权重位于 `checkpoints/single_rt.pth`、`checkpoints/crosslink_rt.pth` 和 `checkpoints/ms2.pth`。

### Docker 启动（推荐）

在 Linux x86_64 主机克隆仓库，进入项目根目录，先完成上面的权重下载。完整 `torch1` 镜像包含 Python 3.8、PyTorch 2.4.0 + CUDA 12.1、PyG 2.6.1、RDKit 2024.3.5 及原环境的其余依赖。GPU 运行需要主机安装 NVIDIA 驱动和 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

从 [torch1 runtime Release](https://github.com/pengyaping21-gif/pPred-GNN/releases/tag/torch1-runtime-20261116) 下载完整镜像；脚本会校验分卷并合并：

```bash
bash scripts/download_runtime.sh docker
docker load -i .cache/runtime/ppred-torch1-linux-amd64.docker.tar

# 进入容器，当前项目目录映射到 /app，输出直接保存在主机项目目录
docker run --rm -it --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/app" -w /app ppred-gnn:torch1 bash

# 以下命令在容器中执行
python scripts/verify_bundle.py
```

没有 GPU 时，删除 `--gpus all`，后续命令使用 `--device cpu`。退出容器后，权重、数据和 `outputs/` 仍保存在主机项目目录。请使用 Bash 执行下载脚本；下载、合并和镜像导入建议预留至少 25 GB 空间。

需要用当前代码重新构建镜像时，在主机项目根目录执行：

```bash
bash scripts/download_runtime.sh conda
docker buildx build --platform linux/amd64 \
  --build-context torch1_snapshot=.cache/runtime \
  --load -t ppred-gnn:torch1 .
```

### Conda 启动（可选）

已有环境可直接使用 `conda activate XXX/torch1`。需要从完整备份恢复时，在 Linux x86_64 主机项目根目录执行：

```bash
bash scripts/download_runtime.sh conda
mkdir -p .cache/envs/torch1
tar -xzf .cache/runtime/torch1-conda-linux-amd64.tar.gz -C .cache/envs/torch1
.cache/envs/torch1/bin/python .cache/envs/torch1/bin/conda-unpack
source .cache/envs/torch1/bin/activate
python scripts/verify_bundle.py
```

解压后的环境完成 `conda-unpack` 后，请勿再次移动；需要换路径时重新从压缩包解压。下列命令均在已启动环境的项目根目录执行；没有 GPU 时使用 `--device cpu`。

### 一键运行 demo

```bash
# 三项任务：评估 → 微调 → 测试 → 预测
python scripts/run_demo.py --task all --device cuda:0 --epochs 1

# 只运行一项
python scripts/run_demo.py --task single_rt --device cuda:0 --epochs 1
python scripts/run_demo.py --task crosslink_rt --device cuda:0 --epochs 1
python scripts/run_demo.py --task ms2 --device cuda:0 --epochs 1
```

结果写入 `outputs/demo_<时间戳>/`；微调权重为各任务目录下的 `finetune/best.pth`。

### 微调

```bash
# 单肽 RT
python -m ppred finetune --task single_rt \
  --checkpoint checkpoints/single_rt.pth \
  --train demo/single_rt/train.rtdata \
  --val demo/single_rt/val.rtdata \
  --test demo/single_rt/test.rtdata \
  --epochs 3 --lr 1e-5 --batch-size 8 \
  --device cuda:0 --output outputs/single_rt

# 交联 RT
python -m ppred finetune --task crosslink_rt \
  --checkpoint checkpoints/crosslink_rt.pth \
  --train demo/crosslink_rt/train.rtdata \
  --val demo/crosslink_rt/val.rtdata \
  --test demo/crosslink_rt/test.rtdata \
  --epochs 3 --lr 1e-5 --batch-size 8 \
  --device cuda:0 --output outputs/crosslink_rt

# 单肽 MS2
python -m ppred finetune --task ms2 \
  --checkpoint checkpoints/ms2.pth \
  --train demo/ms2/train.mgf \
  --val demo/ms2/val.mgf \
  --test demo/ms2/test.mgf \
  --epochs 3 --lr 1e-5 --batch-size 8 \
  --device cuda:0 --output outputs/ms2
```

替换 `--train`、`--val`、`--test` 即可使用自己的数据，`--test` 可省略。训练与验证、测试之间不能有相同肽骨架；同一序列的不同修饰、charge、NCE 应放在同一分区，交联两链交换也视为同组。

微调后使用 `<output>/best.pth`。同任务 RT 默认沿用权重中的归一化尺度；换到新的实验域时，可加 `--rt-normalization train`，用新训练集拟合尺度。

### 评估

```bash
python -m ppred evaluate --task single_rt \
  --checkpoint outputs/single_rt/best.pth \
  --input demo/single_rt/test.rtdata \
  --device cuda:0 --output outputs/single_rt/evaluation.json

python -m ppred evaluate --task crosslink_rt \
  --checkpoint outputs/crosslink_rt/best.pth \
  --input demo/crosslink_rt/test.rtdata \
  --device cuda:0 --output outputs/crosslink_rt/evaluation.json

python -m ppred evaluate --task ms2 \
  --checkpoint outputs/ms2/best.pth \
  --input demo/ms2/test.mgf \
  --device cuda:0 --output outputs/ms2/evaluation.json
```

### 预测

```bash
python -m ppred predict --task single_rt \
  --checkpoint checkpoints/single_rt.pth \
  --input demo/single_rt/test.rtdata \
  --device cuda:0 --output outputs/single_rt_predictions.csv

python -m ppred predict --task crosslink_rt \
  --checkpoint checkpoints/crosslink_rt.pth \
  --input demo/crosslink_rt/test.rtdata \
  --device cuda:0 --output outputs/crosslink_rt_predictions.csv

python -m ppred predict --task ms2 \
  --checkpoint checkpoints/ms2.pth \
  --input demo/ms2/test.mgf \
  --device cuda:0 --output outputs/ms2_predictions.jsonl
```

使用微调后的模型时，将 `--checkpoint` 换成相应的 `outputs/<task>/best.pth`。RT 预测保存为 CSV，MS2 预测保存为 JSONL。需要覆盖已有结果时加 `--overwrite`。

## 数据集输入格式

### 单肽 RT

使用 Tab 分隔的 `.rtdata` 文件，第一行为列名：

```text
seq	mod	rt
ACDK	2,Carbamidomethyl[C];	45.2
PEPTIDEK		62.8
```

- `seq`：20 种标准氨基酸组成的序列，长度不超过 60。
- `mod`：修饰，例如 `2,Carbamidomethyl[C];`；位点从 1 开始，0 表示 N 端，多个修饰用分号分隔，无修饰留空。
- `rt`：数值型 RT 标签，同一数据集保持单位一致。预测时可省略此列。

### 交联 RT

使用 Tab 分隔的 `.rtdata` 文件：

```text
seq	mod	linker	rt
AAKAA(3)-ACKAA(3)	10,Carbamidomethyl[C];	DSSO	75.3
AAKAA(3)-AAKAA(3)		DSSO	68.1
```

- `seq`：格式为 `ALPHA(连接位点)-BETA(连接位点)`，位点从 1 开始，两链合计长度不超过 60。
- `mod`：支持 `position,Name;` 或 `Name(position);`，无修饰留空。α 链使用链内位点；β 链使用 `len(alpha) + 3 + β链内位点`。上例 β 链第 2 位 C 的修饰位置为 10。
- `linker`：填写 `DSSO`，交联位点为 K-K。
- `rt`：数值型 RT 标签。预测时可省略此列。

修饰名称须与 [修饰定义文件](ppred/resources/modification_smiles_new.ini) 一致，且修饰位点应匹配相应残基。

### 单肽 MS2

使用 MGF 文件，每条谱以 `BEGIN IONS` 开始、`END IONS` 结束。以下为字段格式示例：

```text
BEGIN IONS
TITLE=example_1
SEQ=ACDK@2,Carbamidomethyl[C];@2@27
CHARGE=2+
NCE=27
INSTRUMENT_NAME=QE
ACTIVATION_TYPE=HCD
INSTRUMENT_TYPE=FT
100.0 25.0 b1-z1
200.0 50.0 y2-z2
250.0 100.0 Unknown
END IONS
```

| 字段 | 填写方式 |
| --- | --- |
| `SEQ` | `序列@修饰@charge@NCE`，无修饰时对应部分留空；肽长 2–60 |
| `CHARGE` | 整数电荷，可带末尾 `+`，范围 1–10 |
| `NCE` | 碰撞能量，范围 0–100 |
| `INSTRUMENT_NAME` | `QE`、`Velos`、`Elite`、`Fusion` 或 `Lumos` |
| `ACTIVATION_TYPE` | `HCD` 或 `CID` |
| `INSTRUMENT_TYPE` | `FT` 或 `IT` |
| 峰行 | `m/z intensity annotation` |

峰标注支持 `b1-z1`、`b2-z2`、`y1-z1`、`y2-z2` 等，以及 `b2-ModLoss-z1` 形式的修饰丢失峰；未匹配峰可写 `Unknown`。训练和评估必须包含正强度的 b/y 标注峰，预测时可省略实验峰，但仍需填写序列、修饰及仪器条件。

完整输入样例见 `demo/single_rt/`、`demo/crosslink_rt/` 和 `demo/ms2/`。
