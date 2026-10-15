# EM3DFold

[English](README.md) | zh-CN

<div align="center">
<!--
<img src="assets/logo.png" alt="EM3DFold：基于冷冻电镜密度图的 AI 驱动原子结构建模" width="600" />
-->
<img src="assets/logo.png" alt="EM3DFold：基于冷冻电镜密度图的 AI 驱动原子结构建模" />
</div>


## 简介

EM3DFold 是一款从冷冻电镜密度图中自动构建蛋白质、RNA 和 DNA 结构的软件。

<!--
<p align="center">
  <img src="assets/header.png" alt="EM3DFold header" width=720" />
</p>
-->

## 系统要求

**平台**：Linux。

> [!NOTE]
> 如需在 Windows 上使用 EM3DFold，可以通过 [Windows WSL](https://learn.microsoft.com/en-us/windows/wsl/install) 安装 Linux 子系统。

**GPU**：需要至少配备 12 GB 显存的 GPU。

**CUDA**：需要 CUDA >= 11.8。

**磁盘空间**：EM3DFold 预训练权重和序列语言模型权重至少需要 6 GB 可用磁盘空间。

## 使用 AI Agent 安装

向 AI 编程 Agent 提供终端访问权限，然后复制以下提示词，并按需替换路径。

```text
1. 请从 https://github.com/huang-laboratory/EM3DFold 安装 EM3DFold，并完整测试一个任务。
2. 请创建一个兼容 Python 软件包版本的全新隔离环境，不要修改系统 CUDA/驱动，也不要覆盖已有环境和结果。
3. 请下载预训练权重、验证软件包导入是否正常，然后运行 README 中的一个示例，并报告命令、环境、运行时间、GPU 峰值显存和输出结果。
```

## 安装

#### 1. 下载 EM3DFold

推荐使用 `git` 下载 EM3DFold：

```bash
git clone https://github.com/huang-laboratory/EM3DFold.git
cd EM3DFold
```

如果不使用 git，可以通过 `wget` 下载源码压缩包并手动解压，例如：

```bash
wget https://github.com/huang-laboratory/EM3DFold/archive/refs/heads/main.zip
unzip main.zip
cd EM3DFold-main
```

> [!NOTE]
> 下载 EM3DFold 后，请选择一种安装方式：**使用 Conda 安装**或**使用 Mamba 安装**。无需同时运行两套安装流程。

#### 2A. 使用 Conda 安装

如果系统中没有 `conda`，请先安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)、Miniforge 或 Anaconda。然后在 EM3DFold 目录中创建 `em3dfold` 环境并安装 EM3DFold：

```bash
bash scripts/install.sh
conda activate em3dfold
```

下载全部预训练权重，并将权重位置保存到 Conda 环境中：

```bash
# 将此路径替换为所需的权重保存目录。
bash scripts/download.sh /path/to/save/pretrained/weights/

# 重新加载环境，使 EM_WEIGHTS_DIR 生效。
conda deactivate
conda activate em3dfold
echo "$EM_WEIGHTS_DIR"
em3dfold --help
```

#### 2B. 使用 Mamba 安装

如果系统中没有 Mamba，请先[安装 Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)。Mamba 使用与 Conda 相同的 `env.yml`，通常可以更快地解析环境依赖。

如果 `mamba` 已位于 `PATH` 中，请运行：

```bash
bash scripts/install_with_mamba.sh
bash scripts/download_with_mamba.sh /path/to/save/pretrained/weights/
mamba run -n em3dfold em3dfold --help
```

如果 Mamba 不在 `PATH` 中，请始终提供其绝对路径：

```bash
export MAMBA_EXE=/path/to/your/mamba
bash scripts/install_with_mamba.sh
bash scripts/download_with_mamba.sh /path/to/save/pretrained/weights/
"$MAMBA_EXE" run -n em3dfold em3dfold --help
```

两个 Mamba 脚本也支持指定明确的环境目录：

```bash
export EM3DFOLD_ENV_PREFIX=/path/to/environment
bash scripts/install_with_mamba.sh
bash scripts/download_with_mamba.sh /path/to/save/pretrained/weights/
"${MAMBA_EXE:-mamba}" run --prefix "$EM3DFOLD_ENV_PREFIX" em3dfold --help
```

> [!NOTE]
> 如果你熟悉 Conda/Mamba/pip/shell，或者运行脚本时遇到问题，可以逐步执行 `install.sh` 或 `install_with_mamba.sh` 中的命令。如果下载失败，请手动下载权重。如果脚本没有成功设置 `EM_WEIGHTS_DIR`，请手动设置：
> ```bash
> conda env config vars set EM_WEIGHTS_DIR=/path/to/save/pretrained/weights/ -n em3dfold
> mamba env config vars set EM_WEIGHTS_DIR=/path/to/save/pretrained/weights/ -n em3dfold
> ```

## 更新到最新版本

如果已经安装 EM3DFold，并希望更新到最新版本，请运行：

```bash
cd /path/to/EM3DFold/
git pull
conda activate em3dfold
pip install .
```

使用 Mamba 时，可以不激活环境，直接用下面的命令替代最后两条命令：

```bash
mamba run -n em3dfold python -m pip install .
```

如果 `env.yml` 和预训练权重发生变化，我们会在此处补充说明。

## 使用方法

运行 EM3DFold 只需一条命令，例如：

```bash
em3dfold build --map/-m MAP.mrc \
    --protein/-p PROTEIN.fa \
    --rna/-r RNA.fa \
    --dna/-d DNA.fa \
    --output OUT \
    --device 0
```

EM3DFold 默认通过环境变量 `EM_WEIGHTS_DIR` 查找预训练模型权重，该变量应已在上述安装步骤中设置。也可以在每次运行 EM3DFold 时显式指定权重目录：

```bash
EM_WEIGHTS_DIR=/path/to/weights em3dfold build --map ...
```

- 冷冻电镜密度图和输出目录为**必需参数**。
- 每个输入 FASTA 文件均可包含多条序列。
- 可以只提供 `--protein`、`--rna` 或 `--dna`，也可以提供它们的任意有效组合。
- 如果同时启动多个建模任务，每次运行的输出目录**必须**不同。
- GPU 设备控制：
  - `--device 0` 表示使用 `cuda:0`。
  - 支持多 GPU，可使用逗号分隔的列表，例如 `--device 0,1,2,3`。
  - 目前仅在 `CA/C4' prediction` 和 `denovo modeling` 阶段使用多 GPU 加速。
  - 使用更多 GPU 通常可以缩短运行时间，但加速效果并非线性增长。
- 默认会删除中间结果，包括预测的原子概率图和循环优化结构。如果希望保留这些结果，请使用 `--keep-temp-files`。

典型的输出目录结构如下：

```text
OUTPUT_DIR/
├── run.log
├── output.cif
├── output_denovo.cif
├── output_denovo_entropy_scores.cif
└── temp/  # 仅在使用 --keep-temp-files 时保留
    ├── format_map.mrc # 格式化后的输入密度图
    ├── pred/ # 预测的原子图
    ├── denovo/ # 从头建模临时文件
    ├── ...
```

- `run.log`：详细运行日志。
- `output.cif`：最终模型，在当前版本的EM3DFold中与output_denovo.cif一致。
- `output_denovo.cif`：从头建模的模型。
- `output_denovo_entropy_scores.cif`：包含氨基酸熵分数的从头建模模型。

**如果忘记如何运行 EM3DFold 时，可以随时查看命令帮助：**

```bash
em3dfold build --help
```

## 示例

<details>
<summary>蛋白质从头建模</summary>
<br>

```bash
em3dfold build --map MAP.mrc \
  --protein protein.fa \
  --output out_protein \
  --device 0
```

下面演示如何从 PDB/EMDB 下载目标蛋白质的序列和密度图，并使用 EM3DFold 对其进行建模：

```bash
# 下载序列
wget https://www.rcsb.org/fasta/entry/9UO1 -O 9UO1.fa
# 下载密度图
wget https://files.wwpdb.org/pub/emdb/structures/EMD-64369/map/emd_64369.map.gz -O emd_64369.map.gz
gunzip emd_64369.map.gz
```

```bash
# 运行 EM3DFold
em3dfold build --map emd_64369.map --protein 9UO1.fa -o 9UO1
```

<table align="center">
    <tr>
        <td align="center">
            <p>输入密度图</p>
            <img src="assets/9UO1_map.jpg" width="400" />
        </td>
        <td align="center">
            <p>输出模型（蓝色）和 PDB 模型（绿色）</p>
            <img src="assets/9UO1_model.jpg" width="400" />
        </td>
    </tr>
</table>

</details>



<details>
<summary>RNA 从头建模</summary>
<br>

```bash
em3dfold build --map MAP.mrc \
  --rna rna.fa \
  --output out_rna \
  --device 0
```

下面演示如何从 PDB/EMDB 下载目标 RNA 的序列和密度图，并使用 EM3DFold 对其进行建模：

```bash
# 下载序列
wget https://www.rcsb.org/fasta/entry/8T5O -O 8T5O.fa
# 下载密度图
wget https://files.wwpdb.org/pub/emdb/structures/EMD-41059/map/emd_41059.map.gz -O emd_41059.map.gz
gunzip emd_41059.map.gz
```

```bash
# 运行 EM3DFold
em3dfold build --map emd_41059.map --rna 8T5O.fa -o 8T5O
```

<table align="center">
    <tr>
        <td align="center">
            <p>输入密度图</p>
            <img src="assets/8T5O_map.jpg" width="400" />
        </td>
        <td align="center">
            <p>输出模型（蓝色）和 PDB 模型（绿色）</p>
            <img src="assets/8T5O_model.jpg" width="400" />
        </td>
    </tr>
</table>

</details>




<details>
<summary>蛋白质-核酸复合物从头建模</summary>
<br>

```bash
em3dfold build --map MAP.mrc \
  --protein protein.fa \
  --rna rna.fa \
  --dna dna.fa \
  --output out_complex \
  --device 0
```

下面演示如何从 PDB/EMDB 下载目标蛋白质、RNA、DNA 的序列和密度图，并使用 EM3DFold 对其进行建模：

```bash
# 下载序列
wget https://www.rcsb.org/fasta/entry/8Y9N -O 8Y9N.fa
# 将序列拆分为蛋白质、RNA 和 DNA 三部分
: > 8Y9N_prot.fa && : > 8Y9N_rna.fa && : > 8Y9N_dna.fa && awk 'BEGIN{prot="8Y9N_prot.fa";rna="8Y9N_rna.fa";dna="8Y9N_dna.fa"} /^>/{out=($0~/\|crRNA\|/)?rna:(($0~/\|DNA/)? dna:prot)} {if(out!="") print >> out}' 8Y9N.fa
# 下载密度图
wget https://files.wwpdb.org/pub/emdb/structures/EMD-39084/map/emd_39084.map.gz -O emd_39084.map.gz
gunzip emd_39084.map.gz
```

```bash
# 运行 EM3DFold
em3dfold build --map emd_39084.map --protein 8Y9N_prot.fa --rna 8Y9N_rna.fa --dna 8Y9N_dna.fa -o 8Y9N
```

<table align="center">
    <tr>
        <td align="center">
            <p>输入密度图</p>
            <img src="assets/8Y9N_map.jpg" width="400" />
        </td>
        <td align="center">
            <p>输出模型（蓝色）和 PDB 模型（绿色）</p>
            <img src="assets/8Y9N_model.jpg" width="400" />
        </td>
    </tr>
</table>

</details>




<details>
<summary>无序列建模</summary>
<br>

```bash
em3dfold build --map MAP.mrc \
  -o out_no_seq \
  --device 0
```

默认情况下，EM3DFold 会构建所有蛋白质和核酸结构。如果只希望构建蛋白质或核酸部分，可以使用另一个提供更多选项的子程序 `build_no_seq`。

```bash
# build_no_seq 提供更多选项
em3dfold build_no_seq --map MAP.mrc \
  -o out_no_seq \
  --protein \ # 构建蛋白质部分
  --rna \ # 构建核酸部分
  --dna \ # 构建核酸部分
  --device 0
```

下面演示如何从 EMDB 下载目标密度图，并使用 EM3DFold 进行无序列建模：

```bash
# 下载密度图（PDB-9Z2N，EMD-73772）
wget https://files.wwpdb.org/pub/emdb/structures/EMD-73772/map/emd_73772.map.gz -O emd_73772.map.gz
gunzip emd_73772.map.gz
```

```bash
# 使用 build 或 build_no_seq 运行 EM3DFold
em3dfold build --map emd_73772.map -o 73772
```

<table align="center">
    <tr>
        <td align="center">
            <p>输入密度图</p>
            <img src="assets/73772_map.jpg" width="400" />
        </td>
        <td align="center">
            <p>输出模型（蓝色）和 PDB 模型（绿色）</p>
            <img src="assets/73772_model.jpg" width="400" />
        </td>
    </tr>
</table>

</details>




<details>
<summary>从冷冻电镜密度图中鉴定未知蛋白质/核酸</summary>
<br>

首先，对输入密度图启动一个无序列从头建模任务：

```bash
em3dfold build --map MAP.mrc -o out_no_seq
```

随后，运行 EM3DFold 的 `hmm_search` 子程序，从序列数据库中搜索可能的序列：

```bash
em3dfold hmm_search \
  --input-dir out_no_seq \ # 必须与无序列建模任务的输出目录相同
  --protein-fasta protein_db.fa \ # 包含所有候选序列
  --na-fasta na_db.fa \ # 包含所有候选序列
  --out out_dir
```

每个片段的全部匹配序列和最佳匹配序列分别保存在 `<out_dir>/all_hits.tsv` 和 `<out_dir>/best_hits.tsv` 中。

下面演示如何使用 EMD-73772 在 Homo sapiens 序列数据库中搜索可能的序列：

```bash
# 下载 Homo sapiens 序列数据库
wget http://huanglab.phys.hust.edu.cn/EM3DFold/examples/73772/homo_sapiens_protein.fa -O homo_sapiens_protein.fa
wget http://huanglab.phys.hust.edu.cn/EM3DFold/examples/73772/homo_sapiens_rna.fa -O homo_sapiens_rna.fa
# 对之前无序列建模的输出目录运行 hmm_search，例如 73772
em3dfold hmm_search --input-dir 73772 --protein-fasta homo_sapiens_protein.fa --na-fasta homo_sapiens_rna.fa -o 73772/hmm_output --max-target-len 10000 # 忽略过长序列
```

这里的 `homo_sapiens_protein.fa` 和 `homo_sapiens_rna.fa` 的原始来源如下：

- 核酸数据来自 NCBI：
  - https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/GRCh38_latest_rna.fna.gz
- 蛋白质数据来自 UniProtKB：
  - https://www.uniprot.org/uniprotkb?query=Homo+sapiens&facets=reviewed%3Atrue

我们提供了该目标的搜索结果，可以通过以下命令查看：

```bash
wget -qO- 'http://huanglab.phys.hust.edu.cn/EM3DFold/examples/73772/best_hits.tsv'
```

结果显示，我们为核酸鉴定出 5 条候选序列：`NR_024457.2`、`NM_001378633.1`、`NR_027232.1`、`XM_011522507.4` 和 `XR_007086624.1`，其中 `XM_011522507.4` （中的某一部分） 是 PDB-9Z2N B 链的正确匹配。

对于蛋白质，我们鉴定出 2 条候选序列：`Q08J23` 和 `P14618`，其中 `Q08J23` 是 PDB-9Z2N A 链的正确匹配。


</details>



## 故障排查

<details>
<summary><code>No module named "xxx"</code></summary>
<br>

当前环境中缺少软件包 `xxx`。请使用 `pip install xxx`、`conda install xxx` 或 `mamba install xxx` 安装。

</details>

<details>
<summary>Conda/Mamba 依赖解析冲突</summary>
<br>

如果 Conda 或 Mamba 无法一次性解析并安装 [`env.yml`](env.yml) 中的全部依赖，可以按以下顺序尝试：

1. **分阶段安装并定位冲突。** 创建全新的 `em3dfold` 环境并安装 `env.yml` 指定的 Python 版本，然后先安装 Conda 依赖，再安装 pip 依赖。如果 Conda 依赖仍然无法统一解析，可以将其分成较小的批次或逐个安装，以确定具体发生冲突的软件包。请勿继续使用依赖解析失败后留下的环境。

2. **最小化调整冲突软件包的版本。** 根据 Conda/Mamba 的错误信息，只修改发生冲突的软件包，并优先选择最接近项目指定版本的兼容版本。例如，项目固定使用 `pillow==10.0.1`，该版本已在我们的测试环境中成功安装；如果求解器无法安装它，请先检查 channel 配置和优先级、操作系统及 CPU 架构，以及本地软件包元数据是否过期，再考虑调整 Pillow 版本。

修改依赖版本后，请记录所有改动，并运行 `pip check`、关键软件包导入测试以及 `em3dfold --help`，确认环境仍然完整可用。如果问题仍然存在，请提交 issue，并附上操作系统、CPU 架构、Conda/Mamba 版本、channel 配置、完整安装命令、完整的依赖解析错误信息以及所有依赖改动。

</details>

<details>
<summary><code>em3dfold: command not found</code></summary>
<br>

当前可能未进入正确的环境。请激活 EM3DFold 环境后重试。

</details>

<details>
<summary>CUDA 或 PyTorch 错误</summary>
<br>

请检查已安装的 `torch` 是否与本地 CUDA 运行时和 GPU 驱动兼容。

</details>

<details>
<summary>GLIBCXX 或 CXXABI 错误</summary>
<br>

这通常表示当前环境中的运行时 `libstdc++` 版本低于所需版本。一种常见的解决方法是将 EM3DFold 环境的 `lib/` 目录添加到 `$LD_LIBRARY_PATH`：

```bash
export LD_LIBRARY_PATH=/path/to/em3dfold/environment/lib:$LD_LIBRARY_PATH
```

激活 `em3dfold` 环境后，可以使用以下命令确定环境的`lib/`路径：

```bash
echo $CONDA_PREFIX
```

</details>

## 引用

如果 EM3DFold 对你的研究有所帮助，请引用以下论文。

```bibtex
@article{EM3DFold,
  title={Highly accurate protein-nucleic acid modeling from cryo-EM maps with EM3DFold},
  author={Tao Li, Sheng-You Huang},
  journal={In submission},
  year={2026}
}

@article{EMProt,
  title={EMProt improves structure determination from cryo-EM maps},
  author={Tao Li, Ji Chen, Hao Li, Hong Cao & Sheng-You Huang},
  journal={Nature Structural & Molecular Biology},
  year={2025}
}

@article{EM2NA,
  #title={Automated detection and de novo structure modeling of nucleic acids from cryo-EM maps},
  author={Tao Li, Hong Cao, Jiahua He & Sheng-You Huang},
  journal={Nature Communications},
  year={2024}
}
```
