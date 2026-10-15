# EM3DFold

English | [zh-CN](README.zh-CN.md)

<div align="center">
<!--
<img src="assets/logo.png" alt="EM3DFold: AI-driven Atomic Structure Modeling from Cryo-EM Maps" width="600" />
-->
<img src="assets/logo.png" alt="EM3DFold: AI-driven Atomic Structure Modeling from Cryo-EM Maps" />
</div>


## Overview
EM3DFold is a software package for automatic protein, RNA, and DNA structure modeling from cryo-EM density maps.

<!--
<p align="center">
  <img src="assets/header.png" alt="EM3DFold header" width=720" />
</p>
-->

## Requirements
**Platform**: Linux.

> [!NOTE]
> To use EM3DFold on Windows, you may install a Linux sub-system via [Windows WSL](https://learn.microsoft.com/en-us/windows/wsl/install).

**GPU**: A GPU with at least 12 GB VRAM is required.

**CUDA**: CUDA >= 11.8 is required.

**Disk Storage**: EM3DFold pretrained weights and language model weights require at least 6 GB free disk space.

## Install with AI agents

Give an AI coding agent terminal access and copy the prompt below, replacing paths when needed.

```text
1. Install EM3DFold from https://github.com/huang-laboratory/EM3DFold and test one complete item.
2. Create a new isolated environment with compatible Python package versions, without changing the system CUDA/driver or overwriting existing environments and results.
3. Download the pretrained weights, verify imports, then run one README example and report commands, environments, runtime, peak GPU memory, and outputs.
```

## Installation
#### 1. Download EM3DFold

Download EM3DFold via `git` (recommended):
```bash
git clone https://github.com/huang-laboratory/EM3DFold.git
cd EM3DFold
```
If you do not use git, download the source archive via `wget` and extract it manually, e.g. with command:
```bash
wget https://github.com/huang-laboratory/EM3DFold/archive/refs/heads/main.zip
unzip main.zip
cd EM3DFold-main
```

> [!NOTE]
> After downloading EM3DFold, choose one installation method: **Install with Conda** or **Install with Mamba**. You do not need to run both workflows.

#### 2A. Install with Conda

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), Miniforge, or Anaconda first if `conda` is not available. Then, from the EM3DFold directory, create the `em3dfold` environment and install EM3DFold:

```bash
bash scripts/install.sh
conda activate em3dfold
```

Download all pretrained weights and save their location in the Conda environment:

```bash
# Replace this path with your desired weights directory.
bash scripts/download.sh /path/to/save/pretrained/weights/

# Reload the environment so EM_WEIGHTS_DIR becomes available.
conda deactivate
conda activate em3dfold
echo "$EM_WEIGHTS_DIR"
em3dfold --help
```

#### 2B. Install with Mamba

Install [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) first if it is not available. Mamba uses the same `env.yml` as Conda and usually resolves the environment faster.

If `mamba` is in `PATH`, run:

```bash
bash scripts/install_with_mamba.sh
bash scripts/download_with_mamba.sh /path/to/save/pretrained/weights/
mamba run -n em3dfold em3dfold --help
```

If Mamba is not in `PATH`, provide its absolute path consistently:

```bash
export MAMBA_EXE=/path/to/your/mamba
bash scripts/install_with_mamba.sh
bash scripts/download_with_mamba.sh /path/to/save/pretrained/weights/
"$MAMBA_EXE" run -n em3dfold em3dfold --help
```

Both Mamba commands also accept an explicit environment prefix:

```bash
export EM3DFOLD_ENV_PREFIX=/path/to/environment
bash scripts/install_with_mamba.sh
bash scripts/download_with_mamba.sh /path/to/save/pretrained/weights/
"${MAMBA_EXE:-mamba}" run --prefix "$EM3DFOLD_ENV_PREFIX" em3dfold --help
```

> [!NOTE]
> If you are familiar with Conda/Mamba/pip/shell or encounter problems with the scripts, you can run the commands in `install.sh` or `install_with_mamba.sh` step-by-step. If downloading fails, download the weights manually. If the script fails to set `EM_WEIGHTS_DIR`, set it manually:
> ```bash
> conda env config vars set EM_WEIGHTS_DIR=/path/to/save/pretrained/weights/ -n em3dfold
> mamba env config vars set EM_WEIGHTS_DIR=/path/to/save/pretrained/weights/ -n em3dfold
> ```

## Update to latest version
If you have already installed EM3DFold and want to get the latest version, run the following commands:
```bash
cd /path/to/EM3DFold/
git pull
conda activate em3dfold
pip install .
```

With Mamba, the last two commands can instead be run without activation:
```bash
mamba run -n em3dfold python -m pip install .
```

If `env.yml` and pretrained weights changed, we will add more guidance here.

## Usage
Running EM3DFold is straight forward with one command like
```bash
em3dfold build --map/-m MAP.mrc \
    --protein/-p PROTEIN.fa \
    --rna/-r RNA.fa \
    --dna/-d DNA.fa \
    --output OUT \
    --device 0
```

EM3DFold reads env variable `EM_WEIGHTS_DIR` by default to find the pretrained model weights, which should be set during the installation steps above. You can also explicitly set the weights directory every time you running EM3DFold by:
```bash
EM_WEIGHTS_DIR=/path/to/weights em3dfold build --map ...
```

- The cryo-EM density map and output directory are **required**.
- Input FASTA files can each include multiple sequences.
- You can provide only `--protein`, only `--rna`, only `--dna`, or any valid combination of them.
- If you launch >1 modeling job, the output directory **MUST** be different for each run.
- GPU device control:
  - `--device 0` means use `cuda:0`.
  - Multiple GPUs are supported with a comma-separated list such as `--device 0,1,2,3`.
  - Multi-GPU acceleration is currently only used in the `CA/C4' prediction` and `denovo modeling` stages. 
  - More GPUs usually make the run faster, but the speedup is not linear.
- By default, intermediate results (predicted atom probability maps and recycled structures) will be removed. Use `--keep-temp-files` if you want to keep them. 

Typical output directory layout:
```text
OUTPUT_DIR/
├── run.log
├── output.cif
├── output_denovo.cif
├── output_denovo_entropy_scores.cif
└── temp/  # only kept with --keep-temp-files
    ├── format_map.mrc # formated input map
    ├── pred/ # predicted atom maps
    ├── denovo/ # denovo modeling temp files
    ├── ...
```

- `run.log`: detailed running logs.
- `output.cif`: final model.
- `output_denovo.cif`: de novo model.
- `output_denovo_entropy_scores.cif`: de novo model with amino-acid entropy scores.

**Check the command usage any time you forget how to run EM3DFold**
```bash
em3dfold build --help
```

## Examples
<details>
<summary>Protein denovo modeling</summary>
<br>

```bash
em3dfold build --map MAP.mrc \
  --protein protein.fa \
  --output out_protein \
  --device 0
```

Below shows how we download a target protein sequence and map from the PDB/EMDB and run EM3DFold modeling on it:
```bash
# download sequence
wget https://www.rcsb.org/fasta/entry/9UO1 -O 9UO1.fa
# download map
wget https://files.wwpdb.org/pub/emdb/structures/EMD-64369/map/emd_64369.map.gz -O emd_64369.map.gz
gunzip emd_64369.map.gz
```
```bash
# run EM3DFold
em3dfold build --map emd_64369.map --protein 9UO1.fa -o 9UO1
```
<table align="center">
    <tr>
        <td align="center">
            <p>The input map</p>
            <img src="assets/9UO1_map.jpg" width="400" />
        </td>
        <td align="center">
            <p>The output model (blue) and the PDB model (green)</p>
            <img src="assets/9UO1_model.jpg" width="400" />
        </td>
    </tr>
</table>

</details>



<details>
<summary>RNA denovo modeling</summary>
<br>

```bash
em3dfold build --map MAP.mrc \
  --rna rna.fa \
  --output out_rna \
  --device 0
```

Below shows how we download a target RNA sequence and map from the PDB/EMDB and run EM3DFold modeling on it:
```bash
# download sequence
wget https://www.rcsb.org/fasta/entry/8T5O -O 8T5O.fa
# download map
wget https://files.wwpdb.org/pub/emdb/structures/EMD-41059/map/emd_41059.map.gz -O emd_41059.map.gz
gunzip emd_41059.map.gz
```
```bash
# run EM3DFold
em3dfold build --map emd_41059.map --rna 8T5O.fa -o 8T5O
```
<table align="center">
    <tr>
        <td align="center">
            <p>The input map</p>
            <img src="assets/8T5O_map.jpg" width="400" />
        </td>
        <td align="center">
            <p>The output model (blue) and the PDB model (green)</p>
            <img src="assets/8T5O_model.jpg" width="400" />
        </td>
    </tr>
</table>

</details>




<details>
<summary>Protein-nucleic acid complex denovo modeling</summary>
<br>

```bash
em3dfold build --map MAP.mrc \
  --protein protein.fa \
  --rna rna.fa \
  --dna dna.fa \
  --output out_complex \
  --device 0
```

Below shows how we download the target protein, RNA, DNA sequences and map from the PDB/EMDB and run EM3DFold modeling on it:
```bash
# download sequence
wget https://www.rcsb.org/fasta/entry/8Y9N -O 8Y9N.fa
# split sequences into protein, rna and dna parts
: > 8Y9N_prot.fa && : > 8Y9N_rna.fa && : > 8Y9N_dna.fa && awk 'BEGIN{prot="8Y9N_prot.fa";rna="8Y9N_rna.fa";dna="8Y9N_dna.fa"} /^>/{out=($0~/\|crRNA\|/)?rna:(($0~/\|DNA/)? dna:prot)} {if(out!="") print >> out}' 8Y9N.fa
# download map
wget https://files.wwpdb.org/pub/emdb/structures/EMD-39084/map/emd_39084.map.gz -O emd_39084.map.gz
gunzip emd_39084.map.gz
```
```bash
# run EM3DFold
em3dfold build --map emd_39084.map --protein 8Y9N_prot.fa --rna 8Y9N_rna.fa --dna 8Y9N_dna.fa -o 8Y9N
```
<table align="center">
    <tr>
        <td align="center">
            <p>The input map</p>
            <img src="assets/8Y9N_map.jpg" width="400" />
        </td>
        <td align="center">
            <p>The output model (blue) and the PDB model (green)</p>
            <img src="assets/8Y9N_model.jpg" width="400" />
        </td>
    </tr>
</table>

</details>




<details>
<summary>Modeling without sequences</summary>
<br>

```bash
em3dfold build --map MAP.mrc \
  -o out_no_seq \
  --device 0
```
By default, EM3DFold builds all protein and nucleic acid structures. We provide another sub-program `build_no_seq` with more options if you want to build only the protein/nucleic acid parts.
```bash
# build_no_seq provides more options
em3dfold build_no_seq --map MAP.mrc \
  -o out_no_seq \
  --protein \ # build the protein parts
  --rna \ # build the nucleic acid parts
  --dna \ # build the nucleic acid parts
  --device 0
```

Below shows how we download the target map from the EMDB and run EM3DFold sequence-free modeling on it:
```bash
# download map (PDB-9Z2N, EMD-73772)
wget https://files.wwpdb.org/pub/emdb/structures/EMD-73772/map/emd_73772.map.gz -O emd_73772.map.gz
gunzip emd_73772.map.gz
```
```bash
# run EM3DFold with build or build_no_seq
em3dfold build --map emd_73772.map -o 73772
```
<table align="center">
    <tr>
        <td align="center">
            <p>The input map</p>
            <img src="assets/73772_map.jpg" width="400" />
        </td>
        <td align="center">
            <p>The output model (blue) and the PDB model (green)</p>
            <img src="assets/73772_model.jpg" width="400" />
        </td>
    </tr>
</table>

</details>




<details>
<summary>Identify unknown protein/nucleic acids from cryo-EM map</summary>
<br>

First, lanuch a sequence-free job on an input map:
```bash
em3dfold build --map MAP.mrc -o out_no_seq
```
Afterwards, run the `hmm_search` sub-program of em3dfold to search the possible sequences out of sequence databases:
```bash
em3dfold hmm_search \
  --input-dir out_no_seq \ # must be the same output dir of the seq-free modeling job
  --protein-fasta protein_db.fa \ # contains all candidate sequences
  --na-fasta na_db.fa \ # contains all candidate sequences
  --out out_dir
```
The all/best matched sequences for each fragment are saved in `<out_dir>/all_hits.tsv` and `<out_dir>/best_hits.tsv`


Below shows how we search the possible sequences of EMD-73772 against *Homo sapiens* sequence databases:
```bash
# download the sequence database of homo sapiens
wget http://huanglab.phys.hust.edu.cn/EM3DFold/examples/73772/homo_sapiens_protein.fa -O homo_sapiens_protein.fa
wget http://huanglab.phys.hust.edu.cn/EM3DFold/examples/73772/homo_sapiens_rna.fa -O homo_sapiens_rna.fa
# run hmm_search on previous sequence-free modeling output directory, e.g. 73772
em3dfold hmm_search --input-dir 73772 --protein-fasta homo_sapiens_protein.fa --na-fasta homo_sapiens_rna.fa -o 73772/hmm_output --max-target-len 10000 # ignore too long sequences
```

The original sources for the prepared `homo_sapiens_protein.fa` and `homo_sapiens_rna.fa`:
- nucleic acid from ncbi
  - https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/GRCh38_latest_rna.fna.gz
- protein from uniprotkb
  - https://www.uniprot.org/uniprotkb?query=Homo+sapiens&facets=reviewed%3Atrue


We provide our searching result on this target, check the result:
```bash
wget -qO- 'http://huanglab.phys.hust.edu.cn/EM3DFold/examples/73772/best_hits.tsv'
```
As shown, we identified 5 candidate sequences: `NR_024457.2`, `NM_001378633.1`, `NR_027232.1`, `XM_011522507.4`, `XR_007086624.1` for nucleic acids, in which `XM_011522507.4` is the correct match of chain B of PDB-9Z2N.

For proteins, we identified 2 protein sequences as candidates: `Q08J23` and `P14618`, in which `Q08J23` is the correct match of chain A of PDB-9Z2N.


</details>



## Trouble shooting
<details>
<summary><code>No module named "xxx"</code></summary>
<br>

Package `xxx` is missing in your current environment. Install it with `pip install xxx`, `conda install xxx`, or `mamba install xxx`.

</details>

<details>
<summary><code>em3dfold: command not found</code></summary>
<br>

You are probably not in the correct environment. Activate the EM3DFold environment and run again.

</details>

<details>
<summary>CUDA or PyTorch errors</summary>
<br>

Check whether your installed `torch` matches the local CUDA runtime and GPU driver.

</details>

<details>
<summary>GLIBCXX or CXXABI errors</summary>
<br>

This usually means the runtime `libstdc++` in your current environment is older than the one required. A common fix is to add the EM3DFold environment `lib/` directory to `$LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/path/to/em3dfold/environment/lib:$LD_LIBRARY_PATH
```

When the `em3dfold` environment is activated, you can find its prefix with:

```bash
echo $CONDA_PREFIX
```

</details>

## Citation
If you find EM3DFold useful, please cite the following papers.

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
