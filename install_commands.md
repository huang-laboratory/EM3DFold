# Step-by-step installation commands

This file lists the commands used for a first-time EM3DFold installation. Choose either the Conda section or the Mamba section, not both. Run the commands from the EM3DFold repository directory in the same shell so that the shell variables remain available.

For an installation that was cancelled, interrupted, or failed, rerunning `scripts/install.sh` or `scripts/install_with_mamba.sh` is recommended because those scripts can reuse an existing environment and cached downloads.

## Install with Conda

```bash
ENV_NAME=em3dfold

conda create --yes --name "$ENV_NAME" python=3.10.14 pip=23.3.1

conda install --yes --name "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -c defaults numpy==1.26.2
conda install --yes --name "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -c defaults numba==0.59.1
conda install --yes --name "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -c defaults scipy==1.11.4
conda install --yes --name "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -c defaults tqdm==4.66.5
conda install --yes --name "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -c defaults biopython==1.84
conda install --yes --name "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -c defaults einops==0.8.0
conda install --yes --name "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -c defaults mrcfile==1.4.3
conda install --yes --name "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -c defaults pillow==10.0.1
conda install --yes --name "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -c defaults omegaconf==2.3.0

conda run --name "$ENV_NAME" python -m pip install setuptools==68.2.2
conda run --name "$ENV_NAME" python -m pip install 'https://download.pytorch.org/whl/cu118/torch-2.1.2%2Bcu118-cp310-cp310-linux_x86_64.whl'
conda run --name "$ENV_NAME" python -m pip install 'https://download.pytorch.org/whl/cu118/torchvision-0.16.2%2Bcu118-cp310-cp310-linux_x86_64.whl'
conda run --name "$ENV_NAME" python -m pip install 'https://download.pytorch.org/whl/cu118/torchaudio-2.1.2%2Bcu118-cp310-cp310-linux_x86_64.whl'
conda run --name "$ENV_NAME" python -m pip install cupy-cuda11x==13.6.0
conda run --name "$ENV_NAME" python -m pip install fair-esm==2.0.0
conda run --name "$ENV_NAME" python -m pip install pyhmmer==0.11.0
conda run --name "$ENV_NAME" python -m pip install ortools==9.4.1874
conda run --name "$ENV_NAME" python -m pip install pydssp==0.9.0
conda run --name "$ENV_NAME" python -m pip install ninja==1.13.0
conda run --name "$ENV_NAME" python -m pip install PySocks==1.7.1
conda run --name "$ENV_NAME" python -m pip install absl-py==2.3.1
conda run --name "$ENV_NAME" python -m pip install beautifulsoup4==4.14.2
conda run --name "$ENV_NAME" python -m pip install gdown==5.2.0
conda run --name "$ENV_NAME" python -m pip install ml_collections==1.1.0
conda run --name "$ENV_NAME" python -m pip install soupsieve==2.8

FLASH_ABI="$(conda run --name "$ENV_NAME" python -c 'import torch; print(torch.compiled_with_cxx11_abi())')"
FLASH_ABI="${FLASH_ABI^^}"
FLASH_WHEEL="flash_attn-2.3.2+cu118torch2.1cxx11abi${FLASH_ABI}-cp310-cp310-linux_x86_64.whl"
wget --continue "https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.2/${FLASH_WHEEL}" -O "flash_attn_whl/${FLASH_WHEEL}"
conda run --name "$ENV_NAME" python -m pip install "flash_attn_whl/${FLASH_WHEEL}"

conda run --name "$ENV_NAME" python -m pip install .

ENV_MANAGER=conda
```

Continue with [Download weights and verify the installation](#download-weights-and-verify-the-installation).

## Install with Mamba

If Mamba is not in `PATH`, set `MAMBA_EXE` to its absolute path.

```bash
MAMBA_EXE=mamba
ENV_NAME=em3dfold

"$MAMBA_EXE" create --yes --name "$ENV_NAME" python=3.10.14 pip=23.3.1

"$MAMBA_EXE" install --yes --name "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -c defaults numpy==1.26.2
"$MAMBA_EXE" install --yes --name "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -c defaults numba==0.59.1
"$MAMBA_EXE" install --yes --name "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -c defaults scipy==1.11.4
"$MAMBA_EXE" install --yes --name "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -c defaults tqdm==4.66.5
"$MAMBA_EXE" install --yes --name "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -c defaults biopython==1.84
"$MAMBA_EXE" install --yes --name "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -c defaults einops==0.8.0
"$MAMBA_EXE" install --yes --name "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -c defaults mrcfile==1.4.3
"$MAMBA_EXE" install --yes --name "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -c defaults pillow==10.0.1
"$MAMBA_EXE" install --yes --name "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -c defaults omegaconf==2.3.0

"$MAMBA_EXE" run --name "$ENV_NAME" python -m pip install setuptools==68.2.2
"$MAMBA_EXE" run --name "$ENV_NAME" python -m pip install 'https://download.pytorch.org/whl/cu118/torch-2.1.2%2Bcu118-cp310-cp310-linux_x86_64.whl'
"$MAMBA_EXE" run --name "$ENV_NAME" python -m pip install 'https://download.pytorch.org/whl/cu118/torchvision-0.16.2%2Bcu118-cp310-cp310-linux_x86_64.whl'
"$MAMBA_EXE" run --name "$ENV_NAME" python -m pip install 'https://download.pytorch.org/whl/cu118/torchaudio-2.1.2%2Bcu118-cp310-cp310-linux_x86_64.whl'
"$MAMBA_EXE" run --name "$ENV_NAME" python -m pip install cupy-cuda11x==13.6.0
"$MAMBA_EXE" run --name "$ENV_NAME" python -m pip install fair-esm==2.0.0
"$MAMBA_EXE" run --name "$ENV_NAME" python -m pip install pyhmmer==0.11.0
"$MAMBA_EXE" run --name "$ENV_NAME" python -m pip install ortools==9.4.1874
"$MAMBA_EXE" run --name "$ENV_NAME" python -m pip install pydssp==0.9.0
"$MAMBA_EXE" run --name "$ENV_NAME" python -m pip install ninja==1.13.0
"$MAMBA_EXE" run --name "$ENV_NAME" python -m pip install PySocks==1.7.1
"$MAMBA_EXE" run --name "$ENV_NAME" python -m pip install absl-py==2.3.1
"$MAMBA_EXE" run --name "$ENV_NAME" python -m pip install beautifulsoup4==4.14.2
"$MAMBA_EXE" run --name "$ENV_NAME" python -m pip install gdown==5.2.0
"$MAMBA_EXE" run --name "$ENV_NAME" python -m pip install ml_collections==1.1.0
"$MAMBA_EXE" run --name "$ENV_NAME" python -m pip install soupsieve==2.8

FLASH_ABI="$("$MAMBA_EXE" run --name "$ENV_NAME" python -c 'import torch; print(torch.compiled_with_cxx11_abi())')"
FLASH_ABI="${FLASH_ABI^^}"
FLASH_WHEEL="flash_attn-2.3.2+cu118torch2.1cxx11abi${FLASH_ABI}-cp310-cp310-linux_x86_64.whl"
wget --continue "https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.2/${FLASH_WHEEL}" -O "flash_attn_whl/${FLASH_WHEEL}"
"$MAMBA_EXE" run --name "$ENV_NAME" python -m pip install "flash_attn_whl/${FLASH_WHEEL}"

"$MAMBA_EXE" run --name "$ENV_NAME" python -m pip install .

ENV_MANAGER="$MAMBA_EXE"
```

Continue with [Download weights and verify the installation](#download-weights-and-verify-the-installation).

## Download weights and verify the installation

Keep the `ENV_NAME` and `ENV_MANAGER` values set by the selected section above. Replace the example weights path before running these commands.

```bash
WEIGHTS_DIR=/path/to/save/pretrained/weights
mkdir -p "$WEIGHTS_DIR/weights" "$WEIGHTS_DIR/lm_weights"

wget --continue 'http://huanglab.phys.hust.edu.cn/EM3DFold/weights/weights_v1.2.tgz' -O "$WEIGHTS_DIR/weights/weights.tgz"
(cd "$WEIGHTS_DIR/weights" && printf '%s  %s\n' 'a9ab7a65020bcccfceddc0e4df8e300c0b1155fafc6c173af78785ed8ed13ede' 'weights.tgz' | sha256sum --check -)
tar -zxf "$WEIGHTS_DIR/weights/weights.tgz" -C "$WEIGHTS_DIR/weights"

wget --continue 'https://zenodo.org/records/15043668/files/rinalmo_giga_pretrained.pt' -O "$WEIGHTS_DIR/lm_weights/rinalmo_giga_pretrained.pt"
(cd "$WEIGHTS_DIR/lm_weights" && printf '%s  %s\n' 'cd93c3f21eb3e767373c9491192686b5846247bd1110693e453c1dd0f321c0db' 'rinalmo_giga_pretrained.pt' | sha256sum --check -)

wget --continue 'https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt' -O "$WEIGHTS_DIR/lm_weights/esm2_t33_650M_UR50D.pt"
(cd "$WEIGHTS_DIR/lm_weights" && printf '%s  %s\n' 'ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c' 'esm2_t33_650M_UR50D.pt' | sha256sum --check -)

wget --continue 'https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt' -O "$WEIGHTS_DIR/lm_weights/esm2_t33_650M_UR50D-contact-regression.pt"
(cd "$WEIGHTS_DIR/lm_weights" && printf '%s  %s\n' '8ffe6edbd4173dc8d45c2cd5cb27d43aad77ec26b4c768200c58ae1f96693575' 'esm2_t33_650M_UR50D-contact-regression.pt' | sha256sum --check -)

WEIGHTS_DIR="$(cd "$WEIGHTS_DIR" && pwd)"
"$ENV_MANAGER" env config vars set "EM_WEIGHTS_DIR=$WEIGHTS_DIR/weights" --name "$ENV_NAME"

"$ENV_MANAGER" run --name "$ENV_NAME" python -c 'import em3dfold, flash_attn, torch; print("EM3DFold imports passed")'
"$ENV_MANAGER" run --name "$ENV_NAME" python -m em3dfold --help
```

The commands above use the default `official` sources for the three language-model files. If those sources are slow, use the following `huanglab` commands **instead of** the three language-model `wget` commands above, and then run the same SHA-256 checks:

```bash
wget --continue 'http://huanglab.phys.hust.edu.cn/EM3DFold/lm_weights/rinalmo_giga_pretrained.pt' -O "$WEIGHTS_DIR/lm_weights/rinalmo_giga_pretrained.pt"
wget --continue 'http://huanglab.phys.hust.edu.cn/EM3DFold/lm_weights/esm2_t33_650M_UR50D.pt' -O "$WEIGHTS_DIR/lm_weights/esm2_t33_650M_UR50D.pt"
wget --continue 'http://huanglab.phys.hust.edu.cn/EM3DFold/lm_weights/esm2_t33_650M_UR50D-contact-regression.pt' -O "$WEIGHTS_DIR/lm_weights/esm2_t33_650M_UR50D-contact-regression.pt"
```
