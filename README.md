```bash
# 通过 pip freeze 导出了依赖
pip install -r requirements.txt

# 运行
./run.sh
```

## Environment Configuration
```bash
conda create -n "esen" python=3.10
conda activate esen
conda install conda-forge::uv
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install torch_geometric
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
uv pip install lightning==2.5.1 huggingface_hub==0.27.1 hydra-core==1.3.2 ase==3.24.0 lmdb==1.5.1 orjson==3.10.13 pydantic==2.10.4 numba==0.60.0 wandb==0.19.6 e3nn==0.4.4 lion-pytorch==0.2.3 pymatgen==2025.1.9
uv pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'

```