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
uv pip install -r requirements.txt
uv pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
```