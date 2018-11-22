# gnn

Three models are committed (`3x3`, `4x4`, and `5x5`). `nxm` = `n` layers, `m` tracks (`n*m` nodes, `n*(n-1)*m` edges).

```
# setup pytorch environment
conda create --name pytorch-training --file pytorch-training.txt 
source activate pytorch-training
# setup vivado
source /home/jduarte1/setup_vivado.sh
# run inference, and produce weight cpp files from pytroch model files (default 3x3)
python inference.py
# edit myproject_test.cpp and firmware/parameters.h to change inputs if you want to run 4x4 or 5x5 (default 3x3)
vivado_hls -f build_prj.tcl
```