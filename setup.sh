conda create --copy --name hls4ml-env python=2.7

#
#conda install --name hls4ml-env --file pytorch-training.conda

## CPU version of pytorch for now
#conda install pytorch torchvision -c pytorch
conda install pytorch-cpu torchvision-cpu -c pytorch

conda install -c anaconda scikit-learn h5py pyyaml
#source activate hls4ml-env

