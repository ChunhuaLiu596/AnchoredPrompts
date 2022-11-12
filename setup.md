# 221018

module load pytorch/1.9.00python-3.8.6

pip install transformers --use-feature=2020-resolver

pip install h5py --use-feature=2020-resolver

pip install keras --use-feature=2020-resolver

pip install ipython --use-feature=2020-resolver

pip uninstall tensorflow-gpu 



# 221016

 conda install -c conda-forge pytorch-gpu

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

conda install -c conda-forge pytorch-gpu

```

conda install -c huggingface transformers==4.21.1

conda install pandas

conda install scikit-learn

conda install spacy

python3 -m spacy download en_core_web_sm

```python
pip install IPython
```

# Errors

## Error1

```shell
  File "/home/chunhua/.conda/envs/anchor_prompt/lib/python3.7/site-packages/transformers/utils/import_utils.py", line 947, in is_torch_fx_proxy
    import torch.fx
ModuleNotFoundError: No module named 'torch.fx'
>>> import torch
>>> torch.__version__
'1.3.1'
>>> exit()
```

> update the pytorch version from 1.3.1 to 1.8

## Error2

[E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.

> python3 -m spacy download en_core_web_sm

python -c "import torch; print(torch.__version__)"

## Error3: not using GPU

when installing

```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

The following packages will be SUPERSEDED by a higher-priority channel:

* [ ] pytorch            pkgs/main::pytorch-1.3.1-cuda100py37h~ --> pytorch::pytorch-1.3.1-py3.7_cpu_0

* Return False: when

```shell
import torch
torch.cuda.is_available()
```

* check cuda version on spartan: NVIDIA-SMI 515.48.07    Driver Version: 515.48.07    CUDA Version: 11.7
