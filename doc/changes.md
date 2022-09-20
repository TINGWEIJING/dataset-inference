# Change on author code
## Dataset Inference Code Checking Exp
- Change import in `models.py`
```python
from model_src.preactresnet import *
from model_src.wideresnet import *
from model_src.cnn import *
from model_src.resnet import *
from model_src.resnet_8x import *
# from preactresnet import *
# from wideresnet import *
# from cnn import *
# from resnet import *
# from resnet_8x import *
```
- Change model output path for `train.py`
- Change ti_500K_pseudo_labeled.pickle data path for `funcs.py`
- Change teacher model load path for `train.py`
- Fix bug for:
```
  File "/data/weijing/author-DI/src/attacks.py", line 156, in rand_steps
    remaining[remaining] = new_remaining
RuntimeError: unsupported operation: some elements of the input tensor and the written-to tensor refer to a single memory location. Please clone() the tensor before performing the operation.
```
change to `remaining[remaining.clone()] = new_remaining`
- Change model input and feature output paths for `generate_features.py`
- Add `utils/logger.py` script for print output
- Add setup `Ubuffered` output stream code for `generate_features.py` and `attack.py`
- Change `print` file direction for `generate_features.py` and `attack.py`

## Use unrelated datasets
- Add 'MINST' in `n_class` map
- Add 'MINST' in `net_mapper` map for `train.py` & `generate_features.py`
- Add new `--experiment` parser for quick setting
- Add `dataset.py` for alternate dataloading