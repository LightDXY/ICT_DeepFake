
#!/bin/bash
pip install --user bcolz mxnet tensorboardX matplotlib easydict opencv-python --no-cache-dir -U | cat
pip install --user scikit-image imgaug PyTurboJPEG --no-cache-dir -U | cat
pip install --user scikit-learn --no-cache-dir -U | cat
pip install --user torch==1.6.0 --no-cache-dir -U | cat
pip install --user  termcolor imgaug prettytable --no-cache-dir -U | cat
pip install --user --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl --no-cache-dir -U | cat
pip install --user timm==0.3.2 --no-cache-dir -U | cat

