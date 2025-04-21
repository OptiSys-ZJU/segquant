# controlnet数据集使用指南
## 生产数据集 cn_dataset_preprocess.py
``` shell 
# 生产哪个写哪个 
# canny
python -m stable_diff.cn_dataset_preprocess --cn_type=canny
# depth
python -m stable_diff.cn_dataset_preprocess --cn_type=depth
```
其余参数：


## 测试数据集 cn_dataset_test.py
``` shell
# 测哪个写哪个
# canny
python -m stable_diff.cn_dataset_test --cn_type=canny
# depth
python -m stable_diff.cn_dataset_test --cn_type=depth
```
