# Laplacian Score
An implementation of laplacian score by Python since all the code on GitHub are either too complicated or unavailable

Reference
[He, Xiaofei, Deng Cai, and Partha Niyogi. "Laplacian score for feature selection." *Advances in neural information processing systems*
         18 (2005).](https://proceedings.neurips.cc/paper/2005/file/b5b03f06271f8917685d14cea7c6c50a-Paper.pdf)


# Usage

```python
laplacian_score(
    df_arr: numpy array,
    label: None,
    **kwargs
)
```
Arguments:
* `df_arr`: A numpy array represent data
* `label`: Default=`None`, Label if the data is a supervised data.
* `k_nearest`: Default=`8`, if the data is an unsupervised data, use k_nearest to find the edge of distance graph, this parameter takes no action if `label` parameter exists.


## Example
```python
from sklearn import datasets
import pandas as pd
import numpy as np
iris = datasets.load_iris()
df = pd.DataFrame(iris["data"], columns=iris["feature_names"])
target = iris["target"]
print(df.head())
```
| sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) |
|-------------------------|------------------|-------------------|------------------|
| 5.1                     | 3.5              | 1.4               | 0.2              |
| 4.9                     | 3.0              | 1.4               | 0.2              |
| 4.7                     | 3.2              | 1.3               | 0.2              |

### No lable data (Unsupervised)
```python
>>> laplacian_score(np.array(df), k_nearest=3)
array([0.04132189, 0.17426192, 0.01415396, 0.03982228])
```
The value above corresponding to laplacian score of each features, the smaller the score, the higher chance be selected

### Labled data (Supervised)
```python
>>> laplacian_score(np.array(df), label=target)
array([0.60421927, 0.78376157, 0.1138573 , 0.10572175])
```
