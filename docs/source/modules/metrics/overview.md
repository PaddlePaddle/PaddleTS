# PaddleTS metrics user guide

### Metric introduction

Metric is used for measuring the performance of a model, usually by calculating a certain distance between the the ground-truth data and the predicted results by a model.

Similar to the loss function, different metrics are used to evaluate the model performance for different tasks. For example, `MAE(Mean Absolute Error)` and `MSE(Mean squared error)` are usually used for regression tasks. This library has some built-in metrics which are widely used. It also supports customized metrics by allowing developers to implement new subclasses of `base::Metric`.

#### Code example
Take MSE as an example. Given a `predicted TSDataset` and a corresponding `ground-truth TSDataset`, it measures the MSE between these two TSDatasets as below.
```shell
from paddlets import TSDataset
from paddlets.metrics import MSE

"""Note: Both y_true and y_score are pd.DataFrame.
"""
mse = MSE()
tsdataset_true = TSDataset.load_from_dataframe(y_true)
tsdataset_score = TSDataset.load_from_dataframe(y_score)
result = mse(tsdataset_true, tsdataset_score)
```

### Customize Metric
    
To implement a customized metric, it simply takes the following steps:
1. Create a class that inherits from paddlets.metric.base.Metric.
2. Set two class members, `_NAME(metric name)` and `_MAXIMIZE(optimization direction)`.
3. Initialize the required parameter in `__init__`.
4. Implement the metric_fn method according to computation logic.

#### Code example
Again, take MSE as an example. The details of implementation are shown by the following code snipper.
```shell
from paddlets.metrics.base import Metric
import sklearn.metrics as metrics

class MSE(Metric):
    """
    1. Inherited from paddlets.metric.Metric.
    2. Set two class members, _NAME(metric name) and _MAXIMIZE(optimization direction).
    """
    _NAME = "mse"
    _MAXIMIZE = False

    def __init__(
        self,
        mode: str = "normal"
    ):
    """3. Initialize the required parameter in __init__.
    """
        super(MSE, self).__init__(mode)

    @ensure_2d
    def metric_fn(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray
    ) -> float:
        """4. Implement the metric_fn method according to computation logic.
        """
        return metrics.mean_squared_error(y_true, y_score)
```
