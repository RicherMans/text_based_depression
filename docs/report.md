# DAIC Progession Report

## 数据

| 部分 | 长度 | %患者 |

### 建模

数据有视频，音频，文本的。



```python
def parse_poolingfunction(poolingfunction_name='mean'):
    if poolingfunction_name == 'mean':
        def pooling_function(x, d): return x.mean(d)
    elif poolingfunction_name == 'max':
        def pooling_function(x, d): return x.max(d)[0]
    elif poolingfunction_name == 'linear':
        def pooling_function(x, d): return (x**2).sum(d) / x.sum(d)
    elif poolingfunction_name == 'exp':
        def pooling_function(x, d): return (
            x.exp() * x).sum(d) / x.exp().sum(d)
    elif poolingfunction_name == 'time':  # Last timestep
        def pooling_function(x, d): return x.select(d, -1)
    return pooling_function
```

### 准则

$MAE = \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left| x_n - y_n \right|$

$BCE = \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],$


$\ell(x, y) = BCE(x,y) + MAE(x,y)$



## 使用模型


1. LSTM ( 双方)
2. TCN ( 5 层 --> 32 dilation）

## Text - only 结果

特征是100维度的WordVec。

训练方法:


* `Batchsize = 1`
* LSTM(S) = 4 节点, 2 层 （啊？）
* LSTM(L) = 128 节点, 3 层

| Model      | Feature | Pooling      | Pre  | Rec  | F1   | MAE  | RMSE |
|------------|---------|--------------|------|------|------|------|------|
| LSTM(Base) | WordVec | Time         | 0.71 | 0.38 | 0.5  | 7.02 | 9.4  |
| LSTM(S)    | WordVec | Time         | 0.62 | 0.59 | 0.48 | 5.6  | 6.4  |
| LSTM(S)    | WordVec | Mean         | 0.59 | 0.59 | 0.54 | 5.29 | 6.3  |
| LSTM(S)    | WordVec | Linear       | 0.33 | 0.50 | 0.4  | 7.4  | 9.8  |
| LSTM(S)    | WordVec | Exp          | 0.53 | 0.52 | 0.5  | 5.5  | 6.3  |
| LSTM(S)    | WordVec | Max          | 0.64 | 0.62 | 0.63 | 5.5  | 6.3  |
| LSTM(L)    | WordVec | Time         | 0.17 | 0.5  | 0.26 | 6    | 6.8  |
| LSTM(L)    | WordVec | Mean         | 0.44 | 0.44 | 0.4  | 5.6  | 6.5  |
| LSTM(L)    | WordVec | Linear       | 0.33 | 0.50 | 0.4  | 7.4  | 9.9  |
| LSTM(L)    | WordVec | Exp          | 0.84 | 0.54 | 0.48 | 6    | 6.8  |
| LSTM(L)    | WordVec | Max          | 0.52 | 0.51 | 0.51 | 5.8  | 6.6  |
| LSTM(S)    | Bert    | Exp          | 0.17 | 0.5  | 0.26 | 5.5  | 6.4  |
| LSTM(S)    | Bert    | Time         | 0.68 | 0.64 | 0.65 | 5.3  | 6.9  |
| LSTM(S)    | Bert    | Mean         | 0.17 | 0.50 | 0.26 | 5.4  | 6.5  |
| LSTM(S)    | Bert    | Linear       | 0.17 | 0.5  | 0.26 | 5.9  | 6.7  |
| LSTM(S)    | Bert    | Max          | 0.84 | 0.54 | 0.48 | 5.4  | 6.5  |
| LSTM(L)    | Bert    | Linear       | 0.17 | 0.5  | 0.26 | 4.8  | 5.7  |
| LSTM(L)    | Bert    | Time         | 0.65 | 0.66 | 0.6  | 5.7  | 6.8  |
| LSTM(L)    | Bert    | Max          | 0.67 | 0.68 | 0.67 | 5.9  | 6.7  |
| LSTM(L)    | Bert    | Exp          | 0.45 | 0.48 | 0.43 | 5.2  | 6.2  |
| LSTM(L)    | Bert    | Mean         | 0.75 | 0.75 | 0.75 | 4.8  | 5.6  |
| LSTM(L)    | Bert    | PosAttn+Attn | 0.82 | 0.85 | 0.82 | 4.4  | 5.4  |
| LSTM(L)    | Bert    | PosAttn+Attn | 0.87 | 0.81 | 0.83 | 4.2  | 5.3  |
| LSTM(L)    | Bert    | Attn         | 0.84 | 0.85 | 0.84 | 4.1  | 5.2  |


## Audio Only


 | Model      | Feature   | Pooling | Pre  | Rec  | F1   | MAE   | RMSE |
 |------------|-----------|---------|------|------|------|-------|------|
 | LSTM(Base) | HighOrder | Time    | 0.38 | 0.71 | 0.5  | 5.31  | 6.94 |
 | LSTM       | HighOrder | Time    | 0.6  | 0.62 | 0.59 | 5.629 | 6.5  |
 | LSTM       | HighOrder | Mean    | 0.57 | 0.57 | 0.51 | 5.8   | 6.6  |
 | LSTM       | HighOrder | Linear  | 0.17 | 0.5  | 0.26 | 5.66  | 6.57 |
 | LSTM       | HighOrder | Exp     | 0.57 | 0.57 | 0.54 | 5.6   | 6.5  |
 | LSTM       | HighOrder | Max     | 0.62 | 0.61 | 0.54 | 6.27  | 7.4  |
 | TCN        | HighOrder | Max     | 0.7  | 0.62 | 0.63 | 5.63  | 6.8  |
 | TCN        | HighOrder | Mean    | 0.47 | 0.47 | 0.47 | 5.8   | 7.3  |
 | TCN        | HighOrder | Time    | 0.57 | 0.58 | 0.57 | 5.9   | 7.08 |
 | BTCN       | HighOrder | Time    | 0.59 | 0.6  | 0.6  | 5.4   | 6.5  |
 | BTCN       | HighOrder | Attn    | 0.78 | 0.79 | 0.78 | 4.9   | 5.8  |
 | TCN        | LMS       | Mean    | 0.68 | 0.52 | 0.30 | 5.6   | 6.7  |
 | TCN        | LMS       | Max     | 0.65 | 0.66 | 0.6  | 5.55  | 6.5  |
 | TCN        | LMS       | Exp     | 0.45 | 0.45 | 0.45 | 5.9   | 7.0  |
 | LSTM       | LMS       | Mean    | 0.17 | 0.50 | 0.26 | 6.05  | 6.91 |


### Huber Loss


$$
Huber = \text{loss}(x, y) = \frac{1}{n} \sum_{i} z_{i}\\
where,\\
z_{i} =
\begin{cases}
0.5 (x_i - y_i)^2, & \text{if } |x_i - y_i| < 1 \\
|x_i - y_i| - 0.5, & \text{otherwise }
\end{cases}
$$
Here:

$\ell(x,y) = BCE(x,y) + Huber(x,y)$

| Model | Feature   | Pooling | Pre   | Rec | F1  | MAE  | RMSE |
|-------|-----------|---------|------|------|------|------|------|
| LSTM  | HighOrder | Mean    | 0.6  | 0.6  | 0.6  | 5.86 | 6.69 |
| LSTM  | HighOrder | Max     | 0.32 | 0.48 | 0.39 | 5.6  | 6.4  |
| LSTM  | HighOrder | Exp     | 0.61 | 0.61 | 0.57 | 5.9  | 6.8  |
| LSTM  | HighOrder | Linear  | 0.65 | 0.66 | 0.60 | 6.1  | 7.75 |
| TCN   | LMS       | Max     | 0.61 | 0.57 | 0.44 | 5.8  | 6.8  |
| TCN   | LMS       | Mean    | 0.17 | 0.5  | 0.26 | 5.6  | 6.7  |



# MultiModal - Feature extraction

| AudioModel | TextModel | 
