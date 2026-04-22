# 时间特征提取

## 1. 基本时间组件

```python
df['datetime'] = pd.to_datetime(df['datetime_col'])

df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek    # 0=Mon, 6=Sun
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['quarter'] = df['datetime'].dt.quarter
df['day_of_year'] = df['datetime'].dt.dayofyear
```

## 2. 周期性编码

将周期性特征（如小时、月份）编码为正弦/余弦对，保留循环距离。

```python
import numpy as np

def cyclical_encode(df, col, max_val):
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

df = cyclical_encode(df, 'hour', 24)
df = cyclical_encode(df, 'month', 12)
df = cyclical_encode(df, 'dayofweek', 7)
```

## 3. 时间差特征

```python
# 距今天数
df['days_since_event'] = (pd.Timestamp.now() - df['event_date']).dt.days

# 两个时间列之差
df['duration_hours'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 3600

# 距上次事件的间隔（需先排序）
df = df.sort_values(['user_id', 'datetime'])
df['time_since_last'] = df.groupby('user_id')['datetime'].diff().dt.total_seconds()
```

## 4. 滑动窗口统计

```python
# 过去 7 天的滑动均值
df = df.sort_values('datetime')
df['rolling_7d_mean'] = df.groupby('user_id')['value'].transform(
    lambda x: x.rolling('7D', min_periods=1).mean()
)

# 过去 N 条记录的统计
df['rolling_3_std'] = df.groupby('user_id')['value'].transform(
    lambda x: x.rolling(3, min_periods=1).std()
)
```

**注意：** 滑动窗口必须只使用历史数据，不能看到未来。

## 5. 滞后特征 (Lag Features)

```python
for lag in [1, 3, 7]:
    df[f'value_lag_{lag}'] = df.groupby('user_id')['value'].shift(lag)
```

**关键：** 确保 shift 方向正确，只使用历史值。
