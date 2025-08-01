import dataclasses
import datetime
import functools
import math
import re
from typing import Optional
import glob
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
import logging
import json
import os

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax
import numpy as np
import xarray
from jax.profiler import trace

from contextlib import contextmanager
import time
from scipy import stats

# 过滤 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger('absl').setLevel(logging.ERROR)

@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f"{name}: {end - start:.2f} seconds")

print("JAX devices:", jax.devices())
print("Default backend:", jax.default_backend())
print("Available devices:", jax.local_device_count())

def parse_file_parts(file_name):
    return dict(part.split("-", 1) for part in file_name.split("_"))

# 模型加载部分保持不变
params_file_options = [
    name for blob in glob.glob("params/**", recursive=True)
    if (name := blob.replace("params/", ""))]

params_file = 'GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz'
print("params_file:", params_file)

with open(f"params/{params_file}", "rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)
params = ckpt.params
state = {}

model_config = ckpt.model_config
task_config = ckpt.task_config

# 打印task_config信息以便调试
print("Task config input_variables:", task_config.input_variables)
print("Task config target_variables:", task_config.target_variables)
print("Task config forcing_variables:", task_config.forcing_variables)

dataset_file = 'source-fake_date-2022-01-01_res-0.25_levels-13_steps-01.nc'
print("dataset_file:", dataset_file)

with open(f"dataset/{dataset_file}", "rb") as f:
    example_batch = xarray.load_dataset(f).compute()

print("All Examples:", example_batch.dims.mapping)

# 打印所有变量的维度信息
print("\nDataset variables and their dimensions:")
for var_name, var in example_batch.data_vars.items():
    print(f"  {var_name}: {var.dims} - shape: {var.shape}")

# 加载归一化数据
with open("stats/diffs_stddev_by_level.nc", "rb") as f:
    diffs_stddev_by_level = xarray.load_dataset(f).compute()
with open("stats/mean_by_level.nc", "rb") as f:
    mean_by_level = xarray.load_dataset(f).compute()
with open("stats/stddev_by_level.nc", "rb") as f:
    stddev_by_level = xarray.load_dataset(f).compute()

def construct_wrapped_graphcast(
        model_config: graphcast.ModelConfig,
        task_config: graphcast.TaskConfig):
    """Constructs and wraps the GraphCast Predictor."""
    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level)
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor

@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)

def with_configs(fn):
    return functools.partial(
            fn, model_config=model_config, task_config=task_config)

def with_params(fn):
    return functools.partial(fn, params=params, state=state)

def drop_state(fn):
    return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
        run_forward.apply))))

def remove_time_from_static_variables(dataset):
    """
    移除静态变量的时间维度
    """
    # 已知的静态变量列表
    static_vars = ['geopotential_at_surface', 'land_sea_mask']
    
    fixed_dataset = dataset.copy()
    
    for var_name in static_vars:
        if var_name in dataset.data_vars:
            var = dataset[var_name]
            if 'time' in var.dims:
                print(f"Removing time dimension from static variable: {var_name}")
                # 移除时间维度，只保留第一个时间点的数据
                fixed_dataset[var_name] = var.isel(time=0, drop=True)
                print(f"  New dimensions: {fixed_dataset[var_name].dims}")
            else:
                print(f"Static variable {var_name} already has no time dimension")
    
    return fixed_dataset

def replace_prediction_at_target_time(current_batch, pred_ds, target_time_idx=-1):
    """
    用预测结果替换当前批次中指定时间点的数据
    """
    # 复制当前批次
    updated_batch = current_batch.copy(deep=True)
    
    # 获取目标时间点
    target_time = current_batch.time.values[target_time_idx]
    
    print(f"Replacing data at time index {target_time_idx}, time value: {target_time}")
    
    # 替换动态变量的数据
    for var_name in pred_ds.data_vars:
        if var_name in updated_batch.data_vars and 'time' in updated_batch[var_name].dims:
            # 获取预测数据（应该只有一个时间点）
            pred_var = pred_ds[var_name]
            if 'time' in pred_var.dims:
                pred_var = pred_var.isel(time=0)  # 取第一个（也是唯一的）时间点
            
            # 替换指定时间点的数据
            updated_batch[var_name].loc[dict(time=target_time)] = pred_var.values
    
    return updated_batch

def create_next_timestep_batch(current_batch):
    """
    创建下一个预测步骤的输入数据：
    - 移除第一个时间点
    - 为新的时间点创建占位符
    - 保持3个时间点的结构
    - 保持静态变量不变
    """
    # 获取时间信息
    current_times = current_batch.time.values
    time_step = current_times[1] - current_times[0]
    new_time = current_times[-1] + time_step
    
    # 获取最后的datetime并计算新的datetime
    last_datetime = current_batch.datetime.isel(batch=0, time=-1).values
    new_datetime = last_datetime + np.timedelta64(int(time_step), 'ns')
    
    # 分离有时间维度和无时间维度的变量
    time_vars = {}
    static_vars = {}
    
    for var_name, var in current_batch.data_vars.items():
        if 'time' in var.dims:
            time_vars[var_name] = var
        else:
            static_vars[var_name] = var
    
    # 对有时间维度的变量，保留最后两个时间点
    time_vars_ds = xarray.Dataset(time_vars, coords=current_batch.coords)
    last_two_timesteps = time_vars_ds.isel(time=slice(-2, None))
    
    # 创建新时间点的占位符数据（复制最后一个时间点的结构）
    new_timestep_template = time_vars_ds.isel(time=slice(-1, None)).copy()
    
    # 更新新时间点的坐标
    new_timestep_template = new_timestep_template.assign_coords(time=[new_time])
    
    # 更新datetime坐标
    datetime_coord = xarray.DataArray(
        [[new_datetime]], 
        dims=['batch', 'time'],
        coords={'batch': current_batch.batch, 'time': [new_time]}
    )
    new_timestep_template = new_timestep_template.assign_coords(datetime=datetime_coord)
    
    # 合并：最后两个时间点 + 新时间点占位符
    next_batch_time_vars = xarray.concat([last_two_timesteps, new_timestep_template], dim='time')
    
    # 添加静态变量（保持不变）
    for var_name, var in static_vars.items():
        next_batch_time_vars[var_name] = var
    
    return next_batch_time_vars

def save_prediction_to_nc(pred_ds, current_batch, target_time_idx, step, output_dir="predictions"):
    """
    将预测结果保存为nc文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取目标时间信息
    target_time = current_batch.time.values[target_time_idx]
    target_datetime = current_batch.datetime.isel(batch=0, time=target_time_idx).values
    datetime_str = np.datetime_as_string(target_datetime, unit='h')
    
    # 为预测结果设置正确的时间坐标
    pred_to_save = pred_ds.copy()
    pred_to_save = pred_to_save.assign_coords(time=[target_time])
    
    # 设置datetime坐标
    datetime_coord = xarray.DataArray(
        [[target_datetime]], 
        dims=['batch', 'time'],
        coords={'batch': current_batch.batch, 'time': [target_time]}
    )
    pred_to_save = pred_to_save.assign_coords(datetime=datetime_coord)
    
    # 构建文件名
    filename = f"prediction_step_{step:03d}_{datetime_str.replace(':', '-')}.nc"
    filepath = os.path.join(output_dir, filename)
    
    # 保存文件
    pred_to_save.to_netcdf(filepath)
    print(f"Saved prediction step {step} to {filepath}")
    
    return filepath

print('Initial example_batch.datetime.values:', example_batch.datetime.values)

# 移除静态变量的时间维度
print("Removing time dimension from static variables...")
example_batch = remove_time_from_static_variables(example_batch)

print("\nFixed dataset variables and their dimensions:")
for var_name, var in example_batch.data_vars.items():
    print(f"  {var_name}: {var.dims} - shape: {var.shape}")

# 初始化工作数据集（确保有3个时间点用于预测）
if len(example_batch.time) < 3:
    raise ValueError("Initial dataset must have at least 3 time points")

# 如果初始数据超过3个时间点，只取最后3个
if len(example_batch.time) > 3:
    working_batch = example_batch.isel(time=slice(-3, None))
else:
    working_batch = example_batch.copy()

eval_steps = 40

print(f"Starting multi-step prediction with {eval_steps} steps")
print(f"Initial working_batch shape: {dict(working_batch.sizes)}")
print(f"GraphCast logic: Use time points [t-1, t] to predict t+1")

for i in tqdm(range(eval_steps), desc="Multi-step prediction"):
    print(f"\n=== Step {i+1}/{eval_steps} ===")
    
    current_times = working_batch.time.values
    current_datetimes = working_batch.datetime.values
    
    print(f'Current times: {current_times}')
    print(f'Current datetimes: {current_datetimes}')
    print(f'Will use times [0, 1] to predict time [2]')
    
    # 打印变量的维度信息以调试
    print("Variables in working_batch:")
    for var_name, var in working_batch.data_vars.items():
        if var_name in ['geopotential_at_surface', 'land_sea_mask']:
            print(f"  {var_name}: {var.dims} - shape: {var.shape}")
    
    # 提取输入、目标和强迫数据
    # GraphCast会使用前两个时间点预测第三个时间点
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        working_batch, 
        target_lead_times=slice("6h", f"{1*6}h"),
        **dataclasses.asdict(task_config)
    )
    
    print(f"eval_inputs variables: {list(eval_inputs.data_vars)}")
    print(f"eval_targets variables: {list(eval_targets.data_vars)}")  
    print(f"eval_forcings variables: {list(eval_forcings.data_vars) if eval_forcings else 'None'}")
    
    # 检查eval_inputs中静态变量的维度
    print("Static variables in eval_inputs:")
    for var_name in ['geopotential_at_surface', 'land_sea_mask']:
        if var_name in eval_inputs.data_vars:
            var = eval_inputs[var_name]
            print(f"  {var_name}: {var.dims} - shape: {var.shape}")
    
    print(f"Target time being predicted: {working_batch.time.values[-1]} ({working_batch.datetime.values[0][-1]})")
    
    # 执行预测
    with timer(f"Prediction step {i+1}"):
        predictions = rollout.chunked_prediction(
            run_forward_jitted,
            rng=jax.random.PRNGKey(0),
            inputs=eval_inputs,
            targets_template=eval_targets * np.nan,
            forcings=eval_forcings,
            verbose=False
        )
    
    # 保存预测结果（预测的是当前第三个时间点的数据）
    save_prediction_to_nc(predictions, working_batch, -1, i+1)
    
    # 用预测结果替换第三个时间点的真实数据
    working_batch = replace_prediction_at_target_time(working_batch, predictions, target_time_idx=-1)
    
    # 为下一次预测创建新的3时间点批次
    # 这将移除第一个时间点，并为新的时间点创建占位符
    if i < eval_steps - 1:  # 不是最后一次迭代
        working_batch = create_next_timestep_batch(working_batch)
    
    # 打印当前状态
    print(f'Working_batch shape: {dict(working_batch.sizes)} (should always be 3 time points)')
    if len(working_batch.time) == 3:
        print(f'Next prediction will use times {working_batch.time.values[:-1]} to predict {working_batch.time.values[-1]}')
    
    # 手动清理内存
    del predictions
    import gc
    gc.collect()

print(f"\n=== Multi-step prediction completed ===")
print(f"All {eval_steps} predictions have been saved to individual NC files in the 'predictions' directory")

# 创建汇总信息文件
summary_info = {
    "total_steps": eval_steps,
    "time_step_hours": 6,
    "initial_datetime": str(example_batch.datetime.isel(batch=0, time=0).values),
    "final_predicted_datetime": str(working_batch.datetime.isel(batch=0, time=-1).values),
    "prediction_logic": "Use times [t-1, t] to predict t+1",
    "model_config": str(model_config),
    "task_config": str(task_config)
}

os.makedirs("predictions", exist_ok=True)
with open("predictions/prediction_summary.json", "w") as f:
    json.dump(summary_info, f, indent=2)

print("Prediction summary saved to predictions/prediction_summary.json")