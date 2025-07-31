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

# import cartopy.crs as ccrs
# from google.cloud import storage
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
# from IPython.display import HTML
# import ipywidgets as widgets
import haiku as hk
import jax
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib import animation
import numpy as np
import xarray
from jax.profiler import trace

from contextlib import contextmanager
import time
from scipy import stats

# 过滤 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# 设置 absl 日志级别为 ERROR，这将过滤 WARNING 及以下级别的消息
logging.getLogger('absl').setLevel(logging.ERROR)

@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    # print(f"{name}: {end - start:.2f} seconds")

print("JAX devices:", jax.devices())
print("Default backend:", jax.default_backend())
print("Available devices:", jax.local_device_count())

def parse_file_parts(file_name):
    return dict(part.split("-", 1) for part in file_name.split("_"))

# @title Choose the model

params_file_options = [
    # name for blob in gcs_bucket.list_blobs(prefix="params/")
    # if (name := blob.name.removeprefix("params/"))]    # Drop empty string.
    name for blob in glob.glob("params/**", recursive=True)
    if (name := blob.replace("params/", ""))]    # Drop empty string.
# print("params_file_options:", params_file_options)

# params_file = params_file_options[1]    # TODO: 0 (GraphCast_small), 1 (GraphCast), 2 (GraphCast_operational)
params_file = 'GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz'
print("params_file:", params_file)

# @title Load the model

# with gcs_bucket.blob(f"params/{params_file.value}").open("rb") as f:
with open(f"params/{params_file}", "rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)
params = ckpt.params
state = {}

model_config = ckpt.model_config
task_config = ckpt.task_config
# print("Model description:\n", ckpt.description, "\n")
# print("Model license:\n", ckpt.license, "\n")

# print("model_config:", model_config)
# print("task_config:", task_config)


# dataset_file = 'source-hres_date-2022-01-01_res-0.25_levels-13_steps-04.nc'
dataset_file = 'source-fake_date-2022-01-01_res-0.25_levels-13_steps-01.nc'    # TODO: use self constructed nc file
print("dataset_file:", dataset_file)

# @title Load weather data

with open(f"dataset/{dataset_file}", "rb") as f:
    example_batch = xarray.load_dataset(f).compute()

# print(", ".join([f"{k}: {v}" for k, v in parse_file_parts(dataset_file.removesuffix(".nc")).items()]))
# print("example_batch:", example_batch)
    
print("All Examples:    ", example_batch.dims.mapping)

# @title Load normalization data

with open("stats/diffs_stddev_by_level.nc", "rb") as f:
    diffs_stddev_by_level = xarray.load_dataset(f).compute()
with open("stats/mean_by_level.nc", "rb") as f:
    mean_by_level = xarray.load_dataset(f).compute()
with open("stats/stddev_by_level.nc", "rb") as f:
    stddev_by_level = xarray.load_dataset(f).compute()
    
# @title Build jitted functions, and possibly initialize random weights

def construct_wrapped_graphcast(
        model_config: graphcast.ModelConfig,
        task_config: graphcast.TaskConfig):
    """Constructs and wraps the GraphCast Predictor."""
    # Deeper one-step predictor.
    predictor = graphcast.GraphCast(model_config, task_config)

    # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
    # from/to float32 to/from BFloat16.
    predictor = casting.Bfloat16Cast(predictor)

    # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
    # BFloat16 happens after applying normalization to the inputs/targets.
    predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level)

    # Wraps everything so the one-step model can produce trajectories.
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = loss_fn.apply(
                params, state, jax.random.PRNGKey(0), model_config, task_config,
                i, t, f)
        return loss, (diagnostics, next_state)
    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
            _aux, has_aux=True)(params, state, inputs, targets, forcings)
    return loss, diagnostics, next_state, grads

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
    return functools.partial(
            fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
    return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
    return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))

# if params is None:
#     params, state = init_jitted(
#             rng=jax.random.PRNGKey(0),
#             inputs=train_inputs,
#             targets_template=train_targets,
#             forcings=train_forcings)

# loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
# grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
        run_forward.apply))))

def replace_time_point(current_ds, pred_ds, time_to_replace):
    """
    替换特定时间点的数据，正确处理维度
    """
    current_ds = current_ds.copy()
    
    # 分别处理带时间维度和不带时间维度的变量
    for var in current_ds.data_vars:
        if 'time' in current_ds[var].dims:
            # 对于带时间维度的变量，替换特定时间点的数据
            if var in pred_ds:
                # 确保pred_ds的变量只有一个时间点
                if 'time' in pred_ds[var].dims and len(pred_ds[var].time) > 1:
                    var_data = pred_ds[var].isel(time=0)
                else:
                    var_data = pred_ds[var]
                
                # 去掉多余的维度
                var_data = var_data.squeeze()
                
                # 赋值
                current_ds[var].loc[dict(time=time_to_replace)] = var_data.values
        else:
            # 对于静态变量，保持不变或使用pred_ds中的静态值
            if var in pred_ds:
                current_ds[var] = pred_ds[var]
    
    return current_ds
    
# 复制缺失字段到pred_ds
def copy_static_fields(pred_ds, example_batch):
    """
    将静态字段从example_batch复制到pred_ds
    """
    # 创建pred_ds的副本
    pred_ds = pred_ds.copy()
    
    # 复制缺失的字段
    static_fields = ['geopotential_at_surface', 'land_sea_mask']
    for field in static_fields:
        if field in example_batch and field not in pred_ds:
            # 确保时间维度匹配
            field_data = example_batch[field]
            # 添加到pred_ds
            pred_ds[field] = field_data
            
    return pred_ds

print('example_batch.datetime.values:', example_batch.datetime.values)

eval_steps = 40  # 每6小时一个step
# all_predictions = []
for i in range(eval_steps):
    new_example_batch = example_batch  #.isel(time=slice(-3, -1)).copy()
    # print('new_example_batch:', new_example_batch)
    initial_times = new_example_batch.time.values
    time_step = initial_times[1] - initial_times[0]
    last_datetime = new_example_batch.datetime.isel(batch=0, time=-2).values
    print('new_example_batch.datetime.values:', new_example_batch.datetime.values)
    print('time_step:', time_step, 'last_datetime:', last_datetime)
    
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        new_example_batch, target_lead_times=slice("6h", f"{1*6}h"),
        **dataclasses.asdict(task_config))
    
    with timer("Prediction"):
        predictions = rollout.chunked_prediction(
            run_forward_jitted,
            rng=jax.random.PRNGKey(0),
            inputs=eval_inputs,
            targets_template=eval_targets * np.nan,
            forcings=eval_forcings,
            verbose=True)
    # print('predictions:', predictions)
    # all_predictions.append(predictions)
    
    # 创建新的时间点
    new_time = new_example_batch.time.values[-2] + time_step
    # 计算新的datetime
    new_datetime = last_datetime + np.timedelta64(int(time_step), 'ns')
    last_datetime = new_datetime
    print('new_time:', new_time, 'last_datetime:', last_datetime)
    # 创建datetime坐标数组
    datetime_coord = xarray.DataArray(
        [[new_datetime]], 
        dims=['batch', 'time'],
        coords={'batch': new_example_batch.batch, 'time': [new_time]}
    )
    # 将预测结果转换为Dataset，并设置正确的时间坐标
    pred_ds = predictions.assign_coords(time=[new_time], datetime=datetime_coord)
    pred_ds = copy_static_fields(pred_ds, example_batch)
    # print('pred_ds:', pred_ds)
    example_batch = replace_time_point(example_batch, pred_ds, new_time)
