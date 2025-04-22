import dataclasses
import datetime
import functools
import math
import re
from typing import Optional
import glob

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

# @title Choose the model

params_file_options = [
		# name for blob in gcs_bucket.list_blobs(prefix="params/")
		# if (name := blob.name.removeprefix("params/"))]	# Drop empty string.
		name for blob in glob.glob("params/**", recursive=True)
		if (name := blob.replace("params/", ""))]	# Drop empty string.
print("params_file_options:", params_file_options)

# params_file = params_file_options[1]	# TODO: 0 (GraphCast_small), 1 (GraphCast), 2 (GraphCast_operational)
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
print("Model description:\n", ckpt.description, "\n")
print("Model license:\n", ckpt.license, "\n")

print("model_config:", model_config)
print("task_config:", task_config)

# @title Get and filter the list of available example datasets

dataset_file_options = [
		# name for blob in gcs_bucket.list_blobs(prefix="dataset/")
		# if (name := blob.name.removeprefix("dataset/"))]	# Drop empty string.
		name for blob in glob.glob("dataset/**", recursive=True)
		if (name := blob.replace("dataset/", ""))]	# Drop empty string.

def data_valid_for_model(
		file_name: str, model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig):
	file_parts = parse_file_parts(file_name.removesuffix(".nc"))
	return (
			model_config.resolution in (0, float(file_parts["res"])) and
			len(task_config.pressure_levels) == int(file_parts["levels"]) and
			(
					("total_precipitation_6hr" in task_config.input_variables and
					 file_parts["source"] in ("era5", "fake")) or
					("total_precipitation_6hr" not in task_config.input_variables and
					 file_parts["source"] in ("hres", "fake"))
			)
	)

# valid_dataset_file_options = []
# for option in dataset_file_options:
# 		if data_valid_for_model(option, model_config, task_config):
# 				# dataset_file = option
# 				# break
# 				valid_dataset_file_options.append(option)
# print('valid_dataset_file_options:', valid_dataset_file_options)
# dataset_file = valid_dataset_file_options[1]	# TODO: select one file
# dataset_file = 'source-hres_date-2022-01-01_res-0.25_levels-13_steps-04.nc'
# dataset_file = 'source-fake_date-2022-01-01_res-0.25_levels-13_steps-01.nc'	# TODO: use self constructed nc file
dataset_file = 'source-era5_date-2024-08-01_res-0.25_levels-13_steps-02.nc'	# TODO: use self constructed nc file
print("dataset_file:", dataset_file)

# @title Load weather data

# if not data_valid_for_model(dataset_file, model_config, task_config):
# 	raise ValueError(
# 			"Invalid dataset file, rerun the cell above and choose a valid dataset file.")

# with gcs_bucket.blob(f"dataset/{dataset_file.value}").open("rb") as f:
# with open(f"dataset/{dataset_file}", "rb") as f:
#	 example_batch = xarray.load_dataset(f).compute()
example_batch = xarray.load_dataset(f"dataset/{dataset_file}").compute()

assert example_batch.dims["time"] >= 3	# 2 for input, >=1 for targets

print(", ".join([f"{k}: {v}" for k, v in parse_file_parts(dataset_file.removesuffix(".nc")).items()]))
print("example_batch:", example_batch)

# @title Choose training and eval data to extract
train_steps = 1	# min=1, max=example_batch.sizes["time"]-2
eval_steps = example_batch.sizes["time"]-2	# min=1, max=example_batch.sizes["time"]-2
print('eval_steps:', eval_steps)

# @title Extract training and eval data

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
		example_batch, target_lead_times=slice("6h", f"{train_steps*6}h"),
		**dataclasses.asdict(task_config))

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
		example_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"),
		**dataclasses.asdict(task_config))

print("All Examples:	", example_batch.dims.mapping)
print("Train Inputs:	", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)
print("Eval Inputs:	 ", eval_inputs.dims.mapping)
print("Eval Targets:	", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)

print('train_inputs:', train_inputs)
print('train_targets:', train_targets)
print('train_forcings:', train_forcings)

# @title Load normalization data

# with gcs_bucket.blob("stats/diffs_stddev_by_level.nc").open("rb") as f:
#	 diffs_stddev_by_level = xarray.load_dataset(f).compute()
# with gcs_bucket.blob("stats/mean_by_level.nc").open("rb") as f:
#	 mean_by_level = xarray.load_dataset(f).compute()
# with gcs_bucket.blob("stats/stddev_by_level.nc").open("rb") as f:
#	 stddev_by_level = xarray.load_dataset(f).compute()
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

if params is None:
	params, state = init_jitted(
			rng=jax.random.PRNGKey(0),
			inputs=train_inputs,
			targets_template=train_targets,
			forcings=train_forcings)

loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
		run_forward.apply))))

# @title Autoregressive rollout (loop in python)

assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
	"Model resolution doesn't match the data resolution. You likely want to "
	"re-filter the dataset list, and download the correct data.")

print("Inputs:	", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

def calculate_metrics(pred_ds, true_ds):
	"""
	计算每个变量的RMSE和ACC，对有level的变量分别计算每个level的指标
	
	Parameters:
	-----------
	pred_ds : xarray.Dataset
		预测数据集
	true_ds : xarray.Dataset
		真实数据集
		
	Returns:
	--------
	dict
		包含每个变量（和level）的RMSE和ACC的字典
	"""
	metrics = {}
		
	# 获取所有变量名
	variables = list(pred_ds.data_vars)
		
	for var in variables:
		pred_var = pred_ds[var]
		true_var = true_ds[var]

		# 检查变量是否包含level维度
		if 'level' in pred_var.dims:
			metrics[var] = {'by_level': {}}
						
			# 对每个level分别计算
			for lev in pred_var.level.values:
				pred_level = pred_var.sel(level=lev)
				true_level = true_var.sel(level=lev)
								
				# 计算该level的指标
				metrics[var]['by_level'][int(lev)] = calculate_single_metric(
					pred_level.squeeze(), 
					true_level.squeeze()
				)
						
			# 计算所有level的平均指标
			all_rmse = np.mean([m['rmse'] for m in metrics[var]['by_level'].values()])
			all_acc = np.mean([m['acc'] for m in metrics[var]['by_level'].values()])
			metrics[var]['all_levels'] = {'rmse': all_rmse, 'acc': all_acc}
						
		else:
			# 对没有level的变量直接计算
			metrics[var] = calculate_single_metric(
				pred_var.squeeze(), 
				true_var.squeeze()
			)
   
	metrics['wind_speed_surface'] = calculate_single_metric(
		np.sqrt(pred_ds['10m_u_component_of_wind'].squeeze()**2+pred_ds['10m_v_component_of_wind'].squeeze()**2), 
		np.sqrt(true_ds['10m_u_component_of_wind'].squeeze()**2+true_ds['10m_v_component_of_wind'].squeeze()**2)
	)
		
	return metrics

def calculate_single_metric(pred, true):
	"""
	计算单个字段的RMSE和ACC
	
	Parameters:
	-----------
	pred : xarray.DataArray
		预测值
	true : xarray.DataArray
		真实值
		
	Returns:
	--------
	dict
			包含RMSE和ACC的字典
	"""
	# 确保数据形状匹配
	if pred.shape != true.shape:
		raise ValueError(f"Shape mismatch: pred {pred.shape} vs true {true.shape}")
		
	# 去除batch维度如果存在
	if 'batch' in pred.dims:
		pred = pred.squeeze('batch')
		true = true.squeeze('batch')
		
	# 将数据转换为numpy数组并展平
	pred_flat = pred.values.flatten()
	true_flat = true.values.flatten()
		
	# 计算RMSE
	rmse = np.sqrt(np.mean((pred_flat - true_flat) ** 2))
		
	# 计算ACC (Anomaly Correlation Coefficient)
	pred_anom = pred_flat - np.mean(pred_flat)
	true_anom = true_flat - np.mean(true_flat)
		
	# 使用scipy.stats计算相关系数
	acc, _ = stats.pearsonr(pred_anom, true_anom)
		
	return {'rmse': rmse, 'acc': acc}

def print_metrics(metrics):
	"""
	打印评估指标
	
	Parameters:
	-----------
	metrics : dict
			calculate_metrics函数返回的指标字典
	"""
	print("\n评估指标汇总:")
	print("-" * 60)
		
	for var_name, var_metrics in metrics.items():
		print(f"\n变量: {var_name}")
		print("-" * 40)
				
		if 'by_level' in var_metrics:
			print("各层级指标:")
			for level, level_metrics in var_metrics['by_level'].items():
				print(f"	Level {level}:")
				print(f"		RMSE: {level_metrics['rmse']:.6f}")
				print(f"		ACC:	{level_metrics['acc']:.6f}")
						
			print("\n	所有层级平均:")
			print(f"		RMSE: {var_metrics['all_levels']['rmse']:.6f}")
			print(f"		ACC:	{var_metrics['all_levels']['acc']:.6f}")
		else:
			print(f"	RMSE: {var_metrics['rmse']:.6f}")
			print(f"	ACC:	{var_metrics['acc']:.6f}")

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

all_predictions = []
for i in range(eval_steps):
    new_example_batch = example_batch.isel(time=slice(i, i+3)).copy()
    # print('new_example_batch:', new_example_batch)
    initial_times = new_example_batch.time.values
    time_step = initial_times[1] - initial_times[0]
    last_datetime = new_example_batch.datetime.isel(batch=0, time=-2).values
    print('new_example_batch.time.values:', new_example_batch.time.values, 'new_example_batch.datetime.values:', new_example_batch.datetime.values)
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
    all_predictions.append(predictions)
    
	# 计算指标
    metrics = calculate_metrics(predictions, eval_targets)

	# 打印结果
    # print_metrics(metrics)

	# 如果需要访问特定变量的指标
	# 对于没有level的变量
    # rmse = metrics['2m_temperature']['rmse']
    # acc = metrics['2m_temperature']['acc']
    # print(f'2m_temperature: rmse={rmse}, acc={acc}')

    rmse = metrics['10m_u_component_of_wind']['rmse']
    acc = metrics['10m_u_component_of_wind']['acc']
    print(f'10m_u_component_of_wind: rmse={rmse}, acc={acc}')

    rmse = metrics['10m_v_component_of_wind']['rmse']
    acc = metrics['10m_v_component_of_wind']['acc']
    print(f'10m_v_component_of_wind: rmse={rmse}, acc={acc}')
    
    rmse = metrics['wind_speed_surface']['rmse']
    acc = metrics['wind_speed_surface']['acc']
    print(f'wind_speed_surface: rmse={rmse}, acc={acc}')

	# # 对于有level的变量
	# # 访问特定level的指标
    # level_50_metrics = metrics['temperature']['by_level'][50]
	# # 访问所有level的平均指标
    # avg_metrics = metrics['temperature']['all_levels']
    # print(f'temperature: level_50_metrics={level_50_metrics}, avg_metrics={avg_metrics}')

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

# for i in range(4):	# First round: 79.15 seconds, After first round: 0.90 seconds
#	 # with trace("/tmp/jax-trace"):	# 生成性能分析文件
#	 with timer("Prediction"):
#		 predictions = rollout.chunked_prediction(
#				 run_forward_jitted,
#				 rng=jax.random.PRNGKey(0),
#				 inputs=eval_inputs,
#				 targets_template=eval_targets * np.nan,
#				 forcings=eval_forcings,
#				 verbose=True)
# 		 print("predictions:", predictions)

# # @title Choose predictions to plot

# plot_pred_variable = "2m_temperature"	# predictions.data_vars.keys()
# plot_pred_level = 500	# predictions.coords["level"].values
# plot_pred_robust = True	# True or False
# plot_pred_max_steps = predictions.dims["time"]	# min=1, max=predictions.dims["time"]

# # @title Plot predictions

# plot_size = 5
# plot_max_steps = min(predictions.dims["time"], plot_pred_max_steps)

# data = {
#		 "Targets": scale(select(eval_targets, plot_pred_variable, plot_pred_level, plot_max_steps), robust=plot_pred_robust),
#		 "Predictions": scale(select(predictions, plot_pred_variable, plot_pred_level, plot_max_steps), robust=plot_pred_robust),
#		 "Diff": scale((select(eval_targets, plot_pred_variable, plot_pred_level, plot_max_steps) -
#												 select(predictions, plot_pred_variable, plot_pred_level, plot_max_steps)),
#												robust=plot_pred_robust, center=0),
# }
# fig_title = plot_pred_variable
# if "level" in predictions[plot_pred_variable].coords:
#	 fig_title += f" at {plot_pred_level} hPa"

# plot_data(data, fig_title, plot_size, plot_pred_robust)

# # @title Loss computation (autoregressive loss over multiple steps)
# with timer("Loss"):
#	 loss, diagnostics = loss_fn_jitted(
#			 rng=jax.random.PRNGKey(0),
#			 inputs=train_inputs,
#			 targets=train_targets,
#			 forcings=train_forcings)
#	 print("Loss:", float(loss))

# # @title Gradient computation (backprop through time)
# with timer("Grads"):
#	 loss, diagnostics, next_state, grads = grads_fn_jitted(
#			 inputs=train_inputs,
#			 targets=train_targets,
#			 forcings=train_forcings)
#	 mean_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])
#	 print(f"Loss: {loss:.4f}, Mean |grad|: {mean_grad:.6f}")

# # @title Autoregressive rollout (keep the loop in JAX)
# print("Inputs:	", train_inputs.dims.mapping)
# print("Targets: ", train_targets.dims.mapping)
# print("Forcings:", train_forcings.dims.mapping)

# with timer("Prediction"):
#	 predictions = run_forward_jitted(
#			 rng=jax.random.PRNGKey(0),
#			 inputs=train_inputs,
#			 targets_template=train_targets * np.nan,
#			 forcings=train_forcings)
#	 print("predictions:", predictions)




