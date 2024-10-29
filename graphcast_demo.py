import dataclasses
import datetime
import functools
import math
import re
from typing import Optional
import glob

import cartopy.crs as ccrs
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
from IPython.display import HTML
import ipywidgets as widgets
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray
from jax.profiler import trace

from contextlib import contextmanager
import time


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

# @title Plotting functions

def select(
    data: xarray.Dataset,
    variable: str,
    level: Optional[int] = None,
    max_steps: Optional[int] = None
    ) -> xarray.Dataset:
  data = data[variable]
  if "batch" in data.dims:
    data = data.isel(batch=0)
  if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
    data = data.isel(time=range(0, max_steps))
  if level is not None and "level" in data.coords:
    data = data.sel(level=level)
  return data

def scale(
    data: xarray.Dataset,
    center: Optional[float] = None,
    robust: bool = False,
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
  vmin = np.nanpercentile(data, (2 if robust else 0))
  vmax = np.nanpercentile(data, (98 if robust else 100))
  if center is not None:
    diff = max(vmax - center, center - vmin)
    vmin = center - diff
    vmax = center + diff
  return (data, matplotlib.colors.Normalize(vmin, vmax),
          ("RdBu_r" if center is not None else "viridis"))

def plot_data(
    data: dict[str, xarray.Dataset],
    fig_title: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:

  first_data = next(iter(data.values()))[0]
  max_steps = first_data.sizes.get("time", 1)
  assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

  cols = min(cols, len(data))
  rows = math.ceil(len(data) / cols)
  figure = plt.figure(figsize=(plot_size * 2 * cols,
                               plot_size * rows))
  figure.suptitle(fig_title, fontsize=16)
  figure.subplots_adjust(wspace=0, hspace=0)
  figure.tight_layout()

  images = []
  for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
    ax = figure.add_subplot(rows, cols, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    im = ax.imshow(
        plot_data.isel(time=0, missing_dims="ignore"), norm=norm,
        origin="lower", cmap=cmap)
    plt.colorbar(
        mappable=im,
        ax=ax,
        orientation="vertical",
        pad=0.02,
        aspect=16,
        shrink=0.75,
        cmap=cmap,
        extend=("both" if robust else "neither"))
    images.append(im)

  def update(frame):
    if "time" in first_data.dims:
      td = datetime.timedelta(microseconds=first_data["time"][frame].item() / 1000)
      figure.suptitle(f"{fig_title}, {td}", fontsize=16)
    else:
      figure.suptitle(fig_title, fontsize=16)
    for im, (plot_data, norm, cmap) in zip(images, data.values()):
      im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))

  ani = animation.FuncAnimation(
      fig=figure, func=update, frames=max_steps, interval=250)
  plt.close(figure.number)
  return HTML(ani.to_jshtml())

# @title Choose the model

params_file_options = [
    # name for blob in gcs_bucket.list_blobs(prefix="params/")
    # if (name := blob.name.removeprefix("params/"))]  # Drop empty string.
    name for blob in glob.glob("params/**", recursive=True)
    if (name := blob.replace("params/", ""))]  # Drop empty string.
print("params_file_options:", params_file_options)

params_file = params_file_options[1]  # TODO: 0, 1, 2
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
    # if (name := blob.name.removeprefix("dataset/"))]  # Drop empty string.
    name for blob in glob.glob("dataset/**", recursive=True)
    if (name := blob.replace("dataset/", ""))]  # Drop empty string.

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

for option in dataset_file_options:
    if data_valid_for_model(option, model_config, task_config):
        dataset_file = option
        break
print("dataset_file:", dataset_file)

# @title Load weather data

if not data_valid_for_model(dataset_file, model_config, task_config):
  raise ValueError(
      "Invalid dataset file, rerun the cell above and choose a valid dataset file.")

# with gcs_bucket.blob(f"dataset/{dataset_file.value}").open("rb") as f:
with open(f"dataset/{dataset_file}", "rb") as f:
  example_batch = xarray.load_dataset(f).compute()

assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets

print(", ".join([f"{k}: {v}" for k, v in parse_file_parts(dataset_file.removesuffix(".nc")).items()]))
print("example_batch:", example_batch)

# # @title Choose data to plot

# plot_example_variable = "2m_temperature"  # example_batch.data_vars.keys()
# plot_example_level = 500  # example_batch.coords["level"].values
# plot_example_robust = True  # True or False
# plot_example_max_steps = example_batch.dims["time"]  # min=1, max=example_batch.dims["time"]

# # @title Plot example data

# plot_size = 7

# data = {
#     " ": scale(select(example_batch, plot_example_variable, plot_example_level, plot_example_max_steps),
#               robust=plot_example_robust),
# }
# fig_title = plot_example_variable
# if "level" in example_batch[plot_example_variable].coords:
#   fig_title += f" at {plot_example_level} hPa"

# plot_data(data, fig_title, plot_size, plot_example_robust)

# @title Choose training and eval data to extract
train_steps = 1  # min=1, max=example_batch.sizes["time"]-2
eval_steps = example_batch.sizes["time"]-2  # min=1, max=example_batch.sizes["time"]-2

# @title Extract training and eval data

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{train_steps*6}h"),
    **dataclasses.asdict(task_config))

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"),
    **dataclasses.asdict(task_config))

print("All Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)
print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)

# @title Load normalization data

# with gcs_bucket.blob("stats/diffs_stddev_by_level.nc").open("rb") as f:
#   diffs_stddev_by_level = xarray.load_dataset(f).compute()
# with gcs_bucket.blob("stats/mean_by_level.nc").open("rb") as f:
#   mean_by_level = xarray.load_dataset(f).compute()
# with gcs_bucket.blob("stats/stddev_by_level.nc").open("rb") as f:
#   stddev_by_level = xarray.load_dataset(f).compute()
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

print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

# with trace("/tmp/jax-trace"):  # 生成性能分析文件
with timer("Prediction"):
  predictions = rollout.chunked_prediction(
      run_forward_jitted,
      rng=jax.random.PRNGKey(0),
      inputs=eval_inputs,
      targets_template=eval_targets * np.nan,
      forcings=eval_forcings)
  print("predictions:", predictions)

# # @title Choose predictions to plot

# plot_pred_variable = "2m_temperature"  # predictions.data_vars.keys()
# plot_pred_level = 500  # predictions.coords["level"].values
# plot_pred_robust = True  # True or False
# plot_pred_max_steps = predictions.dims["time"]  # min=1, max=predictions.dims["time"]

# # @title Plot predictions

# plot_size = 5
# plot_max_steps = min(predictions.dims["time"], plot_pred_max_steps)

# data = {
#     "Targets": scale(select(eval_targets, plot_pred_variable, plot_pred_level, plot_max_steps), robust=plot_pred_robust),
#     "Predictions": scale(select(predictions, plot_pred_variable, plot_pred_level, plot_max_steps), robust=plot_pred_robust),
#     "Diff": scale((select(eval_targets, plot_pred_variable, plot_pred_level, plot_max_steps) -
#                         select(predictions, plot_pred_variable, plot_pred_level, plot_max_steps)),
#                        robust=plot_pred_robust, center=0),
# }
# fig_title = plot_pred_variable
# if "level" in predictions[plot_pred_variable].coords:
#   fig_title += f" at {plot_pred_level} hPa"

# plot_data(data, fig_title, plot_size, plot_pred_robust)

# # @title Loss computation (autoregressive loss over multiple steps)
# with timer("Loss"):
#   loss, diagnostics = loss_fn_jitted(
#       rng=jax.random.PRNGKey(0),
#       inputs=train_inputs,
#       targets=train_targets,
#       forcings=train_forcings)
#   print("Loss:", float(loss))

# # @title Gradient computation (backprop through time)
# with timer("Grads"):
#   loss, diagnostics, next_state, grads = grads_fn_jitted(
#       inputs=train_inputs,
#       targets=train_targets,
#       forcings=train_forcings)
#   mean_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])
#   print(f"Loss: {loss:.4f}, Mean |grad|: {mean_grad:.6f}")

# # @title Autoregressive rollout (keep the loop in JAX)
# print("Inputs:  ", train_inputs.dims.mapping)
# print("Targets: ", train_targets.dims.mapping)
# print("Forcings:", train_forcings.dims.mapping)

# with timer("Prediction"):
#   predictions = run_forward_jitted(
#       rng=jax.random.PRNGKey(0),
#       inputs=train_inputs,
#       targets_template=train_targets * np.nan,
#       forcings=train_forcings)
#   print("predictions:", predictions)
