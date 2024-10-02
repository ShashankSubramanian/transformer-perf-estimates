# Performance Estimates for Large AI Transformer Models

This repository implements code from this [paper](https://arxiv.org/abs/2410.00273),  that analytically models the performance of transformers in three steps: 

* Count the FLOPs, #memory accesses, communication volume of each operation in the transformer, as well as the memory consumed to hold intermediate activation maps (for the backward pass) and weights. 
* Compute the theoretical time for the forward and backward pass, given underlying system (accelerator and network) characteristics.
* Search (brute-force) over all possible parallelization (and other optimization) configurations and find feasible (model fits on memory) configurations with the fastest training time.

This identifies an *optimal parallelization strategy*, given the transformer model hyperparameters, dataset hyperparameters, and the system size.  The code assumes NVIDIA GPU hardware with HBM on-device memory, two-tier of networks (fast bandwidth through NVLINK and slow bandwidth through IB/Slingshot/ethernet), and communication routines implemented via NCCL.

### Modeling different layers of the transformer 
The [`Estimates`](estimates.py) class tracks the important metrics (FLOPs, memory accesses, communication volumes, memory used for activation/weights, #GPUs used for parallelizing, #GPUs on the NVLINK/fast bandwidth domain) and implements the time computation routines (compute, memory access, communication). 
* **Modeling communication times**: The `Estimates` class also defines the basic routines for computing times in compute (given FLOPs and hardware FLOPs), memory movement (given #memory accesses and bandwidth) and assumes the roofline model for time for any operation. Additionally, the communication time is modeled with two network bandwidths (fast NVLINK and slow IB/Slingshot/ethernet). Assuming NCCL-based communication, for collective routines (such as `ReduceScatter`,`AllGather`), the effective slow bandwidth scales with #NICs (network interface cards) per NVLINK domain -- larger NVLINKs lead to larger effective slow bandwidth. We assume ring-based collective operations.

In [modules.py](modules.py), the two computational engines of the transformer are defined: `MLP` and `SA` (self-attention). Different parallelization methods are exposed here: 1D tensor parallelism, 2D SUMMA tensor parallelism, and 2D sequence/context tensor parallelism. The docstrings/comments under each module explain the mathematical expressions for the forward pass of each layer along with the tensor dimensions that are sharded (depending on the parallel strategy).

In [layers.py](layers.py), each layer of the module is implemented as a derived class of `Estimates` -- this depends on the parallelization used as well as if operations are fused (example, self-attention).  For example, the `Linear` layers have three implementations for batch matrix multiplies using different parallelization routines  (1D tensor parallelism, 2D SUMMA tensor parallelism, and 2D sequence/context tensor parallelism) and `FusedLA` implement the fused Logit-Attend operation in self-attention (similar to  `FlashAttention`).

The following is kept in mind when modeling any layer:
* The docstrings/comments for each function explain the mathematical operation of the forward pass and the backward pass, assuming a specific model parallelism implementation (1D/2D SUMMA/2D context tensor parallelism)
* `parallelism` denotes the number of GPUs used to parallelize the operation and `topology` denotes the number of those GPUs to be placed in the fast bandwidth domain (NVLINK). For example: `parallelism` of 16 and `topology` of 4 indicates groups of four GPUs are connected via NVLINK. Note that NVLINK can be used for tensor, pipeline, or data parallelism and `topology` provides flexibility in assigning these GPUs.
* The following metrics are computed for each layer, for forward and backward pass, independently: FLOPs, amount of accesses from the memory, communication volume and type (example, `ReduceScatter`,  `AllGather`,  `Broadcast`), amount of memory used on HBM (for weights and activation buffers needed for backward pass). The mathematical expressions in the comments allow you to calculate each of these systematically.
* For specific layers, the base class communication time estimates are overridden to account for overlapping with compute (example, SUMMA `Broadcast` operations overlap with much of the compute) or multiple communications for the same layer (example, context parallel fused LA `FusedLASeqp` has different `ReduceScatter` and  `AllGather` operations, SUMMA has `Broadcast` and `Reduce` in the backward pass).
* Pipeline parallel and Data parallel communications are also implemented as layers since both overlap with compute 

### Get high-level performance estimates of the transformer
In [stats.ipynb](stats.ipynb), you can first set your transformer model and obtain high-level metrics. It also shows you how to run the solver to get the optimal parallelization configuration, given the transformer, parallelization strategy (1D, 2D-SUMMA or 2D context parallel), number of GPUs, and global batch size. For example, for 1D tensor parallel, to get the top 100 configurations, here is an example code snippet (see notebook for full code):
```
# set your inputs
# transformer model with sequence length (l), embed (e), heads (h), depth (d)
model = {'l': 2048, 'e': 25600, 'h': 160, 'depth': 128}
system['nvlink_size'] = 4              # change the nvlink size if needed
parallel_strat = '1D'                  # 1D, 2D: summa, 2D-seqp: context parallel
total_gpus = 2048                      # total number of GPUs
global_batch_size = 4096               # global batch size
configs = execute_1d(model, [total_gpus], global_batch_size=global_batch_size, 
                     system=system, verbose=False, nlargest=100)
top_configs_to_print = 1 # how many configs to print? max 100 but dont print all 
pprint.pprint(configs[0][0:top_configs_to_print]) 
```
In [execution.py](execution.py), we define the search space by enumerating candidates for the different parallelization strategies and the assignment of GPUs to the NVLINK/NVSWITCH domain. In function `def totals(..)`, we take care of other optimizations such as distributed optimizer as well as pipeline bubbles when computing total time and keep track of important metrics. The `def  execute_<parallel_strat>(...)` functions loop over all candidates and keeps a priority queue of the top candidates (least training times) and returns the top *n* candidates. 

The notebook also enables you to modify the parallel configurations, allowing you to experiment with different parallelization strategies and understand the various bottlenecks. It also provides a breakdown of metrics for each layer of the transformer independently, helping you identify which layers contribute most to specific bottlenecks. See the comments in the notebook and example outputs to get a sense of this.

### Plot performance numbers
We also include notebooks to plot different performance metrics in `plots/`.
* **Overall performance (training time) vs #GPUs for different systems**: To plot overall training time as a function of #GPUs for different systems (different GPU generations, different NVLINK/NVSWITCH domain sizes), you can run [run_configs.py](run_configs.py). Given a transformer model (from [models.py](models.py), global batch size, and parallelization strategy, the code will run the solver for multiple system configurations and save the configurations in an `npy` file that you can use to plot the results. You can derive your system configurations from the ones in `systems/` which provide you with hardware numbers for different GPU generations (future generations are currently extrapolated). You can change the NVLINK/NVSWITCH domain sizes in the code. The code currently runs the solver for three GPU generations `A100`, `H200`, and `B200`,  each with three NVLINK/NVSWITCH domain sizes (4, 8, 64). Note, that the solver is parallelized with `mpi4py` and assume each system you define runs on a different process/rank. Hence, for these 9 systems above, you can run the solver, as an example, with:
	```
	mpirun -np 9 python run_configs.py --model gpt3_1T --parallel_strat 1d --global_batch_size 4096
	```
	Note that, for 2D versions of tensor parallelism, it might take several minutes to finish since the design/candidate space is large. 1D tensor parallelism is quick. 
	There are already example outputs for `gp3_1T` and `vit_era5` models (see the paper in references for the model explanations) for all the 9 systems in `outputs/*.npy`. Using them, you can directly run [plots/plots_overall_perf.ipynb](plots/plots_overall_perf.ipynb) (without generating the configs), to show strong scaling plots as well as optimal configurations, time broken down by various components, memory used on HBM as a function of #GPUs and system. Follow the notebook to get these plots. Each `.npy` file contains the following information:
	```
  nvs, t_max, t_min, n_gpus, configs = np.load('<file_name>.npy', allow_pickle=True)
	# nvs is NVLINK/NVSWITCH domain size
    # t_max is max throughput over top n configs (we've used n = 10)
    # t_min is min throughput over top n configs
    # n_gpus is list of #GPUs (could be diff based on how many min GPUs are needed to fit the model)
    # configs is list n configs: each is (throughput (float), mem (float) used, parallelization config (dict), stats (dict))
    # parallelization config contains info on DP, TP, PP etc
    # stats contains info on various time, mem components 
    # as well as assignment of GPUs to NVLINK/NVSWITCH domains
    ```
For `gpt3_1T`, the 2D tensor parallel versions are fastest (but not by a large margin for future GPU generations). For `vit_era5`, 2D tensor parallel versions are necessary.

* **Sweep parallelization configurations to understand bottlenecks**: In [plots/plots_sweep_parallel_configs.ipynb](plots/plots_sweep_parallel_configs.ipynb), we show simple plots to sweep over a range of parallel configurations and see their effect, as well as effect of NVLINK sizes, on different bottlenecks. It also provides simple visualizations of the design-space to show the subtle non-convexities in the space when searching for the optimal configurations. Different cells focus on 1D and the different 2D tensor parallelism strategies.
* **Sweep system parameters to understand system design choices**: We show simple plots to sweep over different hardware characteristics (such as FLOP rate, capacity, bandwidth, etc.) and associated scripts in [plots/plots_sweep_bw_capacity.ipynb](plots/plots_sweep_bw_capacity.ipynb) and [plots/plots_sweep_flops_cap_bw.ipynb](plots/plots_sweep_flops_cap_bw.ipynb). To run configs, see [sweeps/run_configs_sweep_bwcap.py](sweeps/run_configs_sweep_bwcap.py) etc. These run similar to [run_configs.py](run_configs.py). We have not included sample outputs for these since they generate many files, but they are run in a similar manner and we have included example plots.

## Reference
If you find this useful, please cite:
```
@misc{subramanian2024performancemodel,
      title={Comprehensive Performance Modeling and System Design Insights for Foundation Models},
      author={Shashank Subramanian and Ermal Rrapaj and Peter Harrington and Smeet Chheda and Steven Farrell and Brian Austin and Samuel Williams and Nicholas Wright and Wahid Bhimji},
      year={2024},
      eprint={2410.00273},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.00273},
}
```
