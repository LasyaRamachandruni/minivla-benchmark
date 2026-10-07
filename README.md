# MiniVLA Benchmark

A CLI for measuring what common inference optimizations (fp16 / bf16, int8 quantization, structured pruning) do to the **latency** and the **actions** of a real Vision-Language-Action policy, plus Ray tooling for batched and served inference.

- **Model:** [SmolVLA](https://huggingface.co/lerobot/smolvla_base) (LeRobot, Apache-2.0), loaded through LeRobot's own policy and processor API. It takes camera images, robot state and a language instruction and outputs a chunk of 50 future actions.
- **Data:** real frames from [LIBERO](https://huggingface.co/datasets/HuggingFaceVLA/libero) in LeRobot format (Apache-2.0), scored against the demonstrator's ground-truth actions.
- **Fair comparisons:** every optimized variant is compared with an FP32 baseline on the same device, with the same frames and the same flow-matching noise, using warmup, at least 50 timed runs, p50 / p95 and 95% confidence intervals.

## Results

**Pending GPU run.** No benchmark numbers are published yet. Everything in `results/` is produced by [`notebooks/colab_gpu_benchmark.ipynb`](notebooks/colab_gpu_benchmark.ipynb) (T4 or L4), which runs:

```bash
python cli.py sample-libero --dataset HuggingFaceVLA/libero --num-frames 500 --episodes-per-task 1 --seed 0 --horizon 10 --cache data_cache/libero_frames.npz
python cli.py optimize --model smolvla_libero --device cuda --frames-cache data_cache/libero_frames.npz --num-frames 500 --eval-batch-size 16 --N 100 --warmup 10
python cli.py optimize --model smolvla_libero --device cuda --variants fp32,bnb_int8 --frames-cache data_cache/libero_frames.npz --num-frames 500 --eval-batch-size 16 --N 100 --warmup 10 --out results/smolvla_libero_gpu_bnb_int8.json
python cli.py optimize --model smolvla_base   --device cuda --frames-cache data_cache/libero_frames.npz --num-frames 500 --eval-batch-size 16 --N 100 --warmup 10
python cli.py optimize --model smolvla_libero --device cpu  --frames-cache data_cache/libero_frames.npz --num-frames 50 --eval-batch-size 1 --N 50 --warmup 5
python cli.py compare --results "results/*.json" --markdown results/RESULTS.md
```

Once that has run, `results/RESULTS.md` has the tables and `results/*.png` the plots. Each `results/*.json` records the GPU / CPU model, torch / CUDA / LeRobot / transformers versions, the checkpoint commit, the exact frames sampled (dataset, seed, episode list), and every raw latency sample.

> An earlier version of this README listed SmolVLM-256M latencies and a "12% faster with pruning" claim. Those numbers were removed. They came from a text-only VLM rather than an action model, from random images, from an Apple-MPS baseline compared with int8 on CPU, from "pruning" that only zeroed weights (so the model stayed the same size), and from 5 timed runs.

## What is measured

| | |
|---|---|
| Eval set | 500 LIBERO frames: one seeded episode from each of the 40 tasks, frames spread evenly across those episodes (seed 0). Each frame carries the next 10 ground-truth actions, padded at the episode end. Cached to `data_cache/` so every run uses the same frames. |
| Error vs ground truth | MSE and L1 between predicted and recorded actions: for the next action and over the first 10 chunk steps, overall and per action dimension (x, y, z, roll, pitch, yaw, gripper). Only computed for a checkpoint trained on LIBERO's action space. |
| Degradation | Change in that error relative to the FP32 baseline (absolute and %). |
| Agreement with FP32 | Relative L2 error `‖a − a_fp32‖ / ‖a_fp32‖` between a variant's action chunk and the FP32 model's chunk for the same frame **and the same noise**, plus MSE / L1. This isolates the effect of the optimization and works for any checkpoint. |
| Noise floor | SmolVLA samples actions with flow matching, so FP32 itself changes when the noise changes. The suite reports FP32 vs FP32 with a different noise seed. If a variant falls below this floor, it changes the actions less than resampling the noise does. |
| Latency | Batch-1 policy forward pass + un-normalization (preprocessing is timed separately), fixed input, warmup discarded, N ≥ 50 timed runs (enforced), CUDA synchronized. Reports p50 / p95 / p99 / mean, a bootstrap 95% CI for p50 and p95, and a normal-approximation 95% CI for the mean. |
| Size / memory | Serialized `state_dict` size (this counts int8 packed weights correctly), parameter count, peak CUDA memory with the baseline offloaded, and process RSS. |

### Models

| Key | Checkpoint | Compared with LIBERO ground truth? |
|---|---|---|
| `smolvla_libero` | [`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero): SmolVLA fine-tuned on LIBERO, with the same camera, state and action features as the dataset | Yes. LIBERO has a single train split, so these frames were probably seen during fine-tuning. The error measures fit to the training data, not generalisation or task success. |
| `smolvla_base` | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base): pretrained on SO-100 community data | **No.** Its action space is 6-D joints, not LIBERO's 7-D end-effector deltas. On LIBERO frames it gets 2 of its 3 cameras and the first 6 state dims, so only latency and agreement with its own FP32 outputs are reported. |
| `mock` | random-weight model with SmolVLA-like structure | No. It is used for tests and plumbing checks only. |

Any other LeRobot policy repo id or local directory can be passed as `--model`.

### Optimizations

| Variant | Device | What it is | Caveats |
|---|---|---|---|
| `fp32` | both | Baseline. All weights are cast to FP32 (LeRobot's own default loads the VLM in bf16). | |
| `fp16`, `bf16` | GPU | VLM and action expert in half precision. The small state, action and time projections stay FP32, which is the same mixed layout LeRobot uses. | bf16 is skipped on GPUs without native bf16 (T4). |
| `int8_dynamic` | CPU | `torch.ao.quantization.quantize_dynamic`: int8 weights for `nn.Linear`, with activations quantized on the fly. Compared with FP32 on the same CPU. | Attention q/k/v/o projections stay FP32 because SmolVLA's forward reads `q_proj.weight.dtype`, which quantized modules don't have. The JSON records what fraction of the weights became int8. `torch.ao` quantization is deprecated in recent PyTorch. |
| `bnb_int8` | GPU | bitsandbytes `Linear8bitLt` for the same Linear layers, with the rest fp16. | Experimental and not yet run. |
| `pruned` | both | **Structured** pruning removes the lowest-importance 20% (`--prune-amount`) of hidden channels in every MLP (SmolVLM text layers, action expert, SigLIP vision encoder). The layers really shrink, so parameters, size and FLOPs go down. | No fine-tuning afterwards, so an accuracy drop is expected. That drop is what the suite measures. |
| `pruned+fp16`, `pruned+int8_dynamic` | GPU / CPU | combinations | |

## Usage

Python ≥ 3.12 is needed for LeRobot 0.6.1 (which pins `torch<2.12`):

```bash
pip install -e ".[lerobot,ray,test]"
```

Core commands:

```bash
python cli.py sample-libero --num-frames 500 --cache data_cache/libero_frames.npz   # fixed eval frames
python cli.py benchmark --model smolvla_libero --device cuda                        # FP32 only
python cli.py optimize  --model smolvla_libero --device cuda                        # FP32 + GPU variants
python cli.py optimize  --model smolvla_libero --device cpu --num-frames 50 --N 50  # FP32 + CPU variants
python cli.py compare   --results "results/*.json"                                  # RESULTS.md + plots
```

To try it without downloads, use the mock (synthetic frames have no ground truth, so only agreement metrics are produced):

```bash
python cli.py optimize --model mock --device cpu --synthetic --num-frames 20 --out /tmp/mock.json
```

`--N` below 50 is rejected. `--variants` takes a comma list (for example `fp32,int8_dynamic`), and `fp32` is always added as the baseline.

## Ray

All Ray paths run the real policy on the eval frames and return its actions. There is no fallback output, and missing inputs are an error.

```bash
python cli.py ray-bench --model smolvla_libero --device cpu --workers 2 --batch-size 4 --num-frames 64 --scaling-test
python cli.py pipeline-bench --model smolvla_libero --device cuda --batch-size 4 --num-frames 64
python cli.py batch --model smolvla_libero --device cuda --batch-size 16 --out preds.npz
python cli.py serve --model smolvla_libero --device cuda --max-batch-size 8
```

- `ray-bench`: data-parallel actors, each holding a model copy, processing batches of frames. The output records the core count, because on one machine workers share the same cores or GPU and throughput can't scale past the hardware.
- `pipeline-bench`: preprocess (tokenize / normalize, CPU), policy forward (CPU or GPU) and un-normalize (CPU) run as three actors. Object refs are chained so the stages overlap.
- `batch`: Ray Data `map_batches` with a stateful predictor.
- `serve`: Ray Serve with `@serve.batch`. `POST /` with `{"instruction", "state", "images": {camera_key: base64 PNG}, "seed"?}` returns the action chunk. Requests without an instruction, state or a required camera get HTTP 400.

No Ray numbers are published.

## Tests

```bash
pytest -q                  # unit tests (mock model) + integration tests if lerobot is installed
pytest -q -m integration   # LeRobot's real SmolVLA and LeRobotDataset code on tiny local fixtures
```

The integration tests build a tiny, randomly initialised SmolVLA and a tiny LIBERO-shaped LeRobot dataset offline. They check the wrapper's use of the LeRobot API: processors, noise injection, un-normalization, episode selection and ground-truth action windows with padding. They also check that every CPU optimization runs on the real SmolVLA module structure. They say nothing about the real checkpoint's accuracy or speed.

## Layout

```
cli.py                       commands: sample-libero, benchmark, optimize, compare, ray-bench, pipeline-bench, batch, serve
models/vla.py                VLAModel interface + LeRobot SmolVLA wrapper (prepare / forward / postprocess, seeded noise)
models/mock_vla.py           random-weight mock with SmolVLA-like module names
models/load_model.py         model registry
models/frames.py             Frame: images + state + instruction + ground-truth actions
pipeline/data.py             seeded LIBERO sampling via LeRobotDataset, npz cache
pipeline/optimize.py         fp16/bf16, int8 (torch.ao, bitsandbytes), structured MLP pruning
pipeline/evaluate.py         error vs ground truth, agreement with FP32, degradation
pipeline/benchmark.py        timing, latency statistics, environment capture
pipeline/suite.py            runs all variants on one device -> results JSON
pipeline/report.py           results JSON -> markdown + plots
ray_workers/                 Ray actors, pipeline stages, Ray Data, Ray Serve
notebooks/colab_gpu_benchmark.ipynb
```

## Known limitations

- Error vs ground truth is offline action-prediction error on frames the fine-tuned model has likely seen. It is not a LIBERO task success rate, which would need the simulator (`lerobot-eval --env.type=libero`).
- Pruning is applied without fine-tuning.
- CPU numbers come from whatever CPU the run used (recorded in the JSON) and are not comparable across machines.
