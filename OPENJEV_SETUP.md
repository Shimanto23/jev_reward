# OpenJEV FP8 for CoRe preference labels

CoRe uses OpenJEV for pairwise video preference labels and per-frame progress
scores. FRM needs a chat model only when `use_FRM_online=true`, because it then
generates reward-function Python code. OpenJEV's `/v1/systemone` endpoint is
used separately for RRM labels.

## Serve OpenJEV

The FP8 checkpoint's serving recipe calls for vLLM 0.29.0, its helper shim, and
the settings below. Use a serving environment that meets OpenJEV's pinned
dependencies; the project's training environment pins an older `openai` package.
Download the FP8 model and the `helper/shim.py` file from the main OpenJEV
repository, then run:

```bash
export OPENJEV_MODEL_DIR=/path/to/openjev-FP8
export OPENJEV_HELPER_PATH=/path/to/openjev/helper/shim.py
export OPENJEV_PYTHON=/path/to/openjev-serving-env/bin/python
bash scripts/serve-openjev-fp8.sh
```

The script defaults to a 16,384-token context and one active sequence. The
model card's 16,384-token, 16-sequence command was checked on an 80 GB H100;
the context limit and one-image setting are part of the helper's supported
request limits, so this keeps the full context while lowering concurrency. The
FP8 weights are about 29 GB, while an L40S has 48 GB, so the weights fit on
paper but actual headroom depends on vLLM's runtime allocations and anything
else using the GPU. Make sure the GPU is otherwise idle; running CoRe training
on that same card at the same time may run out of memory. If the server starts
with memory to spare, increase concurrency with `OPENJEV_MAX_NUM_SEQS`.
`OPENJEV_GPU_MEMORY_UTILIZATION` can also be adjusted if vLLM reports an
allocation error.

Keep this process running. The helper listens on `127.0.0.1:3000`, and vLLM
listens on `127.0.0.1:8000`. For a remote helper, set `OPENJEV_BASE_URL` to its
reachable URL and configure `OPENJEV_TOKEN` if the helper requires a token.

## Run CoRe

Training scripts set `RRM_LABEL_BACKEND=openjev` and use
`OPENJEV_BASE_URL=http://127.0.0.1:3000` by default. The helper must be running
before launching a training script. The included launch scripts set
`use_FRM_online=false` and use their checked-in FRM reward functions, so they do
not need a Qwen server. If you enable online FRM generation, set
`FRM_LLM_MODEL` and `VLLM_BASE_URL` to a compatible chat model service.

You can override the label endpoint without editing a training script:

```bash
export OPENJEV_BASE_URL=http://127.0.0.1:3000
export OPENJEV_MODEL=openjev
export OPENJEV_SKIP_MARGIN=0.05
```

The adapter sends one composite image containing the task's initial and goal
references plus three sampled frames from each clip. OpenJEV returns a typed
A/B choice and typed progress scores; close A/B preference probabilities are
recorded as skipped labels. This changes the input and scoring method from the
old free-form Qwen prompts, so compare labeling behavior on representative
trajectories before interpreting experiment results as directly comparable.

The OpenJEV model weights are licensed CC BY-NC 4.0. Check that license for the
intended use.
