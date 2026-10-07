"""Ray Serve HTTP endpoint for VLA inference with server-side request batching.

POST / with JSON:
    {
      "instruction": "put the bowl on the plate",
      "state": [8 floats],
      "images": {"observation.images.image": "<base64 PNG/JPEG>", "observation.images.image2": "..."},
      "seed": 123            # optional: fixes the flow-matching noise
    }
Response: {"actions": [[...action_dim...] x chunk], "model": ..., "batch_size": ...}

Requests missing the instruction, state or any camera the model needs get HTTP 400; the server
never substitutes random inputs or outputs.
"""

import base64
import io
import random
from typing import List, Sequence

import numpy as np

from models.frames import Frame


def decode_image(b64: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def encode_image(arr: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def parse_request(body: dict, camera_keys: Sequence[str]) -> Frame:
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    task = body.get("instruction") or body.get("task")
    if not task or not isinstance(task, str):
        raise ValueError("missing 'instruction' (string)")
    if "state" not in body:
        raise ValueError("missing 'state' (list of floats)")
    images = body.get("images") or {}
    missing = [k for k in camera_keys if k not in images]
    if missing:
        raise ValueError(f"missing images for cameras {missing}")
    return Frame(images={k: decode_image(images[k]) for k in camera_keys},
                 state=np.asarray(body["state"], dtype=np.float32), task=task, source="request")


def create_app(model_name: str = "mock", device: str = "cpu", num_replicas: int = 1, max_batch_size: int = 8):
    from ray import serve
    from starlette.responses import JSONResponse

    @serve.deployment(num_replicas=num_replicas,
                      ray_actor_options={"num_cpus": 1, "num_gpus": 1 if device.startswith("cuda") else 0})
    class VLADeployment:
        def __init__(self):
            from models.load_model import load_model

            self.model = load_model(model_name, device=device)

        @serve.batch(max_batch_size=max_batch_size, batch_wait_timeout_s=0.01)
        async def predict(self, items: List[tuple]) -> List[dict]:
            frames = [f for f, _ in items]
            seeds = [s for _, s in items]
            actions = self.model.predict(frames, seeds, batch_size=len(frames))
            return [{"actions": a.tolist(), "model": self.model.name, "batch_size": len(frames)} for a in actions]

        async def __call__(self, request):
            try:
                body = await request.json()
                frame = parse_request(body, self.model.camera_keys)
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
            seed = int(body["seed"]) if "seed" in body else random.getrandbits(31)
            return await self.predict((frame, seed))

    return VLADeployment.bind()


def run_serve(model_name: str = "mock", device: str = "cpu", port: int = 8000, num_replicas: int = 1,
              max_batch_size: int = 8):
    import signal

    from ray import serve

    from ray_workers import init_ray

    init_ray()
    serve.start(http_options={"host": "0.0.0.0", "port": port})
    serve.run(create_app(model_name, device, num_replicas, max_batch_size))
    print(f"Serving {model_name} on http://localhost:{port}/ ({num_replicas} replica(s), batch <= {max_batch_size})")
    try:
        signal.pause()
    except KeyboardInterrupt:
        serve.shutdown()
