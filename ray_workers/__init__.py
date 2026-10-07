import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def init_ray(**kwargs):
    """Start (or reuse) a local Ray instance whose workers can import this repo's packages."""
    import ray

    if not ray.is_initialized():
        path = os.pathsep.join(p for p in (REPO_ROOT, os.environ.get("PYTHONPATH", "")) if p)
        ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"PYTHONPATH": path}}, **kwargs)
