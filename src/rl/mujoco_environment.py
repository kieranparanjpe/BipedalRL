import signal
from contextlib import nullcontext
from typing import Callable, Optional

from .environment import Environment
import mujoco
import mujoco.viewer

class MujocoEnvironment(Environment):

    def __init__(self, model : mujoco.MjModel, data : mujoco.MjData, on_key : Optional[Callable[[int], None]] = None,
                 use_viewer=True):
        self.model = model
        self.data = data

        self._viewer_ctx = None
        self.viewer = None
        self.use_viewer = use_viewer
        self.render_enabled = True
        self.on_key = on_key
        self.interrupt = False
        signal.signal(signal.SIGINT, self._handle_sigint)

    def set_on_key(self, on_key : Optional[Callable[[int], None]]):
        self.on_key = on_key

    def _on_key(self, keycode: int):
        # Example: V key toggles rendering
        if keycode in (ord('v'), ord('V')):
            self.render_enabled = not self.render_enabled
        self.on_key(keycode)

    def __enter__(self):
        self._viewer_ctx = (
            mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._on_key)
            if self.use_viewer
            else nullcontext(None)
        )
        self.viewer = self._viewer_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._viewer_ctx is not None:
            self._viewer_ctx.__exit__(exc_type, exc, tb)
        self.viewer = None
        self._viewer_ctx = None

    def step(self):
        mujoco.mj_step(self.model, self.data)

        if self.viewer is not None and self.render_enabled:
            self.viewer.sync()

    def is_running(self):
        if self.interrupt:
            return False

        return self.viewer is None or self.viewer.is_running()

    def _handle_sigint(self, sig, frame):
        self.interrupt = True

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
