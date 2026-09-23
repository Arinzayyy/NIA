"""capture.py -- the frame source for NIA's eyes.

Swappable input (PRD Section 3): the live path is the PS5 through a capture
card; the dev path is screen capture so you can test against a game window or a
video on the PC. Either way the rest of the pipeline is identical.

  source: capture_card  -> OpenCV VideoCapture on a device index (the card)
  source: screen        -> grab a monitor (or a region of it) via mss

Downscaling happens here, once, because it is the main cost/latency lever before
anything else touches the frame.
"""


class FrameSource:
    def __init__(self, config):
        c = config.get("capture", {})
        self.source = c.get("source", "screen")
        self.device_index = c.get("device_index", 0)
        self.max_width = c.get("max_width", 768)
        self.monitor = c.get("monitor", 1)     # screen: 1 = primary monitor
        self.region = c.get("region")          # screen: optional [x, y, w, h]
        self._cap = None
        self._sct = None

    def _ensure(self):
        if self.source == "capture_card":
            if self._cap is None:
                import cv2
                self._cap = cv2.VideoCapture(self.device_index)
        else:  # screen
            if self._sct is None:
                import mss
                self._sct = mss.mss()

    def get_frame(self):
        """Return the latest frame as a BGR numpy array, or None on failure."""
        import numpy as np
        self._ensure()
        if self.source == "capture_card":
            ok, frame = self._cap.read()
            return frame if ok else None
        # screen grab
        if self.region:
            x, y, w, h = self.region
            mon = {"left": x, "top": y, "width": w, "height": h}
        else:
            mon = self._sct.monitors[self.monitor]
        img = self._sct.grab(mon)
        arr = np.array(img)        # BGRA
        return arr[:, :, :3]       # drop alpha -> BGR

    def downscale(self, frame):
        """Shrink to max_width (keeps aspect). The one place size is reduced."""
        if frame is None:
            return None
        import cv2
        h, w = frame.shape[:2]
        if w <= self.max_width:
            return frame
        scale = self.max_width / float(w)
        return cv2.resize(frame, (self.max_width, int(h * scale)))

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
