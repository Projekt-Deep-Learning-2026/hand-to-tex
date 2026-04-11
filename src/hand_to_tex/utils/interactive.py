import time
import tkinter as tk

import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from hand_to_tex.datasets.dataset import _HMEDatasetBase
from hand_to_tex.datasets.ink_data import InkData


class HMEDrawingApp:
    def __init__(self, master, model, device):
        self.master = master
        self.model = model
        self.device = device

        self.master.title("HME Interactive Inference")
        self.master.geometry("800x650")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.canvas = tk.Canvas(self.master, bg="white", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.btn_frame = tk.Frame(self.master)
        self.btn_frame.pack(fill=tk.X, padx=10, pady=5)

        self.btn_predict = tk.Button(
            self.btn_frame,
            text="Generate TeX",
            command=self.predict,
            bg="lightgreen",
            font=("Arial", 14, "bold"),
        )
        self.btn_predict.pack(side=tk.LEFT, padx=5)

        self.btn_clear = tk.Button(
            self.btn_frame, text="Clear", command=self.clear, font=("Arial", 14)
        )
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        self.btn_undo = tk.Button(
            self.btn_frame, text="Undo", command=self.undo_last_trace, font=("Arial", 14)
        )
        self.btn_undo.pack(side=tk.LEFT, padx=5)

        self.fig, self.ax = plt.subplots(figsize=(6, 1.5))
        self.fig.patch.set_facecolor("#f0f0f0")
        self.ax.axis("off")

        self.text_obj = self.ax.text(
            0.5, 0.5, "Draw a symbol and press Generate", ha="center", va="center", fontsize=18
        )

        self.latex_canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.latex_canvas.get_tk_widget().pack(pady=10)

        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

        self.traces = []
        self.current_trace = []
        self.start_time = None
        self.last_x = None
        self.last_y = None

    def start_stroke(self, event):
        if self.start_time is None:
            self.start_time = time.time()

        t = time.time() - self.start_time
        self.current_trace = [(float(event.x), float(event.y), t)]
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        t = time.time() - self.start_time  # type: ignore
        self.current_trace.append((float(event.x), float(event.y), t))

        self.canvas.create_line(
            self.last_x,  # type:ignore
            self.last_y,  # type:ignore
            event.x,
            event.y,
            width=3,
            fill="black",
            capstyle=tk.ROUND,
            smooth=tk.TRUE,
        )
        self.last_x = event.x
        self.last_y = event.y

    def end_stroke(self, event):
        if len(self.current_trace) > 1:
            self.traces.append(self.current_trace)
        self.current_trace = []

    def _redraw_traces(self):
        self.canvas.delete("all")

        for trace in self.traces:
            for i in range(1, len(trace)):
                x1, y1, _ = trace[i - 1]
                x2, y2, _ = trace[i]
                self.canvas.create_line(
                    x1,
                    y1,
                    x2,
                    y2,
                    width=3,
                    fill="black",
                    capstyle=tk.ROUND,
                    smooth=tk.TRUE,
                )

    def undo_last_trace(self):
        if self.current_trace:
            self.current_trace = []

        if self.traces:
            self.traces.pop()
            self._redraw_traces()

    def clear(self):
        self.canvas.delete("all")
        self.traces = []
        self.current_trace = []
        self.start_time = None

        # Reset silnika renderującego
        self.text_obj.set_color("black")
        self.text_obj.set_text("Draw a symbol and press Generate")
        self.latex_canvas.draw()

    def predict(self):
        if not self.traces:
            self.text_obj.set_color("red")
            self.text_obj.set_text("No traces found!")
            self.latex_canvas.draw()
            return

        ink = InkData(
            tag="test", sample_id="interactive_demo", tex_raw="", tex_norm="", traces=self.traces
        )

        features = _HMEDatasetBase.extract_features(ink)

        if features.size(0) == 0:
            self.text_obj.set_color("red")
            self.text_obj.set_text("Invalid traces")
            self.latex_canvas.draw()
            return

        features_batched = features.unsqueeze(0).to(self.device)
        lengths = torch.tensor([features.size(0)], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            generated_tokens = self.model.generate(src=features_batched, src_lengths=lengths)

        predicted_expr = self.model._to_expr(generated_tokens[0])

        self.text_obj.set_color("blue")
        self.text_obj.set_text(rf"${predicted_expr}$")
        self.latex_canvas.draw()

    def on_closing(self):
        """Cleanly exit the application and release Matplotlib memory."""
        plt.close(self.fig)
        self.master.quit()
        self.master.destroy()
