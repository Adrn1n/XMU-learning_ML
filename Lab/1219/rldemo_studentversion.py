"""
Machine Learning Lab: RL (Q-learning) + Visualization (Tkinter)

You MUST complete:
Task A:
  - TODO(A1): epsilon-greedy action selection
  - TODO(A2): Q-learning update rule

Task B:
  - TODO(B1): Value heatmap fill color based on V(s)=max_a Q(s,a)
  - TODO(B2): Draw greedy policy arrows for all non-terminal, non-obstacle states

Constraints:
  - Standard library only (tkinter, random, etc.)
  - Keep the program responsive (training is chunked with Tkinter .after()).
"""

import tkinter as tk
from tkinter import ttk
import random


# =========================================================
# 1) Environment: GridWorld
# =========================================================
class GridWorld:
    # Actions: 0=Up, 1=Right, 2=Down, 3=Left
    ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    ACTION_NAMES = ["Up", "Right", "Down", "Left"]

    def __init__(self, w=6, h=6):
        self.w = w
        self.h = h
        self.start = (0, 0)
        self.goal = (w - 1, h - 1)

        # Fixed obstacles for lab (do not modify unless instructed)
        self.obstacles = {
            (2, 0), (2, 1), (2, 2),
            (4, 3), (4, 4),
            (1, 4), (2, 4),
        }

        self.reset()

    def reset(self):
        self.agent = self.start
        return self.agent

    def in_bounds(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h

    def is_blocked(self, x, y):
        return (x, y) in self.obstacles

    def step(self, action: int):
        """Return: next_state, reward, done"""
        dx, dy = self.ACTIONS[action]
        x, y = self.agent
        nx, ny = x + dx, y + dy

        # Hit wall or obstacle -> stay, penalty
        if (not self.in_bounds(nx, ny)) or self.is_blocked(nx, ny):
            return self.agent, -10, False

        self.agent = (nx, ny)

        # Reached goal
        if self.agent == self.goal:
            return self.agent, 20, True

        # Step cost
        return self.agent, -1, False


# =========================================================
# 2) Agent: Tabular Q-Learning (Students complete A1/A2)
# =========================================================
class QLearningAgent:
    def __init__(self, alpha=0.25, gamma=0.95, epsilon=0.30, eps_min=0.02, eps_decay=0.9995):
        self.Q = {}  # dict: state -> [q_up, q_right, q_down, q_left]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay

    def _ensure_state(self, s):
        if s not in self.Q:
            self.Q[s] = [0.0, 0.0, 0.0, 0.0]

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def value(self, s):
        """V(s)=max_a Q(s,a)"""
        self._ensure_state(s)
        return max(self.Q[s])

    def greedy_action(self, s):
        """argmax_a Q(s,a) with random tie-break"""
        self._ensure_state(s)
        q = self.Q[s]
        m = max(q)
        best = [i for i, v in enumerate(q) if v == m]
        return random.choice(best)

    def choose_action(self, s, exploit_only=False):
        """
        TODO(A1): Implement epsilon-greedy action selection.

        Requirements:
          - If exploit_only=True: always choose greedy action (argmax Q).
          - Else: with prob epsilon choose random action; otherwise choose greedy.
        """
        self._ensure_state(s)

        # ===== TODO(A1) START =====
        # Replace the stub below with correct epsilon-greedy.
        if exploit_only:
            return self.greedy_action(s)
        # stub: random action (INCORRECT; students must fix)
        return random.randint(0, 3)
        # ===== TODO(A1) END =====

    def update(self, s, a, r, s2, done):
        """
        TODO(A2): Implement Q-learning update.

        Q-learning target:
          target = r                       if done
                   r + gamma * max_a' Q(s2,a') otherwise

        Update:
          Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))
        """
        self._ensure_state(s)
        self._ensure_state(s2)

        # ===== TODO(A2) START =====
        # Replace the stub below with the correct Q-learning update.
        # stub: no learning (INCORRECT; students must fix)
        _ = (s, a, r, s2, done)
        return
        # ===== TODO(A2) END =====


# =========================================================
# 3) GUI App (Students complete B1/B2)
# =========================================================
class RLLabApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RL Lab Starter: Q-learning GridWorld (Task A & B)")
        self.resizable(False, False)

        self.env = GridWorld(w=6, h=6)
        self.agent = QLearningAgent()

        # Episode stats
        self.episode = 0
        self.steps_in_ep = 0
        self.total_steps = 0
        self.last_reward = 0
        self.ep_return = 0
        self.returns = []  # episode return curve
        self.max_curve_points = 250

        # Control flags
        self.running = False
        self.training = False

        self._build_ui()
        self._new_episode(record_prev=False)
        self._redraw_all()

    # ---------------- UI ----------------
    def _build_ui(self):
        main = ttk.Frame(self, padding=10)
        main.grid(row=0, column=0)

        # Grid canvas
        self.cell = 70
        self.pad = 10
        wpx = self.env.w * self.cell + 2 * self.pad
        hpx = self.env.h * self.cell + 2 * self.pad
        self.grid = tk.Canvas(main, width=wpx, height=hpx, bg="white",
                              highlightthickness=1, highlightbackground="#bbb")
        self.grid.grid(row=0, column=0, rowspan=3, padx=(0, 12), pady=0)

        # Right panel
        panel = ttk.Frame(main)
        panel.grid(row=0, column=1, sticky="n")

        self.status_var = tk.StringVar(value="")
        ttk.Label(panel, text="Status").grid(row=0, column=0, sticky="w")
        ttk.Label(panel, textvariable=self.status_var, justify="left").grid(row=1, column=0, sticky="w", pady=(2, 10))

        # Controls
        ttk.Label(panel, text="Controls").grid(row=2, column=0, sticky="w")
        btns = ttk.Frame(panel)
        btns.grid(row=3, column=0, sticky="w", pady=(4, 10))

        ttk.Button(btns, text="Reset Episode", command=self.reset_episode).grid(row=0, column=0, padx=(0, 6), pady=3)
        ttk.Button(btns, text="Step (ε-greedy)", command=self.step_once).grid(row=0, column=1, padx=(0, 6), pady=3)
        ttk.Button(btns, text="Run Greedy", command=self.run_greedy).grid(row=1, column=0, padx=(0, 6), pady=3)
        ttk.Button(btns, text="Stop", command=self.stop_all).grid(row=1, column=1, padx=(0, 6), pady=3)

        # Visualization toggles (Task B)
        vis = ttk.LabelFrame(panel, text="Visualization (Task B)", padding=8)
        vis.grid(row=4, column=0, sticky="we")

        self.show_heatmap = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis, text="Show Value Heatmap (V(s)=max Q)",
                        variable=self.show_heatmap, command=self._redraw_grid).grid(row=0, column=0, sticky="w")

        self.show_value_text = tk.BooleanVar(value=False)
        ttk.Checkbutton(vis, text="Show numeric V(s) in cells",
                        variable=self.show_value_text, command=self._redraw_grid).grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.show_policy_arrows = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis, text="Show greedy policy arrows for all cells",
                        variable=self.show_policy_arrows, command=self._redraw_grid).grid(row=2, column=0, sticky="w", pady=(4, 0))

        # Training
        train = ttk.LabelFrame(panel, text="Training", padding=8)
        train.grid(row=5, column=0, sticky="we", pady=(10, 0))

        ttk.Label(train, text="Episodes:").grid(row=0, column=0, sticky="w")
        self.train_eps = tk.IntVar(value=500)
        ttk.Entry(train, textvariable=self.train_eps, width=10).grid(row=0, column=1, sticky="w")

        ttk.Label(train, text="Speed (ms/step):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.speed = tk.IntVar(value=120)
        ttk.Scale(train, from_=10, to=600, orient="horizontal", variable=self.speed)\
            .grid(row=1, column=1, sticky="we", pady=(6, 0))
        train.columnconfigure(1, weight=1)

        self.render_during_training = tk.BooleanVar(value=True)
        ttk.Checkbutton(train, text="Render during training", variable=self.render_during_training)\
            .grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Button(train, text="Train (non-blocking)", command=self.train_start)\
            .grid(row=3, column=0, columnspan=2, sticky="we", pady=(8, 0))

        # Return curve canvas
        curve = ttk.LabelFrame(panel, text="Episode Return Curve", padding=8)
        curve.grid(row=6, column=0, sticky="we", pady=(10, 0))

        self.curve_w, self.curve_h = 320, 160
        self.curve = tk.Canvas(curve, width=self.curve_w, height=self.curve_h,
                               bg="white", highlightthickness=1, highlightbackground="#bbb")
        self.curve.grid(row=0, column=0, sticky="we")

    # ---------------- Geometry & Color helpers ----------------
    def _cell_rect(self, x, y):
        x0 = self.pad + x * self.cell
        y0 = self.pad + y * self.cell
        return x0, y0, x0 + self.cell, y0 + self.cell

    @staticmethod
    def _lerp(a, b, t):
        return int(a + (b - a) * t)

    @staticmethod
    def _rgb_hex(r, g, b):
        return f"#{r:02x}{g:02x}{b:02x}"

    def _value_to_color(self, v, vmin, vmax):
        """Map v in [vmin,vmax] to a blue colormap (light -> dark)."""
        if vmax <= vmin + 1e-12:
            t = 0.0
        else:
            t = (v - vmin) / (vmax - vmin)
            t = max(0.0, min(1.0, t))
        r0, g0, b0 = (247, 251, 255)  # light
        r1, g1, b1 = (8, 48, 107)     # dark
        return self._rgb_hex(self._lerp(r0, r1, t), self._lerp(g0, g1, t), self._lerp(b0, b1, t))

    def _collect_values(self):
        """Collect V(s) for all non-obstacle states; return dict, vmin, vmax."""
        vals = {}
        for y in range(self.env.h):
            for x in range(self.env.w):
                s = (x, y)
                if s in self.env.obstacles:
                    continue
                vals[s] = self.agent.value(s)  # ensures Q initialized
        if not vals:
            return {}, 0.0, 1.0
        vmin = min(vals.values())
        vmax = max(vals.values())
        return vals, vmin, vmax

    # ---------------- Drawing ----------------
    def _redraw_all(self):
        self._redraw_grid()
        self._redraw_curve()
        self._update_status()

    def _redraw_grid(self):
        self.grid.delete("all")

        vals, vmin, vmax = self._collect_values()

        for y in range(self.env.h):
            for x in range(self.env.w):
                s = (x, y)
                x0, y0, x1, y1 = self._cell_rect(x, y)

                # Base fill
                if s in self.env.obstacles:
                    fill = "#444"
                elif s == self.env.goal:
                    fill = "#74c476"
                elif s == self.env.start:
                    fill = "#c6dbef"
                else:
                    # ===== TODO(B1) START =====
                    # If show_heatmap is ON, set fill color based on V(s)=max_a Q(s,a).
                    # Use: self._value_to_color(v, vmin, vmax) where v = vals.get(s, 0.0)
                    # Otherwise set fill = "white".
                    fill = "white"  # stub (INCORRECT when heatmap is enabled)
                    # ===== TODO(B1) END =====

                self.grid.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#bbb")

                # Optional numeric V(s)
                if self.show_value_text.get() and (s not in self.env.obstacles):
                    v = vals.get(s, 0.0)
                    self.grid.create_text((x0 + x1) / 2, (y0 + y1) / 2,
                                          text=f"{v:.1f}", fill="#111",
                                          font=("Arial", 11, "bold"))

        # ===== TODO(B2) START =====
        # Draw greedy policy arrows for all cells (optional).
        # Requirements:
        #  - Only when self.show_policy_arrows.get() is True.
        #  - Skip obstacles and the goal cell.
        #  - For each remaining state s, compute a = argmax_a Q(s,a)
        #    (you can call self.agent.greedy_action(s)).
        #  - Draw a short arrow centered in the cell using create_line(..., arrow=tk.LAST).
        #  - Use dx,dy mapping: Up=(0,-d), Right=(d,0), Down=(0,d), Left=(-d,0)
        #
        # NOTE: a correct implementation should show a coherent flow toward goal after training.
        # ===== TODO(B2) END =====

        # Draw agent on top
        ax, ay = self.env.agent
        x0, y0, x1, y1 = self._cell_rect(ax, ay)
        m = 12
        self.grid.create_oval(x0 + m, y0 + m, x1 - m, y1 - m, fill="#e34a33", outline="")

        # Emphasize current greedy direction (thicker arrow)
        s = self.env.agent
        a = self.agent.greedy_action(s)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        d = 18
        dx, dy = {0: (0, -d), 1: (d, 0), 2: (0, d), 3: (-d, 0)}[a]
        self.grid.create_line(cx, cy, cx + dx, cy + dy, width=3, arrow=tk.LAST)

        self._update_status()

    def _redraw_curve(self):
        c = self.curve
        c.delete("all")
        w, h = self.curve_w, self.curve_h
        margin = 18

        # axes
        c.create_line(margin, h - margin, w - margin, h - margin)
        c.create_line(margin, margin, margin, h - margin)

        data = self.returns[-self.max_curve_points:]
        if not data:
            c.create_text(w/2, h/2, text="No completed episodes yet", fill="#666")
            return

        ymin, ymax = min(data), max(data)
        if abs(ymax - ymin) < 1e-9:
            ymax = ymin + 1.0

        # polyline
        n = len(data)
        xs, ys = [], []
        for i, r in enumerate(data):
            x = margin + (w - 2*margin) * (i / max(1, n - 1))
            t = (r - ymin) / (ymax - ymin)
            y = (h - margin) - (h - 2*margin) * t
            xs.append(x); ys.append(y)

        for i in range(n - 1):
            c.create_line(xs[i], ys[i], xs[i+1], ys[i+1], width=2)

        # marker
        c.create_oval(xs[-1]-3, ys[-1]-3, xs[-1]+3, ys[-1]+3, fill="#111", outline="")

        # label
        k = min(20, n)
        avg = sum(data[-k:]) / k
        c.create_text(w - margin, margin, text=f"Last={data[-1]:.1f}  Avg({k})={avg:.1f}",
                      fill="#111", anchor="ne")

    def _update_status(self):
        self.status_var.set(
            f"Episode started: {self.episode}\n"
            f"Steps in episode: {self.steps_in_ep}\n"
            f"Episode return: {self.ep_return}\n"
            f"Total steps: {self.total_steps}\n"
            f"Epsilon: {self.agent.epsilon:.3f}\n"
            f"Last reward: {self.last_reward}\n"
            f"Agent state: {self.env.agent}\n"
            f"Goal: {self.env.goal}"
        )

    # ---------------- Episode & Interaction ----------------
    def _new_episode(self, record_prev: bool):
        if record_prev and self.steps_in_ep > 0:
            self.returns.append(self.ep_return)
            if len(self.returns) > 5000:
                self.returns = self.returns[-5000:]
            self._redraw_curve()

        self.env.reset()
        self.episode += 1
        self.steps_in_ep = 0
        self.ep_return = 0
        self.last_reward = 0

    def reset_episode(self):
        self._new_episode(record_prev=False)
        self._redraw_all()

    def step_once(self):
        s = self.env.agent
        a = self.agent.choose_action(s, exploit_only=False)
        s2, r, done = self.env.step(a)

        self.agent.update(s, a, r, s2, done)
        self.agent.decay_epsilon()

        self.steps_in_ep += 1
        self.total_steps += 1
        self.last_reward = r
        self.ep_return += r

        if done:
            self._new_episode(record_prev=True)

        self._redraw_grid()

    def run_greedy(self):
        if self.running or self.training:
            return
        self.running = True
        self._run_greedy_loop()

    def _run_greedy_loop(self):
        if not self.running:
            return

        s = self.env.agent
        a = self.agent.choose_action(s, exploit_only=True)
        _, r, done = self.env.step(a)

        self.steps_in_ep += 1
        self.total_steps += 1
        self.last_reward = r
        self.ep_return += r

        if done:
            self.running = False
            self._new_episode(record_prev=True)

        self._redraw_grid()
        self.after(max(10, int(self.speed.get())), self._run_greedy_loop)

    def stop_all(self):
        self.running = False
        self.training = False

    # ---------------- Training (non-blocking) ----------------
    def train_start(self):
        if self.running or self.training:
            return
        try:
            target = int(self.train_eps.get())
            if target <= 0:
                return
        except Exception:
            return
        self.training = True
        self._train_remaining = target
        self._train_chunk()

    def _train_chunk(self):
        if not self.training:
            return
        if self._train_remaining <= 0:
            self.training = False
            self._redraw_all()
            return

        # chunk episodes per tick
        chunk = min(40, self._train_remaining)
        for _ in range(chunk):
            self._train_one_episode()

        self._train_remaining -= chunk

        if self.render_during_training.get():
            self._redraw_all()
        else:
            self._redraw_curve()
            self._update_status()

        self.after(1, self._train_chunk)

    def _train_one_episode(self):
        self.env.reset()
        self.episode += 1
        ep_ret = 0
        max_steps = 200

        for _ in range(max_steps):
            s = self.env.agent
            a = self.agent.choose_action(s, exploit_only=False)
            s2, r, done = self.env.step(a)

            self.agent.update(s, a, r, s2, done)
            self.agent.decay_epsilon()

            ep_ret += r
            if done:
                break

        self.returns.append(ep_ret)
        if len(self.returns) > 5000:
            self.returns = self.returns[-5000:]


if __name__ == "__main__":
    app = RLLabApp()
    app.mainloop()
