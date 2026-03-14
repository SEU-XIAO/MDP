from __future__ import annotations

import json
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any


@dataclass
class EnemyConfig:
    enemy_id: int
    pos: list[int]
    comment: str = ""
    detection_zones: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"r": 4, "p": 0.95},
            {"r": 6, "p": 0.60},
            {"r": 8, "p": 0.30},
        ]
    )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.enemy_id,
            "pos": self.pos,
            "detection_shape": "circle",
            "comment": self.comment,
            "detection_zones": self.detection_zones,
        }


class ScenarioDesignerApp:
    GRID_SIZE = 64
    CELL_SIZE = 10
    CANVAS_SIZE = GRID_SIZE * CELL_SIZE

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("64x64 Scenario Designer")

        self.start_pos = [1, 62]
        self.goal_pos = [62, 1]
        self.enemies: list[EnemyConfig] = [EnemyConfig(enemy_id=1, pos=[8, 50])]

        self.mode = tk.StringVar(value="move")
        self.selected_enemy_index: int | None = 0
        self.dragging_enemy_index: int | None = None

        self._build_ui()
        self._refresh_enemy_list()
        self._draw_scene()

    def _build_ui(self) -> None:
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=8, pady=8)

        right = tk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Mode control
        mode_frame = tk.LabelFrame(left, text="Interaction Mode")
        mode_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Radiobutton(mode_frame, text="Move Enemy", variable=self.mode, value="move").pack(anchor="w")
        tk.Radiobutton(mode_frame, text="Set Start", variable=self.mode, value="start").pack(anchor="w")
        tk.Radiobutton(mode_frame, text="Set Goal", variable=self.mode, value="goal").pack(anchor="w")

        # Enemy list and controls
        enemy_frame = tk.LabelFrame(left, text="Enemies")
        enemy_frame.pack(fill=tk.BOTH, expand=True)

        self.enemy_listbox = tk.Listbox(enemy_frame, width=34, height=10)
        self.enemy_listbox.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.enemy_listbox.bind("<<ListboxSelect>>", self._on_select_enemy)

        btn_row = tk.Frame(enemy_frame)
        btn_row.pack(fill=tk.X, padx=4, pady=4)
        tk.Button(btn_row, text="Add Enemy", command=self._add_enemy).pack(side=tk.LEFT)
        tk.Button(btn_row, text="Remove", command=self._remove_enemy).pack(side=tk.LEFT, padx=(6, 0))

        # Enemy details
        detail_frame = tk.LabelFrame(left, text="Selected Enemy Details")
        detail_frame.pack(fill=tk.X, pady=(8, 8))

        tk.Label(detail_frame, text="Comment").pack(anchor="w", padx=4, pady=(4, 0))
        self.comment_var = tk.StringVar(value="")
        tk.Entry(detail_frame, textvariable=self.comment_var, width=32).pack(fill=tk.X, padx=4)

        tk.Label(detail_frame, text="Detection Zones JSON").pack(anchor="w", padx=4, pady=(4, 0))
        self.zones_text = tk.Text(detail_frame, height=6, width=32)
        self.zones_text.pack(fill=tk.BOTH, padx=4, pady=(0, 4))

        tk.Button(detail_frame, text="Apply Enemy Changes", command=self._apply_enemy_changes).pack(
            anchor="e", padx=4, pady=(0, 6)
        )

        # Scenario controls
        scenario_frame = tk.LabelFrame(left, text="Scenario")
        scenario_frame.pack(fill=tk.X)

        self.start_label = tk.Label(scenario_frame, text="Start: [1, 62]")
        self.start_label.pack(anchor="w", padx=4, pady=(4, 0))
        self.goal_label = tk.Label(scenario_frame, text="Goal: [62, 1]")
        self.goal_label.pack(anchor="w", padx=4)

        io_row = tk.Frame(scenario_frame)
        io_row.pack(fill=tk.X, padx=4, pady=6)
        tk.Button(io_row, text="Load JSON", command=self._load_json).pack(side=tk.LEFT)
        tk.Button(io_row, text="Save JSON", command=self._save_json).pack(side=tk.LEFT, padx=(6, 0))

        # Canvas
        self.canvas = tk.Canvas(
            right,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            bg="#ffffff",
            highlightthickness=1,
            highlightbackground="#888888",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

    def _grid_to_canvas(self, x: int, y: int) -> tuple[int, int]:
        return x * self.CELL_SIZE + self.CELL_SIZE // 2, y * self.CELL_SIZE + self.CELL_SIZE // 2

    def _canvas_to_grid(self, px: int, py: int) -> tuple[int, int]:
        gx = max(0, min(self.GRID_SIZE - 1, px // self.CELL_SIZE))
        gy = max(0, min(self.GRID_SIZE - 1, py // self.CELL_SIZE))
        return gx, gy

    def _draw_scene(self) -> None:
        self.canvas.delete("all")

        # Grid lines
        for i in range(self.GRID_SIZE + 1):
            color = "#d0d0d0" if i % 8 else "#b0b0b0"
            w = 1 if i % 8 else 2
            p = i * self.CELL_SIZE
            self.canvas.create_line(0, p, self.CANVAS_SIZE, p, fill=color, width=w)
            self.canvas.create_line(p, 0, p, self.CANVAS_SIZE, fill=color, width=w)

        # Enemies and detection circles
        for idx, enemy in enumerate(self.enemies):
            ex, ey = enemy.pos
            cx, cy = self._grid_to_canvas(ex, ey)

            zone_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c"]
            for zi, zone in enumerate(sorted(enemy.detection_zones, key=lambda z: z["r"], reverse=True)):
                radius_px = int(zone["r"] * self.CELL_SIZE)
                color = zone_colors[zi % len(zone_colors)]
                self.canvas.create_oval(
                    cx - radius_px,
                    cy - radius_px,
                    cx + radius_px,
                    cy + radius_px,
                    outline=color,
                    width=2,
                )
                self.canvas.create_text(cx + radius_px + 12, cy, text=f"p={zone['p']:.2f}", fill=color, anchor="w")

            dot_r = 5 if idx != self.selected_enemy_index else 7
            dot_color = "#0047ab" if idx != self.selected_enemy_index else "#6a00ff"
            self.canvas.create_oval(cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r, fill=dot_color, outline="")
            self.canvas.create_text(cx + 8, cy - 8, text=f"E{enemy.enemy_id}", fill=dot_color, anchor="w")

        # Start and goal
        sx, sy = self._grid_to_canvas(self.start_pos[0], self.start_pos[1])
        gx, gy = self._grid_to_canvas(self.goal_pos[0], self.goal_pos[1])

        self.canvas.create_rectangle(
            sx - 6,
            sy - 6,
            sx + 6,
            sy + 6,
            fill="#16a34a",
            outline="#0f7a38",
            width=2,
        )
        self.canvas.create_text(sx + 10, sy + 10, text="Start", fill="#0f7a38", anchor="w")

        self.canvas.create_rectangle(
            gx - 6,
            gy - 6,
            gx + 6,
            gy + 6,
            fill="#ef4444",
            outline="#b91c1c",
            width=2,
        )
        self.canvas.create_text(gx + 10, gy + 10, text="Goal", fill="#b91c1c", anchor="w")

        self.start_label.config(text=f"Start: {self.start_pos}")
        self.goal_label.config(text=f"Goal: {self.goal_pos}")

    def _refresh_enemy_list(self) -> None:
        self.enemy_listbox.delete(0, tk.END)
        for i, enemy in enumerate(self.enemies):
            text = f"#{enemy.enemy_id} pos={enemy.pos} zones={len(enemy.detection_zones)}"
            self.enemy_listbox.insert(tk.END, text)
            if i == self.selected_enemy_index:
                self.enemy_listbox.selection_set(i)

        self._populate_enemy_details()

    def _populate_enemy_details(self) -> None:
        if self.selected_enemy_index is None or self.selected_enemy_index >= len(self.enemies):
            self.comment_var.set("")
            self.zones_text.delete("1.0", tk.END)
            return

        enemy = self.enemies[self.selected_enemy_index]
        self.comment_var.set(enemy.comment)
        self.zones_text.delete("1.0", tk.END)
        self.zones_text.insert("1.0", json.dumps(enemy.detection_zones, ensure_ascii=False, indent=2))

    def _add_enemy(self) -> None:
        next_id = (max((e.enemy_id for e in self.enemies), default=0) + 1)
        self.enemies.append(EnemyConfig(enemy_id=next_id, pos=[self.GRID_SIZE // 2, self.GRID_SIZE // 2]))
        self.selected_enemy_index = len(self.enemies) - 1
        self._refresh_enemy_list()
        self._draw_scene()

    def _remove_enemy(self) -> None:
        if self.selected_enemy_index is None or not self.enemies:
            return
        del self.enemies[self.selected_enemy_index]
        if not self.enemies:
            self.selected_enemy_index = None
        else:
            self.selected_enemy_index = max(0, self.selected_enemy_index - 1)
        self._refresh_enemy_list()
        self._draw_scene()

    def _on_select_enemy(self, _event: Any) -> None:
        selected = self.enemy_listbox.curselection()
        if not selected:
            return
        self.selected_enemy_index = int(selected[0])
        self._populate_enemy_details()
        self._draw_scene()

    def _apply_enemy_changes(self) -> None:
        if self.selected_enemy_index is None or self.selected_enemy_index >= len(self.enemies):
            return

        enemy = self.enemies[self.selected_enemy_index]
        zone_raw = self.zones_text.get("1.0", tk.END).strip()

        try:
            zones = json.loads(zone_raw)
            if not isinstance(zones, list):
                raise ValueError("detection_zones must be a list")
            validated: list[dict[str, float]] = []
            for z in zones:
                if not isinstance(z, dict) or "r" not in z or "p" not in z:
                    raise ValueError("each zone must have r and p")
                r = float(z["r"])
                p = float(z["p"])
                if r <= 0:
                    raise ValueError("r must be > 0")
                if p < 0 or p > 1:
                    raise ValueError("p must be in [0, 1]")
                validated.append({"r": r, "p": p})

            enemy.comment = self.comment_var.get().strip()
            enemy.detection_zones = validated
            self._refresh_enemy_list()
            self._draw_scene()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Invalid detection_zones", str(exc))

    def _pick_enemy_at(self, gx: int, gy: int) -> int | None:
        best_idx: int | None = None
        best_dist = float("inf")
        for idx, enemy in enumerate(self.enemies):
            ex, ey = enemy.pos
            d = (gx - ex) ** 2 + (gy - ey) ** 2
            if d < best_dist and d <= 4:
                best_dist = d
                best_idx = idx
        return best_idx

    def _on_canvas_click(self, event: tk.Event[Any]) -> None:
        gx, gy = self._canvas_to_grid(event.x, event.y)

        if self.mode.get() == "start":
            self.start_pos = [gx, gy]
            self._draw_scene()
            return

        if self.mode.get() == "goal":
            self.goal_pos = [gx, gy]
            self._draw_scene()
            return

        picked = self._pick_enemy_at(gx, gy)
        if picked is not None:
            self.selected_enemy_index = picked
            self.dragging_enemy_index = picked
            self._refresh_enemy_list()
            self._draw_scene()

    def _on_canvas_drag(self, event: tk.Event[Any]) -> None:
        if self.mode.get() != "move":
            return
        if self.dragging_enemy_index is None:
            return

        gx, gy = self._canvas_to_grid(event.x, event.y)
        self.enemies[self.dragging_enemy_index].pos = [gx, gy]
        self._refresh_enemy_list()
        self._draw_scene()

    def _on_canvas_release(self, _event: tk.Event[Any]) -> None:
        self.dragging_enemy_index = None

    def _load_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Load scenario JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with Path(path).open("r", encoding="utf-8") as f:
                data = json.load(f)

            m = data.get("map", {})
            start = m.get("start_pos", [1, 62])
            goal = m.get("goal_pos", [62, 1])

            self.start_pos = [int(max(0, min(self.GRID_SIZE - 1, start[0]))), int(max(0, min(self.GRID_SIZE - 1, start[1])))]
            self.goal_pos = [int(max(0, min(self.GRID_SIZE - 1, goal[0]))), int(max(0, min(self.GRID_SIZE - 1, goal[1])))]

            enemies: list[EnemyConfig] = []
            for e in data.get("enemies", []):
                ex, ey = e.get("pos", [self.GRID_SIZE // 2, self.GRID_SIZE // 2])
                enemy = EnemyConfig(
                    enemy_id=int(e.get("id", len(enemies) + 1)),
                    pos=[int(max(0, min(self.GRID_SIZE - 1, ex))), int(max(0, min(self.GRID_SIZE - 1, ey)))],
                    comment=str(e.get("comment", "")),
                    detection_zones=e.get("detection_zones", []),
                )
                enemies.append(enemy)

            self.enemies = enemies or [EnemyConfig(enemy_id=1, pos=[8, 50])]
            self.selected_enemy_index = 0 if self.enemies else None

            self._refresh_enemy_list()
            self._draw_scene()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Load failed", str(exc))

    def _to_json_dict(self) -> dict[str, Any]:
        return {
            "map": {
                "grid_size": self.GRID_SIZE,
                "start_pos": self.start_pos,
                "goal_pos": self.goal_pos,
            },
            "enemies": [enemy.to_json() for enemy in self.enemies],
        }

    def _save_json(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save scenario JSON",
            defaultextension=".json",
            initialfile="scenario_64.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        data = self._to_json_dict()
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        messagebox.showinfo("Saved", f"Scenario saved to\n{path}")


def main() -> None:
    root = tk.Tk()
    app = ScenarioDesignerApp(root)
    root.minsize(980, 700)
    root.mainloop()


if __name__ == "__main__":
    main()
