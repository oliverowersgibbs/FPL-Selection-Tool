import tkinter as tk
from tkinter import ttk, messagebox
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

BASE_URL = "https://fantasy.premierleague.com/api"


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None

    def show_tip(self, x, y):
        if self.tipwindow:
            return
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify="left",
            background="#ffffe0", relief="solid", borderwidth=1,
            font=("Segoe UI", 9)
        )
        label.pack(ipadx=5, ipady=3)

    def hide_tip(self):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


def fetch_bootstrap():
    r = requests.get(f"{BASE_URL}/bootstrap-static/")
    data = r.json()
    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])
    events = pd.DataFrame(data["events"])

    numeric_fields = [
        "form", "points_per_game", "ep_next",
        "expected_goal_involvements", "expected_goals",
        "expected_assists", "now_cost", "minutes"
    ]
    for col in numeric_fields:
        if col in players.columns:
            players[col] = pd.to_numeric(players[col], errors="coerce")

    return players, teams, events


def fetch_fixtures_for_gw(gw: int) -> pd.DataFrame:
    r = requests.get(f"{BASE_URL}/fixtures/?event={gw}")
    return pd.DataFrame(r.json())


def fixture_difficulty_for_player(fixtures_df: pd.DataFrame, player_row: pd.Series) -> float:
    team_id = player_row["team"]
    fx = fixtures_df[
        (fixtures_df["team_h"] == team_id) | (fixtures_df["team_a"] == team_id)
    ]

    if fx.empty:
        return 0.5

    row = fx.iloc[0]
    if row["team_h"] == team_id:
        diff = row["team_h_difficulty"]
        home = 1
    else:
        diff = row["team_a_difficulty"]
        home = 0

    ease = (6 - diff) / 5.0
    ease += 0.1 * home
    return max(0.0, min(1.2, ease))


def normalize(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    if s.max() == s.min():
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def compute_attractiveness(players: pd.DataFrame,
                           fixtures_df: pd.DataFrame,
                           weights: list[float]) -> pd.DataFrame:
    df = players.copy()

    df["price_m"] = df["now_cost"] / 10.0
    df["value_ppm"] = df["points_per_game"] / df["price_m"].replace(0, 1)
    df["fixture_ease"] = df.apply(
        lambda row: fixture_difficulty_for_player(fixtures_df, row), axis=1
    )
    df["minutes_security"] = df["minutes"]

    form_n = normalize(df["form"])
    value_n = normalize(df["value_ppm"])
    xgi_n = normalize(df["expected_goal_involvements"])
    ep_n = normalize(df["ep_next"])
    fix_n = normalize(df["fixture_ease"])
    min_n = normalize(df["minutes_security"])

    w_form, w_value, w_xgi, w_ep, w_fix, w_min = weights

    df["attractiveness"] = (
        w_form * form_n +
        w_value * value_n +
        w_xgi * xgi_n +
        w_ep * ep_n +
        w_fix * fix_n +
        w_min * min_n
    )

    df["model_ep"] = (
        0.4 * xgi_n +
        0.3 * form_n +
        0.2 * fix_n +
        0.1 * min_n
    ) * 10

    return df


def radar_chart(players_df: pd.DataFrame, rows: list[int], title: str = "Player comparison") -> None:
    if len(rows) < 1:
        return

    metrics = ["form", "value_ppm", "expected_goal_involvements",
               "ep_next", "fixture_ease", "minutes_security"]
    labels = ["Form", "Value", "xGI", "EP Next", "Fixture", "Minutes"]

    data = []
    names = []
    for r in rows:
        data.append([
            players_df.loc[r, "form"],
            players_df.loc[r, "value_ppm"],
            players_df.loc[r, "expected_goal_involvements"],
            players_df.loc[r, "ep_next"],
            players_df.loc[r, "fixture_ease"],
            players_df.loc[r, "minutes_security"],
        ])
        names.append(players_df.loc[r, "web_name"])

    data = np.array(data, dtype=float)
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-9)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    data_norm = np.concatenate((data_norm, data_norm[:, :1]), axis=1)
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    for i, row in enumerate(data_norm):
        ax.plot(angles, row, label=names[i])
        ax.fill(angles, row, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.show()


class FPLApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("FPL Advanced Picks")
        self.root.geometry("1250x780")

        self.players, self.teams, self.events = fetch_bootstrap()
        self.current_df: pd.DataFrame | None = None

        top_frame = tk.Frame(root)
        top_frame.pack(pady=5, fill="x")

        tk.Label(top_frame, text="Gameweek:").pack(side=tk.LEFT, padx=5)

        self.gw_var = tk.StringVar()
        gw_options = self.events[self.events["finished"] == False]["id"].tolist()
        if not gw_options:
            gw_options = self.events["id"].tolist()
        self.gw_var.set(gw_options[0])

        self.gw_menu = ttk.Combobox(
            top_frame, textvariable=self.gw_var, values=gw_options, width=5, state="readonly"
        )
        self.gw_menu.pack(side=tk.LEFT)

        self.btn = tk.Button(
            top_frame, text="Calculate Top Picks", command=self.update_view
        )
        self.btn.pack(side=tk.LEFT, padx=10)

        tk.Label(top_frame, text="Position:").pack(side=tk.LEFT, padx=5)
        self.pos_var = tk.StringVar()
        self.pos_var.set("ALL")
        self.pos_menu = ttk.Combobox(
            top_frame,
            textvariable=self.pos_var,
            values=["ALL", "GK", "DEF", "MID", "FWD"],
            width=5,
            state="readonly",
        )
        self.pos_menu.pack(side=tk.LEFT)

        tk.Button(
            top_frame, text="Compare Selected (Radar)",
            command=self.compare_selected
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            top_frame, text="Transfer Advisor",
            command=self.open_transfer_advisor
        ).pack(side=tk.LEFT, padx=10)

        preset_frame = tk.Frame(root)
        preset_frame.pack(pady=5)

        tk.Label(preset_frame, text="Preset:").pack(side=tk.LEFT, padx=5)

        self.preset_var = tk.StringVar()
        self.preset_var.set("Template (Balanced)")

        preset_options = [
            "Template (Balanced)",
            "Aggressive (High Upside)",
            "Safe (High Floor)",
            "Differential (High Risk)",
        ]

        self.preset_menu = ttk.Combobox(
            preset_frame, textvariable=self.preset_var,
            values=preset_options, width=25, state="readonly"
        )
        self.preset_menu.pack(side=tk.LEFT)
        self.preset_menu.bind("<<ComboboxSelected>>", self.apply_preset)

        weights_frame = tk.LabelFrame(root, text="Score Weights (%)")
        weights_frame.pack(fill="x", padx=10, pady=5)

        self.weight_vars: dict[str, tk.DoubleVar] = {}
        components = [
            ("Form", "form", 20),
            ("Value", "value", 20),
            ("xGI", "xgi", 25),
            ("EP Next", "ep", 20),
            ("Fixture", "fix", 10),
            ("Minutes", "min", 5),
        ]
        for label, key, default in components:
            var = tk.DoubleVar(value=default)
            self.weight_vars[key] = var
            frame = tk.Frame(weights_frame)
            frame.pack(side=tk.LEFT, padx=5)
            tk.Label(frame, text=label).pack()
            tk.Scale(
                frame, from_=0, to=50, orient=tk.HORIZONTAL,
                variable=var, length=120
            ).pack()

        columns = ("Name", "Team", "Pos", "Price", "Form", "xGI",
                   "EP Next", "Value", "Score", "Model EP")
        self.table = ttk.Treeview(root, columns=columns, show="headings", height=15)
        for col in columns:
            self.table.heading(col, text=col)
            self.table.column(col, width=100, anchor="center")
        self.table.pack(fill="x", padx=10, pady=10)

        self.table.bind("<Double-1>", self.show_player_breakdown)

        self.col_tooltips: dict[str, str] = {
            "#1": "Player name",
            "#2": "Team short name",
            "#3": "Position (GK/DEF/MID/FWD)",
            "#4": "Price in millions",
            "#5": "FPL form (recent performance)",
            "#6": "Expected goal involvements (xG + xA)",
            "#7": "FPL expected points for next GW",
            "#8": "Points per game per million (value)",
            "#9": "Attractiveness Score (weighted composite)",
            "#10": "Model EP: predictive expected points using xGI, form, fixture, minutes",
        }
        self.active_tooltip: ToolTip | None = None
        self.table.bind("<Motion>", self.on_table_motion)

        self.fig, self.ax = plt.subplots(figsize=(9, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def apply_preset(self, event=None) -> None:
        preset = self.preset_var.get()

        presets: dict[str, list[int]] = {
            "Template (Balanced)": [20, 20, 25, 20, 10, 5],
            "Aggressive (High Upside)": [15, 10, 40, 20, 10, 5],
            "Safe (High Floor)": [25, 25, 15, 20, 10, 5],
            "Differential (High Risk)": [10, 10, 45, 15, 15, 5],
        }

        values = presets[preset]

        for (key, var), val in zip(self.weight_vars.items(), values):
            var.set(val)

        self.update_view()

    def position_from_element_type(self, et: int) -> str:
        mapping = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        return mapping.get(et, "?")

    def get_weights(self) -> list[float]:
        w_form = self.weight_vars["form"].get()
        w_value = self.weight_vars["value"].get()
        w_xgi = self.weight_vars["xgi"].get()
        w_ep = self.weight_vars["ep"].get()
        w_fix = self.weight_vars["fix"].get()
        w_min = self.weight_vars["min"].get()
        total = w_form + w_value + w_xgi + w_ep + w_fix + w_min
        if total == 0:
            return [1 / 6] * 6
        return [w_form / total, w_value / total, w_xgi / total,
                w_ep / total, w_fix / total, w_min / total]

    def update_view(self) -> None:
        gw = int(self.gw_var.get())
        fixtures = fetch_fixtures_for_gw(gw)
        weights = self.get_weights()
        df = compute_attractiveness(self.players, fixtures, weights)

        team_lookup = dict(zip(self.teams["id"], self.teams["short_name"]))
        df["team_name"] = df["team"].map(team_lookup)
        df["pos"] = df["element_type"].apply(self.position_from_element_type)

        pos_filter = self.pos_var.get()
        if pos_filter != "ALL":
            df = df[df["pos"] == pos_filter]

        df = df.copy()
        self.current_df = df

        top = df.sort_values("attractiveness", ascending=False).head(20)

        for row in self.table.get_children():
            self.table.delete(row)

        for idx, r in top.iterrows():
            self.table.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(
                    r["web_name"],
                    r["team_name"],
                    r["pos"],
                    f"{r['price_m']:.1f}",
                    f"{r['form']:.2f}",
                    f"{r['expected_goal_involvements']:.2f}",
                    f"{r['ep_next']:.2f}",
                    f"{r['value_ppm']:.2f}",
                    f"{r['attractiveness']:.3f}",
                    f"{r['model_ep']:.2f}",
                ),
            )

        self.ax.clear()
        x_positions = range(len(top))
        self.ax.bar(x_positions, top["attractiveness"], color="royalblue")
        self.ax.set_xticks(x_positions)
        self.ax.set_xticklabels(top["web_name"], rotation=45, ha="right")
        self.ax.set_title(f"Top Picks – GW {gw}")
        self.ax.set_ylabel("Attractiveness score")
        self.fig.tight_layout()
        self.canvas.draw()

    def on_table_motion(self, event) -> None:
        region = self.table.identify_region(event.x, event.y)
        column = self.table.identify_column(event.x)

        if region == "heading" and column in self.col_tooltips:
            x = self.table.winfo_rootx() + event.x + 10
            y = self.table.winfo_rooty() + event.y + 20
            text = self.col_tooltips[column]
            if self.active_tooltip is None:
                self.active_tooltip = ToolTip(self.table, text)
                self.active_tooltip.show_tip(x, y)
            else:
                self.active_tooltip.hide_tip()
                self.active_tooltip = ToolTip(self.table, text)
                self.active_tooltip.show_tip(x, y)
        else:
            if self.active_tooltip is not None:
                self.active_tooltip.hide_tip()
                self.active_tooltip = None

    def show_player_breakdown(self, event) -> None:
        item_id = self.table.focus()
        if not item_id or self.current_df is None:
            return

        idx = int(item_id)
        r = self.current_df.loc[idx]

        weights = self.get_weights()
        w_form, w_value, w_xgi, w_ep, w_fix, w_min = weights

        form_n = normalize(self.current_df["form"])[idx]
        value_n = normalize(self.current_df["value_ppm"])[idx]
        xgi_n = normalize(self.current_df["expected_goal_involvements"])[idx]
        ep_n = normalize(self.current_df["ep_next"])[idx]
        fix_n = normalize(self.current_df["fixture_ease"])[idx]
        min_n = normalize(self.current_df["minutes_security"])[idx]

        contribs = {
            "Form": w_form * form_n,
            "Value": w_value * value_n,
            "xGI": w_xgi * xgi_n,
            "EP Next": w_ep * ep_n,
            "Fixture": w_fix * fix_n,
            "Minutes": w_min * min_n,
        }

        win = tk.Toplevel(self.root)
        win.title(f"Why {r['web_name']}?")
        win.geometry("420x380")

        tk.Label(win, text=f"{r['web_name']} ({r['team_name']} - {r['pos']})",
                 font=("Segoe UI", 11, "bold")).pack(pady=5)

        tk.Label(win, text=f"Score: {r['attractiveness']:.3f}").pack()
        tk.Label(win, text=f"Model EP: {r['model_ep']:.2f}").pack(pady=5)

        tk.Label(win, text="Component contributions:",
                 font=("Segoe UI", 10, "bold")).pack(pady=5)
        frame = tk.Frame(win)
        frame.pack(pady=5)

        for k, v in contribs.items():
            tk.Label(frame, text=f"{k}: {v:.3f}").pack(anchor="w")

        tk.Label(win, text="Raw metrics:", font=("Segoe UI", 10, "bold")).pack(pady=5)
        tk.Label(
            win,
            text=(
                f"Form: {r['form']:.2f}\n"
                f"Value PPM: {r['value_ppm']:.2f}\n"
                f"xGI: {r['expected_goal_involvements']:.2f}\n"
                f"EP Next: {r['ep_next']:.2f}\n"
                f"Fixture Ease: {r['fixture_ease']:.2f}\n"
                f"Minutes: {r['minutes_security']:.0f}\n\n"
                f"Model EP explanation:\n"
                f"Model EP = 10 * (0.4*xGI_n + 0.3*Form_n + 0.2*Fixture_n + 0.1*Minutes_n)"
            ),
            justify="left"
        ).pack(pady=5)

    def compare_selected(self) -> None:
        if self.current_df is None:
            messagebox.showinfo("Info", "Run 'Calculate Top Picks' first.")
            return

        selected = self.table.selection()
        if len(selected) < 1:
            messagebox.showinfo("Info", "Select at least one player in the table.")
            return

        idxs = [int(iid) for iid in selected]
        radar_chart(self.current_df, idxs, title="Player comparison")

    def open_transfer_advisor(self) -> None:
        if self.current_df is None:
            messagebox.showinfo("Info", "Run 'Calculate Top Picks' first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Transfer Advisor")
        win.geometry("600x500")

        tk.Label(win, text="Max Price (millions):").pack(pady=5)
        price_var = tk.DoubleVar(value=8.0)
        tk.Entry(win, textvariable=price_var).pack()

        tk.Label(win, text="Per-position top picks:").pack(pady=5)

        text = tk.Text(win, wrap="word")
        text.pack(fill="both", expand=True, padx=10, pady=10)

        def run_advisor() -> None:
            max_price = price_var.get()
            df = self.current_df[self.current_df["price_m"] <= max_price].copy()
            if df.empty:
                text.delete("1.0", tk.END)
                text.insert(tk.END, "No players under that price.\n")
                return

            out_lines: list[str] = []
            for pos in ["GK", "DEF", "MID", "FWD"]:
                sub = df[df["pos"] == pos].sort_values(
                    "attractiveness", ascending=False
                ).head(5)
                if sub.empty:
                    continue
                out_lines.append(f"=== {pos} ===")
                for _, r in sub.iterrows():
                    out_lines.append(
                        f"{r['web_name']} ({r['team_name']}) "
                        f"- {r['price_m']:.1f}m | Score {r['attractiveness']:.3f} | "
                        f"Model EP {r['model_ep']:.2f}"
                    )
                out_lines.append("")

            text.delete("1.0", tk.END)
            text.insert(tk.END, "\n".join(out_lines))

        tk.Button(win, text="Run Advisor", command=run_advisor).pack(pady=5)


def main() -> None:
    root = tk.Tk()
    app = FPLApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()