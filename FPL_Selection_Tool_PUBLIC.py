import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_URL = "https://fantasy.premierleague.com/api"


@st.cache_data
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


@st.cache_data
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


def position_from_element_type(et: int) -> str:
    mapping = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    return mapping.get(et, "?")


def main():
    st.set_page_config(page_title="FPL Advanced Picks", layout="wide")
    st.title("FPL Advanced Picks – Web")

    players, teams, events = fetch_bootstrap()

    # Sidebar controls
    st.sidebar.header("Filters")

    gw_options = events[events["finished"] == False]["id"].tolist()
    if not gw_options:
        gw_options = events["id"].tolist()
    gw = st.sidebar.selectbox("Gameweek", gw_options)

    pos_filter = st.sidebar.selectbox("Position", ["ALL", "GK", "DEF", "MID", "FWD"])

    preset = st.sidebar.selectbox(
        "Preset",
        ["Template (Balanced)",
         "Aggressive (High Upside)",
         "Safe (High Floor)",
         "Differential (High Risk)"]
    )

    preset_map = {
        "Template (Balanced)": [20, 20, 25, 20, 10, 5],
        "Aggressive (High Upside)": [15, 10, 40, 20, 10, 5],
        "Safe (High Floor)": [25, 25, 15, 20, 10, 5],
        "Differential (High Risk)": [10, 10, 45, 15, 15, 5],
    }
    default_weights = preset_map[preset]

    st.sidebar.markdown("### Custom weights (%)")
    w_form = st.sidebar.slider("Form", 0, 50, default_weights[0])
    w_value = st.sidebar.slider("Value", 0, 50, default_weights[1])
    w_xgi = st.sidebar.slider("xGI", 0, 50, default_weights[2])
    w_ep = st.sidebar.slider("EP Next", 0, 50, default_weights[3])
    w_fix = st.sidebar.slider("Fixture", 0, 50, default_weights[4])
    w_min = st.sidebar.slider("Minutes", 0, 50, default_weights[5])

    total = w_form + w_value + w_xgi + w_ep + w_fix + w_min
    if total == 0:
        weights = [1/6] * 6
    else:
        weights = [w_form/total, w_value/total, w_xgi/total,
                   w_ep/total, w_fix/total, w_min/total]

    fixtures = fetch_fixtures_for_gw(int(gw))
    df = compute_attractiveness(players, fixtures, weights)

    team_lookup = dict(zip(teams["id"], teams["short_name"]))
    df["team_name"] = df["team"].map(team_lookup)
    df["pos"] = df["element_type"].apply(position_from_element_type)

    if pos_filter != "ALL":
        df = df[df["pos"] == pos_filter]

    df = df.copy()
    df_display = df.sort_values("attractiveness", ascending=False).head(30)

    st.subheader(f"Top picks – GW {gw}")
    st.caption("Score = weighted blend of Form, Value, xGI, EP Next, Fixture, Minutes. Model EP is a predictive expected-points estimate.")

    cols = ["web_name", "team_name", "pos", "price_m", "form",
            "expected_goal_involvements", "ep_next", "value_ppm",
            "attractiveness", "model_ep"]
    df_display = df_display[cols].rename(columns={
        "web_name": "Name",
        "team_name": "Team",
        "pos": "Pos",
        "price_m": "Price",
        "form": "Form",
        "expected_goal_involvements": "xGI",
        "ep_next": "EP Next",
        "value_ppm": "Value",
        "attractiveness": "Score",
        "model_ep": "Model EP"
    })

    st.dataframe(
        df_display.style.format({
            "Price": "{:.1f}",
            "Form": "{:.2f}",
            "xGI": "{:.2f}",
            "EP Next": "{:.2f}",
            "Value": "{:.2f}",
            "Score": "{:.3f}",
            "Model EP": "{:.2f}",
        }),
        use_container_width=True,
        height=500
    )

    # Bar chart
    st.subheader("Score bar chart")
    fig, ax = plt.subplots(figsize=(10, 4))
    top = df_display.head(20)
    ax.bar(range(len(top)), top["Score"], color="royalblue")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top["Name"], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Top Score – GW {gw}")
    st.pyplot(fig)

    # Simple "why this player" inspector
    st.subheader("Player breakdown")
    selected_name = st.selectbox("Select player", top["Name"].tolist())
    row = df[df["web_name"] == selected_name].iloc[0]

    st.markdown(f"**{row['web_name']} ({row['team_name']} – {row['pos']})**")
    st.write(f"Score: {row['attractiveness']:.3f}")
    st.write(f"Model EP: {row['model_ep']:.2f}")

    st.markdown("**Raw metrics**")
    st.write(
        f"- Form: {row['form']:.2f}\n"
        f"- Value PPM: {row['value_ppm']:.2f}\n"
        f"- xGI: {row['expected_goal_involvements']:.2f}\n"
        f"- EP Next: {row['ep_next']:.2f}\n"
        f"- Fixture Ease: {row['fixture_ease']:.2f}\n"
        f"- Minutes: {row['minutes_security']:.0f}"
    )

    st.markdown("**Model EP explanation**")
    st.write(
        "Model EP ≈ 10 × (0.4·xGIₙ + 0.3·Formₙ + 0.2·Fixtureₙ + 0.1·Minutesₙ), "
        "where each component is normalized 0–1."
    )


if __name__ == "__main__":
    main()


