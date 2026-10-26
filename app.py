from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="TCP Congestion Control Analyzer",
    page_icon="📡",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_FILE = PROJECT_DIR / "results" / "summary.csv"
EXPERIMENTS_DIR = PROJECT_DIR / "results" / "experiments"

st.title("📡 TCP Congestion-Control Performance Analyzer")
st.caption(
    "NS-3 analysis of TCP Cubic and NewReno under different network conditions"
)

if not RESULTS_FILE.exists():
    st.error(f"Results file not found: {RESULTS_FILE}")
    st.stop()

df = pd.read_csv(RESULTS_FILE)

required = [
    "Experiment",
    "ConfiguredTCP",
    "ConfiguredRate",
    "ConfiguredDelayMs",
    "ConfiguredLoss",
    "ConfiguredBufferPackets",
    "ConfiguredNumFlows",
    "ThroughputMbps",
    "MeanDelayMs",
]

missing = [column for column in required if column not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

for column in [
    "ConfiguredDelayMs",
    "ConfiguredLoss",
    "ConfiguredBufferPackets",
    "ConfiguredNumFlows",
    "ThroughputMbps",
    "MeanDelayMs",
    "TxPackets",
    "RxPackets",
    "LostPackets",
]:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

# One row per experiment, aggregating the individual TCP flow records.
config_columns = [
    "Experiment",
    "ConfiguredTCP",
    "ConfiguredRate",
    "ConfiguredDelayMs",
    "ConfiguredLoss",
    "ConfiguredBufferPackets",
    "ConfiguredNumFlows",
]

aggregations = {
    "ThroughputMbps": "sum",
    "MeanDelayMs": "mean",
}

for column in ["TxPackets", "RxPackets", "LostPackets"]:
    if column in df.columns:
        aggregations[column] = "sum"

experiments = df.groupby(config_columns, dropna=False).agg(aggregations).reset_index()
experiments = experiments.rename(
    columns={
        "ThroughputMbps": "AggregateThroughputMbps",
        "MeanDelayMs": "AverageFlowDelayMs",
    }
)

experiments["FlowRecords"] = df.groupby("Experiment").size().reindex(
    experiments["Experiment"]
).to_numpy()

# Jain's fairness index is calculated from the throughput of flows
# belonging to the same experiment.
def jain_index(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or np.sum(values ** 2) == 0:
        return np.nan
    return float(np.sum(values) ** 2 / (len(values) * np.sum(values ** 2)))

fairness = (
    df.groupby("Experiment")["ThroughputMbps"]
    .apply(jain_index)
    .rename("JainFairnessIndex")
    .reset_index()
)

experiments = experiments.merge(fairness, on="Experiment", how="left")

if {"TxPackets", "RxPackets"}.issubset(experiments.columns):
    experiments["DeliveryRatioPct"] = np.where(
        experiments["TxPackets"] > 0,
        100 * experiments["RxPackets"] / experiments["TxPackets"],
        np.nan,
    )

# ---------------- Sidebar filters ----------------

st.sidebar.header("Experiment Filters")

def multiselect_filter(label, column, data):
    options = sorted(data[column].dropna().unique().tolist())
    return st.sidebar.multiselect(label, options, default=options)

selected_tcp = multiselect_filter("TCP algorithm", "ConfiguredTCP", experiments)
selected_rates = multiselect_filter("Bottleneck bandwidth", "ConfiguredRate", experiments)
selected_delays = multiselect_filter("One-way delay (ms)", "ConfiguredDelayMs", experiments)
selected_losses = multiselect_filter("Packet error probability", "ConfiguredLoss", experiments)
selected_buffers = multiselect_filter(
    "Queue buffer (packets)", "ConfiguredBufferPackets", experiments
)
selected_flows = multiselect_filter(
    "Number of TCP flows", "ConfiguredNumFlows", experiments
)

filtered = experiments[
    experiments["ConfiguredTCP"].isin(selected_tcp)
    & experiments["ConfiguredRate"].isin(selected_rates)
    & experiments["ConfiguredDelayMs"].isin(selected_delays)
    & experiments["ConfiguredLoss"].isin(selected_losses)
    & experiments["ConfiguredBufferPackets"].isin(selected_buffers)
    & experiments["ConfiguredNumFlows"].isin(selected_flows)
].copy()

filtered_flow_rows = df[df["Experiment"].isin(filtered["Experiment"])].copy()

if filtered.empty:
    st.warning("No experiments match these filters. Select more options in the sidebar.")
    st.stop()

# ---------------- Summary metrics ----------------

st.subheader("Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Configurations", f"{filtered['Experiment'].nunique()}")
c2.metric(
    "Mean aggregate throughput",
    f"{filtered['AggregateThroughputMbps'].mean():.3f} Mbps",
)
c3.metric(
    "Mean flow delay",
    f"{filtered['AverageFlowDelayMs'].mean():.2f} ms",
)
c4.metric(
    "Mean Jain fairness",
    f"{filtered['JainFairnessIndex'].mean():.3f}",
)

st.caption(
    "Aggregate throughput is the sum of measured TCP-flow throughputs in each "
    "experiment. Mean flow delay averages the reported delays across its flows. "
    "Fairness describes the relative throughput distribution between flows."
)

# ---------------- Charts ----------------

st.divider()
st.subheader("1. TCP Algorithm Comparison")

left, right = st.columns(2)

with left:
    fig = px.box(
        filtered,
        x="ConfiguredTCP",
        y="AggregateThroughputMbps",
        color="ConfiguredTCP",
        points="all",
        title="Aggregate throughput by TCP algorithm",
        labels={
            "ConfiguredTCP": "TCP algorithm",
            "AggregateThroughputMbps": "Aggregate throughput (Mbps)",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.box(
        filtered,
        x="ConfiguredTCP",
        y="AverageFlowDelayMs",
        color="ConfiguredTCP",
        points="all",
        title="Average flow delay by TCP algorithm",
        labels={
            "ConfiguredTCP": "TCP algorithm",
            "AverageFlowDelayMs": "Average flow delay (ms)",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("2. Effect of Queue Buffer and Number of Flows")

left, right = st.columns(2)

with left:
    fig = px.bar(
        filtered.groupby(
            ["ConfiguredBufferPackets", "ConfiguredTCP"], as_index=False
        )["AggregateThroughputMbps"].mean(),
        x="ConfiguredBufferPackets",
        y="AggregateThroughputMbps",
        color="ConfiguredTCP",
        barmode="group",
        title="Mean aggregate throughput by buffer size",
        labels={
            "ConfiguredBufferPackets": "Buffer size (packets)",
            "AggregateThroughputMbps": "Mean aggregate throughput (Mbps)",
            "ConfiguredTCP": "TCP algorithm",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    flow_means = filtered.groupby(
        ["ConfiguredNumFlows", "ConfiguredTCP"], as_index=False
    ).agg(
        AggregateThroughputMbps=("AggregateThroughputMbps", "mean"),
        AverageFlowDelayMs=("AverageFlowDelayMs", "mean"),
    )
    fig = px.bar(
        flow_means,
        x="ConfiguredNumFlows",
        y="AggregateThroughputMbps",
        color="ConfiguredTCP",
        barmode="group",
        title="Mean aggregate throughput by number of flows",
        labels={
            "ConfiguredNumFlows": "Number of TCP flows",
            "AggregateThroughputMbps": "Mean aggregate throughput (Mbps)",
            "ConfiguredTCP": "TCP algorithm",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("3. Throughput vs Delay")

fig = px.scatter(
    filtered,
    x="AverageFlowDelayMs",
    y="AggregateThroughputMbps",
    color="ConfiguredTCP",
    symbol="ConfiguredNumFlows",
    size="ConfiguredBufferPackets",
    hover_data=[
        "Experiment",
        "ConfiguredRate",
        "ConfiguredDelayMs",
        "ConfiguredLoss",
        "ConfiguredBufferPackets",
        "ConfiguredNumFlows",
    ],
    title="Throughput-delay relationship",
    labels={
        "AverageFlowDelayMs": "Average flow delay (ms)",
        "AggregateThroughputMbps": "Aggregate throughput (Mbps)",
        "ConfiguredTCP": "TCP algorithm",
        "ConfiguredNumFlows": "Number of flows",
        "ConfiguredBufferPackets": "Buffer size (packets)",
    },
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("4. TCP Fairness")

fig = px.bar(
    filtered,
    x="Experiment",
    y="JainFairnessIndex",
    color="ConfiguredTCP",
    hover_data=[
        "ConfiguredNumFlows",
        "ConfiguredBufferPackets",
        "ConfiguredRate",
        "ConfiguredDelayMs",
        "ConfiguredLoss",
    ],
    title="Jain's fairness index for each experiment",
    labels={
        "Experiment": "Experiment",
        "JainFairnessIndex": "Jain fairness index",
        "ConfiguredTCP": "TCP algorithm",
    },
)
fig.update_layout(xaxis={"categoryorder": "category ascending"})
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "For a single-flow experiment, Jain's index is 1 by definition: there is no "
    "competition between flows. Compare fairness mainly between experiments with "
    "two flows. Fairness alone does not indicate high throughput."
)

# ---------------- Packet statistics ----------------

st.subheader("5. Packet Statistics")

packet_columns = [
    column for column in ["TxPackets", "RxPackets", "LostPackets"]
    if column in filtered.columns
]

if packet_columns:
    st.dataframe(
        filtered[
            ["Experiment", "ConfiguredTCP", "ConfiguredRate",
             "ConfiguredDelayMs", "ConfiguredLoss",
             "ConfiguredBufferPackets", "ConfiguredNumFlows"] + packet_columns
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "LostPackets comes from FlowMonitor's loss accounting. It is not a direct "
        "measurement of TCP retransmissions, and it should not be described as such."
    )
else:
    st.info("Packet-count columns are not available in the combined CSV.")

# ---------------- Experiment table and download ----------------

st.subheader("6. Filtered Experiment Results")

st.dataframe(
    filtered.sort_values(
        ["ConfiguredTCP", "ConfiguredRate", "ConfiguredDelayMs",
         "ConfiguredLoss", "ConfiguredBufferPackets", "ConfiguredNumFlows"]
    ),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download filtered experiment summary (CSV)",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="tcp_congestion_filtered_summary.csv",
    mime="text/csv",
)

st.download_button(
    "Download individual TCP flow records (CSV)",
    data=filtered_flow_rows.to_csv(index=False).encode("utf-8"),
    file_name="tcp_congestion_flow_records.csv",
    mime="text/csv",
)

# ---------------- Congestion window ----------------

st.divider()
st.subheader("7. Congestion Window (CWND) Inspection")

available_experiments = sorted(filtered["Experiment"].unique().tolist())
selected_experiment = st.selectbox(
    "Choose an experiment",
    available_experiments,
)

cwnd_file = EXPERIMENTS_DIR / selected_experiment / "cwnd.csv"

if cwnd_file.exists():
    try:
        cwnd = pd.read_csv(cwnd_file)

        if cwnd.empty:
            st.warning("The congestion-window CSV exists but contains no records.")
        else:
            st.write("Congestion-window data columns:", ", ".join(cwnd.columns))

            numeric_columns = cwnd.select_dtypes(include="number").columns.tolist()
            time_candidates = [
                col for col in cwnd.columns
                if "time" in col.lower() or col.lower() in ["t", "seconds", "second"]
            ]
            cwnd_candidates = [
                col for col in numeric_columns
                if "cwnd" in col.lower() or "congestion" in col.lower()
            ]

            if numeric_columns:
                x_column = (
                    time_candidates[0]
                    if time_candidates and time_candidates[0] in numeric_columns
                    else numeric_columns[0]
                )

                possible_y = [col for col in cwnd_candidates if col != x_column]
                if not possible_y:
                    possible_y = [col for col in numeric_columns if col != x_column]

                if possible_y:
                    y_column = st.selectbox(
                        "Congestion-window value column",
                        possible_y,
                    )
                    fig = px.line(
                        cwnd,
                        x=x_column,
                        y=y_column,
                        title=f"Congestion window: {selected_experiment}",
                        labels={
                            x_column: x_column,
                            y_column: y_column,
                        },
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No separate numeric congestion-window column was detected.")
                    st.dataframe(cwnd, use_container_width=True)
            else:
                st.dataframe(cwnd, use_container_width=True)

            st.download_button(
                "Download selected CWND data",
                data=cwnd.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_experiment}_cwnd.csv",
                mime="text/csv",
            )

    except Exception as error:
        st.error(f"Could not read or plot the CWND file: {error}")
else:
    st.warning(f"CWND file not found: {cwnd_file}")

st.divider()
st.caption(
    "Project: TCP Congestion-Control Performance Under Network Conditions | "
    "Simulator: NS-3 | Algorithms: TCP Cubic and TCP NewReno"
)
