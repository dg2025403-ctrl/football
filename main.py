import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ---------------------------------
# 페이지 설정
# ---------------------------------
st.set_page_config(
    page_title="선수 유형 나누기",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ 선수 유형 나누기")
st.caption("축구 게임 선수들의 능력치를 바탕으로 K-평균 군집화를 수행합니다.")


# ---------------------------------
# 데이터 불러오기
# ---------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/main/data/"
    "eafc25_top100.csv"
)

try:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
except Exception as error:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {error}")
    st.stop()


required_columns = [
    "name_ko",
    "name",
    "positions",
    "club",
    "overall",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
    "value_eur",
    "age",
    "height_cm",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "다음 필수 열이 데이터에 없습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()


# ---------------------------------
# 능력치 이름
# ---------------------------------
ability_columns = {
    "pace": "속도",
    "shooting": "슈팅",
    "passing": "패스",
    "dribbling": "드리블",
    "defending": "수비",
    "physic": "몸싸움",
}

ability_names = list(ability_columns.values())

display_to_data_column = {
    display_name: data_column
    for data_column, display_name in ability_columns.items()
}


# ---------------------------------
# 숫자형 변환
# ---------------------------------
number_columns = [
    "overall",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
    "value_eur",
    "age",
    "height_cm",
]

for column in number_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


analysis_df = df.dropna(
    subset=list(ability_columns.keys()) + ["overall", "name_ko"]
).copy()

if len(analysis_df) < 7:
    st.error("분석할 선수가 7명보다 적습니다.")
    st.stop()


# ---------------------------------
# 분석 설정
# ---------------------------------
st.subheader("1. 분석 설정")

selected_abilities = st.multiselect(
    "묶는 데 사용할 능력치를 두 개 이상 선택하세요.",
    options=ability_names,
    default=ability_names,
)

if len(selected_abilities) < 2:
    st.warning("능력치를 두 개 이상 선택해야 합니다.")
    st.stop()


selected_cluster_count = st.slider(
    "묶음 수",
    min_value=2,
    max_value=7,
    value=3,
    step=1,
)


# ---------------------------------
# 선택한 능력치 표준화
# ---------------------------------
selected_data_columns = [
    display_to_data_column[ability]
    for ability in selected_abilities
]

scaler = StandardScaler()

scaled_values = scaler.fit_transform(
    analysis_df[selected_data_columns]
)


# ---------------------------------
# 묶음 기호
# ---------------------------------
cluster_symbols = [
    "㉮",
    "㉯",
    "㉰",
    "㉱",
    "㉲",
    "㉳",
    "㉴",
]


# ---------------------------------
# 현재 선택한 묶음 수로 군집화
# ---------------------------------
selected_kmeans = KMeans(
    n_clusters=selected_cluster_count,
    random_state=42,
    n_init=10,
)

selected_cluster_numbers = selected_kmeans.fit_predict(
    scaled_values
)

analysis_df["cluster_raw"] = selected_cluster_numbers


# 슈팅 평균이 높은 묶음부터 기호 부여
shooting_means = (
    analysis_df
    .groupby("cluster_raw")["shooting"]
    .mean()
    .sort_values(ascending=False)
)

cluster_order = shooting_means.index.tolist()

cluster_label_map = {
    raw_cluster: cluster_symbols[index]
    for index, raw_cluster in enumerate(cluster_order)
}

analysis_df["묶음"] = analysis_df["cluster_raw"].map(
    cluster_label_map
)

ordered_labels = cluster_symbols[:selected_cluster_count]


# ---------------------------------
# 기본 정보
# ---------------------------------
st.subheader("2. 분석 결과")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.metric("전체 선수 수", f"{len(df):,}명")

with info_col2:
    st.metric("분석 선수 수", f"{len(analysis_df):,}명")

with info_col3:
    st.metric("현재 묶음 수", f"{selected_cluster_count}개")


# ---------------------------------
# 2차원 산점도
# ---------------------------------
st.subheader("3. 2차원 산점도")

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_axis = st.selectbox(
        "가로축 능력치",
        options=selected_abilities,
        index=0,
        key="x_axis_2d",
    )

with axis_col2:
    y_axis = st.selectbox(
        "세로축 능력치",
        options=selected_abilities,
        index=1,
        key="y_axis_2d",
    )

x_data_column = display_to_data_column[x_axis]
y_data_column = display_to_data_column[y_axis]

fig_2d = px.scatter(
    analysis_df,
    x=x_data_column,
    y=y_data_column,
    color="묶음",
    category_orders={"묶음": ordered_labels},
    hover_name="name_ko",
    hover_data={
        "name_ko": True,
        "묶음": True,
        x_data_column: True,
        y_data_column: True,
    },
    labels={
        x_data_column: x_axis,
        y_data_column: y_axis,
        "묶음": "선수 유형",
        "name_ko": "한글 이름",
    },
    title=f"{x_axis}와 {y_axis}에 따른 선수 유형",
)

fig_2d.update_traces(marker={"size": 9})
fig_2d.update_layout(legend_title_text="선수 유형")

st.plotly_chart(fig_2d, use_container_width=True)


# ---------------------------------
# 3차원 산점도
# ---------------------------------
st.subheader("4. 3차원 산점도")

if len(selected_abilities) < 3:
    st.info(
        "3차원 산점도를 표시하려면 능력치를 세 개 이상 선택해야 합니다."
    )
else:
    axis3d_col1, axis3d_col2, axis3d_col3 = st.columns(3)

    with axis3d_col1:
        x_axis_3d = st.selectbox(
            "3차원 x축 능력치",
            options=selected_abilities,
            index=0,
            key="x_axis_3d",
        )

    with axis3d_col2:
        y_axis_3d = st.selectbox(
            "3차원 y축 능력치",
            options=selected_abilities,
            index=1,
            key="y_axis_3d",
        )

    with axis3d_col3:
        z_axis_3d = st.selectbox(
            "3차원 z축 능력치",
            options=selected_abilities,
            index=2,
            key="z_axis_3d",
        )

    x_data_column_3d = display_to_data_column[x_axis_3d]
    y_data_column_3d = display_to_data_column[y_axis_3d]
    z_data_column_3d = display_to_data_column[z_axis_3d]

    fig_3d = px.scatter_3d(
        analysis_df,
        x=x_data_column_3d,
        y=y_data_column_3d,
        z=z_data_column_3d,
        color="묶음",
        category_orders={"묶음": ordered_labels},
        hover_name="name_ko",
        hover_data={
            "name_ko": True,
            "묶음": True,
            x_data_column_3d: True,
            y_data_column_3d: True,
            z_data_column_3d: True,
        },
        labels={
            x_data_column_3d: x_axis_3d,
            y_data_column_3d: y_axis_3d,
            z_data_column_3d: z_axis_3d,
            "묶음": "선수 유형",
            "name_ko": "한글 이름",
        },
        title=(
            f"{x_axis_3d}, {y_axis_3d}, {z_axis_3d}"
            "에 따른 선수 유형"
        ),
    )

    fig_3d.update_traces(
        marker={
            "size": 3,
            "opacity": 0.85,
        }
    )

    fig_3d.update_layout(
        legend_title_text="선수 유형",
        scene={
            "xaxis_title": x_axis_3d,
            "yaxis_title": y_axis_3d,
            "zaxis_title": z_axis_3d,
        },
    )

    st.plotly_chart(fig_3d, use_container_width=True)


# ---------------------------------
# 묶음별 평균 표
# ---------------------------------
st.subheader("5. 묶음별 능력치 평균")

summary_rows = []

for label in ordered_labels:
    group = analysis_df[analysis_df["묶음"] == label]

    row = {
        "묶음": label,
        "인원": len(group),
    }

    for data_column, display_name in ability_columns.items():
        row[f"{display_name} 평균"] = group[data_column].mean()

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

for display_name in ability_names:
    summary_df[f"{display_name} 평균"] = (
        summary_df[f"{display_name} 평균"].round(1)
    )

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------
# 묶음별 종합 능력치 상위 선수
# ---------------------------------
st.subheader("6. 묶음별 종합 능력치가 높은 선수 5명")

top_player_rows = []

for label in ordered_labels:
    group = (
        analysis_df[analysis_df["묶음"] == label]
        .sort_values("overall", ascending=False)
        .head(5)
    )

    names = group["name_ko"].tolist()

    row = {"묶음": label}

    for rank in range(5):
        row[f"{rank + 1}위"] = (
            names[rank] if rank < len(names) else ""
        )

    top_player_rows.append(row)

top_players_df = pd.DataFrame(top_player_rows)

st.dataframe(
    top_players_df,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------
# 포지션 분류
# ---------------------------------
def classify_position(position_text):
    """positions 열의 맨 앞 포지션을 세 가지로 분류합니다."""

    if pd.isna(position_text):
        return "수비수"

    first_position = str(position_text).split(",")[0].strip()
    first_position = first_position.split("/")[0].strip()
    first_position = first_position.upper()

    attacker_positions = {
        "ST", "CF", "LW", "RW", "LF", "RF"
    }

    midfielder_positions = {
        "CAM", "CM", "CDM", "LM", "RM",
        "LAM", "RAM", "LDM", "RDM"
    }

    defender_positions = {
        "CB", "LB", "RB", "LWB", "RWB", "SW", "GK"
    }

    if first_position in attacker_positions:
        return "공격수"

    if first_position in midfielder_positions:
        return "미드필더"

    if first_position in defender_positions:
        return "수비수"

    return "수비수"


analysis_df["포지션 분류"] = analysis_df["positions"].apply(
    classify_position
)


# ---------------------------------
# 묶음과 포지션 교차표
# ---------------------------------
st.subheader("7. 묶음과 포지션의 교차표")

position_order = [
    "공격수",
    "미드필더",
    "수비수",
]

crosstab_df = pd.crosstab(
    analysis_df["묶음"],
    analysis_df["포지션 분류"],
)

crosstab_df = crosstab_df.reindex(
    index=ordered_labels,
    columns=position_order,
    fill_value=0,
)

crosstab_df["합계"] = crosstab_df.sum(axis=1)

total_row = crosstab_df.sum(axis=0)
total_row.name = "합계"

crosstab_df = pd.concat(
    [
        crosstab_df,
        total_row.to_frame().T,
    ]
)

st.dataframe(
    crosstab_df,
    use_container_width=True,
)


# ---------------------------------
# 묶음 수별 군집 내 제곱거리 합
# ---------------------------------
st.subheader("8. 묶음 수에 따른 군집 내 제곱거리 합")

inertia_rows = []

for cluster_count in range(1, 8):
    inertia_kmeans = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=10,
    )

    inertia_kmeans.fit(scaled_values)

    inertia_rows.append(
        {
            "묶음 수": cluster_count,
            "군집 내 제곱거리 합": inertia_kmeans.inertia_,
        }
    )

inertia_df = pd.DataFrame(inertia_rows)

fig_inertia = go.Figure()

fig_inertia.add_trace(
    go.Scatter(
        x=inertia_df["묶음 수"],
        y=inertia_df["군집 내 제곱거리 합"],
        mode="lines+markers",
        name="군집 내 제곱거리 합",
        hovertemplate=(
            "묶음 수: %{x}<br>"
            "군집 내 제곱거리 합: %{y:.2f}"
            "<extra></extra>"
        ),
    )
)

fig_inertia.add_vline(
    x=selected_cluster_count,
    line_width=2,
    line_dash="dash",
    line_color="red",
    annotation_text=f"현재 선택: {selected_cluster_count}개",
    annotation_position="top right",
)

fig_inertia.update_layout(
    title="묶음 수별 군집 내 제곱거리 합",
    xaxis_title="묶음 수",
    yaxis_title="군집 내 제곱거리 합",
    xaxis={
        "tickmode": "linear",
        "dtick": 1,
    },
)

st.plotly_chart(fig_inertia, use_container_width=True)


# ---------------------------------
# 제곱거리 합과 감소량 표
# ---------------------------------
inertia_table_df = inertia_df.copy()

inertia_table_df["바로 앞 값에서 감소한 값"] = (
    inertia_table_df["군집 내 제곱거리 합"].shift(1)
    - inertia_table_df["군집 내 제곱거리 합"]
)

# 첫 줄은 비교할 앞 값이 없으므로 빈칸
inertia_table_df.loc[
    inertia_table_df.index[0],
    "바로 앞 값에서 감소한 값",
] = None

inertia_table_df["군집 내 제곱거리 합"] = (
    inertia_table_df["군집 내 제곱거리 합"].round(4)
)

inertia_table_df["바로 앞 값에서 감소한 값"] = (
    inertia_table_df["바로 앞 값에서 감소한 값"].round(4)
)

st.dataframe(
    inertia_table_df,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------
# 실루엣 점수 계산
# ---------------------------------
st.subheader("9. 묶음 수별 실루엣 점수")

silhouette_rows = []

for cluster_count in range(2, 8):
    silhouette_kmeans = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=10,
    )

    silhouette_labels = silhouette_kmeans.fit_predict(
        scaled_values
    )

    score = silhouette_score(
        scaled_values,
        silhouette_labels,
    )

    silhouette_rows.append(
        {
            "묶음 수": cluster_count,
            "실루엣 점수": score,
        }
    )

silhouette_df = pd.DataFrame(silhouette_rows)

fig_silhouette = go.Figure()

fig_silhouette.add_trace(
    go.Scatter(
        x=silhouette_df["묶음 수"],
        y=silhouette_df["실루엣 점수"],
        mode="lines+markers",
        name="실루엣 점수",
        hovertemplate=(
            "묶음 수: %{x}<br>"
            "실루엣 점수: %{y:.4f}"
            "<extra></extra>"
        ),
    )
)

fig_silhouette.add_vline(
    x=selected_cluster_count,
    line_width=2,
    line_dash="dash",
    line_color="red",
    annotation_text=f"현재 선택: {selected_cluster_count}개",
    annotation_position="top right",
)

fig_silhouette.update_layout(
    title="묶음 수별 실루엣 점수",
    xaxis_title="묶음 수",
    yaxis_title="실루엣 점수",
    yaxis={
        "range": [-1, 1],
    },
    xaxis={
        "tickmode": "linear",
        "dtick": 1,
    },
)

st.plotly_chart(fig_silhouette, use_container_width=True)


# ---------------------------------
# 실루엣 점수 표
# ---------------------------------
silhouette_table_df = silhouette_df.copy()

silhouette_table_df["실루엣 점수"] = (
    silhouette_table_df["실루엣 점수"].round(4)
)

st.dataframe(
    silhouette_table_df,
    use_container_width=True,
    hide_index=True,
)
