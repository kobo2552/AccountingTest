import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# ページ設定

st.set_page_config(
page_title=“Streamlit サンプルアプリ”,
page_icon=“📊”,
layout=“wide”,
initial_sidebar_state=“expanded”
)

# メインタイトル

st.title(“📊 Streamlit サンプルアプリケーション”)
st.markdown(”—”)

# サイドバー

st.sidebar.header(“設定”)
selected_demo = st.sidebar.selectbox(
“デモを選択してください”,
[“データ分析”, “機械学習予測”, “リアルタイムチャート”, “画像処理”]
)

# データ分析デモ

if selected_demo == “データ分析”:
st.header(“📈 データ分析デモ”)

```
# サンプルデータ生成
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
    data = {
        "date": dates,
        "sales": np.random.normal(1000, 200, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 100,
        "visitors": np.random.poisson(150, len(dates)),
        "conversion_rate": np.random.uniform(0.02, 0.08, len(dates)),
        "category": np.random.choice(["A", "B", "C"], len(dates))
    }
    df = pd.DataFrame(data)
    df["sales"] = np.maximum(df["sales"], 0)  # 負の値を0に
    return df

df = generate_sample_data()

# フィルター
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "開始日",
        value=datetime(2024, 1, 1),
        min_value=df["date"].min().date(),
        max_value=df["date"].max().date()
    )
with col2:
    end_date = st.date_input(
        "終了日",
        value=datetime(2024, 12, 31),
        min_value=df["date"].min().date(),
        max_value=df["date"].max().date()
    )

# データフィルタリング
filtered_df = df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))]

# メトリクス表示
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("総売上", f"¥{filtered_df['sales'].sum():,.0f}")
with col2:
    st.metric("総訪問者数", f"{filtered_df['visitors'].sum():,}")
with col3:
    avg_conversion = filtered_df["conversion_rate"].mean()
    st.metric("平均コンバージョン率", f"{avg_conversion:.2%}")
with col4:
    avg_daily_sales = filtered_df["sales"].mean()
    st.metric("平均日次売上", f"¥{avg_daily_sales:,.0f}")

# グラフ表示
st.subheader("売上推移")
fig = px.line(filtered_df, x="date", y="sales", title="日次売上推移")
st.plotly_chart(fig, use_container_width=True)

# 散布図
st.subheader("訪問者数と売上の関係")
fig2 = px.scatter(filtered_df, x="visitors", y="sales", color="category",
                 title="訪問者数 vs 売上", trendline="ols")
st.plotly_chart(fig2, use_container_width=True)
```

# 機械学習デモ

elif selected_demo == “機械学習予測”:
st.header(“🤖 機械学習予測デモ”)

```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# サンプルデータ生成
@st.cache_data
def generate_ml_data():
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 4)
    y = X[:, 0] * 2 + X[:, 1] * -1.5 + X[:, 2] * 0.5 + X[:, 3] * 3 + np.random.randn(n_samples) * 0.1
    
    feature_names = ["feature_1", "feature_2", "feature_3", "feature_4"]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df

ml_df = generate_ml_data()

# モデル選択
model_type = st.selectbox("モデルを選択", ["線形回帰", "ランダムフォレスト"])

# データ分割
X = ml_df.drop("target", axis=1)
y = ml_df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデル訓練
if model_type == "線形回帰":
    model = LinearRegression()
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 結果表示
col1, col2 = st.columns(2)
with col1:
    mse = mean_squared_error(y_test, y_pred)
    st.metric("平均二乗誤差 (MSE)", f"{mse:.4f}")
with col2:
    r2 = r2_score(y_test, y_pred)
    st.metric("決定係数 (R²)", f"{r2:.4f}")

# 予測結果のプロット
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name="予測値"))
fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                       y=[y_test.min(), y_test.max()], 
                       mode="lines", name="理想線"))
fig.update_layout(title="実際の値 vs 予測値", xaxis_title="実際の値", yaxis_title="予測値")
st.plotly_chart(fig, use_container_width=True)

# 特徴量重要度（ランダムフォレストの場合）
if model_type == "ランダムフォレスト":
    st.subheader("特徴量重要度")
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    fig = px.bar(importance_df, x="importance", y="feature", orientation="h",
                title="特徴量重要度")
    st.plotly_chart(fig, use_container_width=True)
```

# リアルタイムチャートデモ

elif selected_demo == “リアルタイムチャート”:
st.header(“📊 リアルタイムチャートデモ”)

```
# プレースホルダー
chart_placeholder = st.empty()
metrics_placeholder = st.empty()

# 制御ボタン
col1, col2 = st.columns(2)
with col1:
    if st.button("データ更新開始"):
        st.session_state.update_data = True
with col2:
    if st.button("データ更新停止"):
        st.session_state.update_data = False

# セッション状態の初期化
if "update_data" not in st.session_state:
    st.session_state.update_data = False
if "data_history" not in st.session_state:
    st.session_state.data_history = []

# データ更新
if st.session_state.update_data:
    # 新しいデータポイント生成
    current_time = datetime.now()
    new_value = np.random.normal(100, 15)
    
    st.session_state.data_history.append({
        "time": current_time,
        "value": new_value
    })
    
    # 最新100件のみ保持
    if len(st.session_state.data_history) > 100:
        st.session_state.data_history = st.session_state.data_history[-100:]

# データがある場合のみ表示
if st.session_state.data_history:
    df_realtime = pd.DataFrame(st.session_state.data_history)
    
    # メトリクス更新
    with metrics_placeholder.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("現在値", f"{df_realtime['value'].iloc[-1]:.2f}")
        with col2:
            st.metric("平均値", f"{df_realtime['value'].mean():.2f}")
        with col3:
            st.metric("標準偏差", f"{df_realtime['value'].std():.2f}")
    
    # チャート更新
    with chart_placeholder.container():
        fig = px.line(df_realtime, x="time", y="value", 
                     title="リアルタイムデータ")
        st.plotly_chart(fig, use_container_width=True)
    
    # 自動更新
    if st.session_state.update_data:
        st.rerun()
```

# 画像処理デモ

elif selected_demo == “画像処理”:
st.header(“🖼️ 画像処理デモ”)

```
# 画像アップロード
uploaded_file = st.file_uploader("画像をアップロードしてください", 
                               type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    from PIL import Image, ImageFilter, ImageEnhance
    
    # 画像読み込み
    image = Image.open(uploaded_file)
    
    # オリジナル画像表示
    st.subheader("オリジナル画像")
    st.image(image, caption="アップロードされた画像", use_column_width=True)
    
    # 画像処理オプション
    st.subheader("画像処理オプション")
    col1, col2 = st.columns(2)
    
    with col1:
        brightness = st.slider("明度", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("コントラスト", 0.5, 2.0, 1.0, 0.1)
        saturation = st.slider("彩度", 0.0, 2.0, 1.0, 0.1)
    
    with col2:
        blur_radius = st.slider("ぼかし", 0, 10, 0)
        rotation = st.slider("回転角度", -180, 180, 0)
    
    # 画像処理適用
    processed_image = image.copy()
    
    # 明度調整
    enhancer = ImageEnhance.Brightness(processed_image)
    processed_image = enhancer.enhance(brightness)
    
    # コントラスト調整
    enhancer = ImageEnhance.Contrast(processed_image)
    processed_image = enhancer.enhance(contrast)
    
    # 彩度調整
    enhancer = ImageEnhance.Color(processed_image)
    processed_image = enhancer.enhance(saturation)
    
    # ぼかし
    if blur_radius > 0:
        processed_image = processed_image.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # 回転
    if rotation != 0:
        processed_image = processed_image.rotate(rotation, expand=True)
    
    # 処理済み画像表示
    st.subheader("処理済み画像")
    st.image(processed_image, caption="処理済み画像", use_column_width=True)
    
    # 画像情報表示
    st.subheader("画像情報")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"サイズ: {image.size}")
    with col2:
        st.write(f"モード: {image.mode}")
    with col3:
        st.write(f"フォーマット: {image.format}")
```

# フッター

st.markdown(”—”)
st.markdown(“🚀 Streamlit サンプルアプリケーション - データ分析・機械学習・可視化のデモンストレーション”)
