import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Streamlit Sample", page_icon="🎈", layout="centered")

st.title("🎈 Streamlit サンプルアプリ")
st.write("これは簡単な Streamlit アプリの例です。")

# 1. インタラクティブウィジェット
st.header("1. インタラクティブウィジェット")
number = st.slider("数を選んでください", 0, 100, 50)
st.write(f"選択した値の 2 乗は **{number**2}** です。")

# 2. DataFrame とグラフ
st.header("2. DataFrame とグラフ")
df = pd.DataFrame({
    "x": np.arange(0, 100),
    "y": np.random.randn(100).cumsum()
})
st.write("生成したデータフレーム（先頭 5 行）:", df.head())

chart = (
    alt.Chart(df)
    .mark_line()
    .encode(x="x", y="y")
    .properties(width=600, height=400, title="累積ランダムウォーク")
)
st.altair_chart(chart, use_container_width=True)

# 3. ファイルダウンロード
st.header("3. ファイルダウンロード")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("CSV をダウンロード", csv, "sample.csv", "text/csv")

st.write("👈 左側のメニューから *Rerun* してもう一度試してください。")
