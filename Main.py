import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ページ設定

st.set_page_config(
page_title=“データ分析ダッシュボード”,
page_icon=“📊”,
layout=“wide”
)

# タイトル

st.title(“📊 データ分析ダッシュボード”)
st.markdown(”—”)

# サイドバー

st.sidebar.header(“設定”)

# サンプルデータ生成関数

@st.cache_data
def generate_sample_data(n_samples=1000):
“”“サンプルデータを生成”””
np.random.seed(42)

```
dates = pd.date_range(
    start=datetime.now() - timedelta(days=365),
    end=datetime.now(),
    freq='D'
)

data = []
for i, date in enumerate(dates):
    n_records = np.random.randint(1, 6)
    for _ in range(n_records):
        data.append({
            'date': date,
            'sales': np.random.normal(1000, 200),
            'category': np.random.choice(['A', 'B', 'C', 'D']),
            'region': np.random.choice(['東京', '大阪', '名古屋', '福岡']),
            'customer_age': np.random.randint(18, 70),
            'satisfaction': np.random.randint(1, 6)
        })

df = pd.DataFrame(data)
df['sales'] = np.maximum(df['sales'], 0)  # 負の値を除去
return df
```

# データ読み込み

df = generate_sample_data()

# サイドバーでフィルタリング

st.sidebar.subheader(“データフィルタ”)

# 日付範囲選択

date_range = st.sidebar.date_input(
“日付範囲を選択”,
value=(df[‘date’].min().date(), df[‘date’].max().date()),
min_value=df[‘date’].min().date(),
max_value=df[‘date’].max().date()
)

# カテゴリ選択

categories = st.sidebar.multiselect(
“カテゴリを選択”,
options=df[‘category’].unique(),
default=df[‘category’].unique()
)

# 地域選択

regions = st.sidebar.multiselect(
“地域を選択”,
options=df[‘region’].unique(),
default=df[‘region’].unique()
)

# データフィルタリング

if len(date_range) == 2:
start_date, end_date = date_range
filtered_df = df[
(df[‘date’].dt.date >= start_date) &
(df[‘date’].dt.date <= end_date) &
(df[‘category’].isin(categories)) &
(df[‘region’].isin(regions))
]
else:
filtered_df = df[
(df[‘category’].isin(categories)) &
(df[‘region’].isin(regions))
]

# メインコンテンツ

col1, col2, col3, col4 = st.columns(4)

with col1:
st.metric(“総売上”, f”¥{filtered_df[‘sales’].sum():,.0f}”)

with col2:
st.metric(“平均売上”, f”¥{filtered_df[‘sales’].mean():,.0f}”)

with col3:
st.metric(“取引件数”, f”{len(filtered_df):,}”)

with col4:
st.metric(“平均満足度”, f”{filtered_df[‘satisfaction’].mean():.1f}/5”)

st.markdown(”—”)

# グラフエリア

col1, col2 = st.columns(2)

with col1:
st.subheader(“📈 日別売上推移”)
daily_sales = filtered_df.groupby(‘date’)[‘sales’].sum().reset_index()

```
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(daily_sales['date'], daily_sales['sales'], linewidth=2)
ax.set_xlabel('日付')
ax.set_ylabel('売上 (¥)')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
st.pyplot(fig)
```

with col2:
st.subheader(“🥧 カテゴリ別売上”)
category_sales = filtered_df.groupby(‘category’)[‘sales’].sum()

```
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(category_sales.values, labels=category_sales.index, autopct='%1.1f%%')
st.pyplot(fig)
```

st.markdown(”—”)

# 詳細分析

col1, col2 = st.columns(2)

with col1:
st.subheader(“🌍 地域別売上”)
region_sales = filtered_df.groupby(‘region’)[‘sales’].sum().sort_values(ascending=True)

```
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(region_sales)), region_sales.values)
ax.set_yticks(range(len(region_sales)))
ax.set_yticklabels(region_sales.index)
ax.set_xlabel('売上 (¥)')

# カラフルなバー
colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.tight_layout()
st.pyplot(fig)
```

with col2:
st.subheader(“👥 年齢層別分析”)

```
# 年齢層を作成
filtered_df['age_group'] = pd.cut(
    filtered_df['customer_age'], 
    bins=[0, 30, 40, 50, 100], 
    labels=['~30歳', '30-40歳', '40-50歳', '50歳~']
)

age_analysis = filtered_df.groupby('age_group').agg({
    'sales': 'mean',
    'satisfaction': 'mean'
}).round(1)

st.dataframe(age_analysis, use_container_width=True)
```

# データテーブル

st.markdown(”—”)
st.subheader(“📋 データ詳細”)

# データ表示オプション

show_raw_data = st.checkbox(“生データを表示”)

if show_raw_data:
st.dataframe(filtered_df.head(100), use_container_width=True)

# 統計サマリー

with st.expander(“📊 統計サマリー”):
st.write(filtered_df.describe())

# インタラクティブ要素

st.markdown(”—”)
st.subheader(“🎛️ インタラクティブ分析”)

analysis_type = st.selectbox(
“分析タイプを選択”,
[“相関分析”, “売上予測”, “顧客セグメント”]
)

if analysis_type == “相関分析”:
st.write(”**数値項目間の相関**”)
numeric_cols = [‘sales’, ‘customer_age’, ‘satisfaction’]
corr_matrix = filtered_df[numeric_cols].corr()

```
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
st.pyplot(fig)
```

elif analysis_type == “売上予測”:
st.write(”**簡易売上予測（線形回帰）**”)

```
# 簡単な予測モデル
from sklearn.linear_model import LinearRegression

# 特徴量エンジニアリング
model_df = filtered_df.copy()
model_df['day_of_year'] = model_df['date'].dt.dayofyear

X = model_df[['customer_age', 'satisfaction', 'day_of_year']]
y = model_df['sales']

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

col1, col2 = st.columns(2)
with col1:
    st.metric("予測精度 (R²)", f"{model.score(X, y):.3f}")
with col2:
    st.metric("平均予測誤差", f"¥{np.mean(np.abs(y - predictions)):,.0f}")
```

elif analysis_type == “顧客セグメント”:
st.write(”**顧客セグメント分析**”)

```
segment_analysis = filtered_df.groupby(['region', 'category']).agg({
    'sales': ['count', 'mean', 'sum'],
    'satisfaction': 'mean'
}).round(2)

segment_analysis.columns = ['取引数', '平均売上', '総売上', '平均満足度']
st.dataframe(segment_analysis, use_container_width=True)
```

# フッター

st.markdown(”—”)
st.markdown(”*このダッシュボードはStreamlitで作成されました*”)
