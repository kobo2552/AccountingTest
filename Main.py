import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Configure the page

st.set_page_config(
page_title="Interactive Data Dashboard",
page_icon="📊",
layout="wide",
initial_sidebar_state="expanded"
)

# Custom CSS for better styling

st.markdown("""

<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>

""", unsafe_allow_html=True)

# Main title

st.markdown('<h1 class="main-header">📊 Interactive Data Dashboard</h1>', unsafe_allow_html=True)

# Sidebar

st.sidebar.title("🎛️ Dashboard Controls")
st.sidebar.markdown("—")

# Sample data generation

@st.cache_data
def generate_sample_data():
"""Generate sample data for the dashboard"""
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')

```
df = pd.DataFrame({
    'date': dates,
    'sales': np.random.normal(1000, 200, len(dates)).cumsum(),
    'customers': np.random.poisson(50, len(dates)),
    'revenue': np.random.normal(5000, 1000, len(dates)),
    'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], len(dates)),
    'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates))
})

# Add some trend
df['sales'] = df['sales'] + np.arange(len(df)) * 2
df['revenue'] = df['revenue'] + np.arange(len(df)) * 10

return df
```

# Load data

df = generate_sample_data()

# Sidebar filters

st.sidebar.subheader("📅 Date Range")
date_range = st.sidebar.date_input(
"Select date range:",
value=(df['date'].min(), df['date'].max()),
min_value=df['date'].min(),
max_value=df['date'].max()
)

st.sidebar.subheader("🏷️ Category Filter")
categories = st.sidebar.multiselect(
"Select categories:",
options=df['category'].unique(),
default=df['category'].unique()
)

st.sidebar.subheader("🗺️ Region Filter")
regions = st.sidebar.multiselect(
"Select regions:",
options=df['region'].unique(),
default=df['region'].unique()
)

# Filter data based on selections

if len(date_range) == 2:
mask = (df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))
filtered_df = df[mask]
else:
filtered_df = df

filtered_df = filtered_df[filtered_df['category'].isin(categories)]
filtered_df = filtered_df[filtered_df['region'].isin(regions)]

# Main dashboard content

col1, col2, col3, col4 = st.columns(4)

# Key metrics

with col1:
total_sales = filtered_df['sales'].iloc[-1] if len(filtered_df) > 0 else 0
st.metric(
label="📈 Total Sales",
value=f"{total_sales:,.0f}",
delta=f"{filtered_df['sales'].diff().mean():.1f} avg daily"
)

with col2:
total_customers = filtered_df['customers'].sum()
st.metric(
label="👥 Total Customers",
value=f"{total_customers:,}",
delta=f"{filtered_df['customers'].mean():.1f} avg daily"
)

with col3:
total_revenue = filtered_df['revenue'].sum()
st.metric(
label="💰 Total Revenue",
value=f"${total_revenue:,.0f}",
delta=f"${filtered_df['revenue'].mean():.0f} avg daily"
)

with col4:
avg_order_value = total_revenue / total_customers if total_customers > 0 else 0
st.metric(
label="🛒 Avg Order Value",
value=f"${avg_order_value:.2f}",
delta="12.3%"
)

st.markdown("—")

# Charts section

col1, col2 = st.columns(2)

with col1:
st.subheader("📊 Sales Trend Over Time")
if len(filtered_df) > 0:
fig_sales = px.line(
filtered_df,
x='date',
y='sales',
title='Daily Sales Progression',
color_discrete_sequence=['#1f77b4']
)
fig_sales.update_layout(
xaxis_title="Date",
yaxis_title="Cumulative Sales",
hovermode='x unified'
)
st.plotly_chart(fig_sales, use_container_width=True)
else:
st.info("No data available for selected filters")

with col2:
st.subheader("🥧 Sales by Category")
if len(filtered_df) > 0:
category_sales = filtered_df.groupby('category')['revenue'].sum().reset_index()
fig_pie = px.pie(
category_sales,
values='revenue',
names='category',
title='Revenue Distribution by Category'
)
st.plotly_chart(fig_pie, use_container_width=True)
else:
st.info("No data available for selected filters")

# Full width chart

st.subheader("📈 Revenue and Customer Trends")
if len(filtered_df) > 0:
fig_dual = go.Figure()

```
# Add revenue trace
fig_dual.add_trace(
    go.Scatter(
        x=filtered_df['date'],
        y=filtered_df['revenue'],
        name='Revenue',
        line=dict(color='#1f77b4'),
        yaxis='y'
    )
)

# Add customer trace
fig_dual.add_trace(
    go.Scatter(
        x=filtered_df['date'],
        y=filtered_df['customers'],
        name='Customers',
        line=dict(color='#ff7f0e'),
        yaxis='y2'
    )
)

# Update layout for dual y-axis
fig_dual.update_layout(
    title='Revenue and Customer Count Over Time',
    xaxis_title='Date',
    yaxis=dict(
        title='Revenue ($)',
        side='left'
    ),
    yaxis2=dict(
        title='Customers',
        side='right',
        overlaying='y'
    ),
    hovermode='x unified'
)

st.plotly_chart(fig_dual, use_container_width=True)
```

st.markdown("—")

# Interactive features section

col1, col2 = st.columns(2)

with col1:
st.subheader("🎯 Interactive Features")

```
# Number input
threshold = st.number_input(
    "Revenue Threshold:",
    min_value=0,
    max_value=10000,
    value=3000,
    step=500
)

# Show days above threshold
if len(filtered_df) > 0:
    above_threshold = len(filtered_df[filtered_df['revenue'] > threshold])
    st.write(f"Days with revenue above ${threshold}: **{above_threshold}**")

# Slider for moving average
ma_days = st.slider("Moving Average Days:", 1, 30, 7)

# Selectbox for chart type
chart_type = st.selectbox(
    "Chart Style:",
    ["Line", "Area", "Bar"]
)
```

with col2:
st.subheader("📋 Data Sample")
if len(filtered_df) > 0:
st.dataframe(
filtered_df.tail(10)[['date', 'sales', 'customers', 'revenue', 'category']],
use_container_width=True
)

# File upload section

st.markdown("—")
st.subheader("📁 Upload Your Own Data")
uploaded_file = st.file_uploader(
"Choose a CSV file",
type=['csv'],
help="Upload a CSV file with columns: date, sales, customers, revenue, category, region"
)

if uploaded_file is not None:
try:
user_df = pd.read_csv(uploaded_file)
st.success("File uploaded successfully!")
st.write("Preview of uploaded data:")
st.dataframe(user_df.head())
except Exception as e:
st.error(f"Error reading file: {e}")

# Real-time simulation

st.markdown("—")
st.subheader("⚡ Real-time Data Simulation")

if st.button("Start Real-time Simulation"):
placeholder = st.empty()
progress_bar = st.progress(0)

```
for i in range(10):
    # Generate random data point
    new_data = {
        'Time': datetime.now().strftime('%H:%M:%S'),
        'Value': np.random.randint(50, 150)
    }
    
    # Update placeholder
    with placeholder.container():
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Value", new_data['Value'])
        with col2:
            st.metric("Timestamp", new_data['Time'])
    
    # Update progress
    progress_bar.progress((i + 1) / 10)
    time.sleep(1)

st.success("Simulation completed!")
```

# Footer

st.markdown("—")
st.markdown("""

<div style='text-align: center; color: #666666;'>
    <p>Built with ❤️ using Streamlit | Last updated: {}</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M')), unsafe_allow_html=True)

# Sidebar info

st.sidebar.markdown("—")
st.sidebar.info("""
**Dashboard Features:**

- 📊 Interactive charts
- 🎛️ Dynamic filtering
- 📈 Real-time simulation
- 📁 File upload
- 📱 Responsive design
  """)

st.sidebar.success("Dashboard loaded successfully!")
