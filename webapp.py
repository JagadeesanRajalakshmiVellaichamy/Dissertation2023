import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import Word, TextBlob

#configure the dashboard
# Page setting
st.set_page_config(layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title('Indian General Election 2019 Youtube Sentiment Dashboard')

#Read data for charts
df = pd.read_csv("D:\\0_SHU_31018584\\Data\\translated.csv", sep=',')

#############################################################################################
#CHART-1: multi-selection box for years
chartdata1 = df.groupby(['PublishMonth', 'PublishYear']).size().reset_index(name='frequency')
chartdata1['vara'] = df.groupby(['PublishMonth', 'PublishYear'])['ing'].sum().values
chartdata1['varb'] = df.groupby(['PublishMonth', 'PublishYear'])['bjp'].sum().values
st.markdown("Section-1: Trend analysis on monthly youtube comments")
selected_years = st.multiselect('Select years', chartdata1['PublishYear'].unique())
# Filter the DataFrame based on selected years
filtered_df = chartdata1[chartdata1['PublishYear'].isin(selected_years)]
# Bar chart using Plotly Express
fig = px.bar(filtered_df, x='PublishMonth', y='frequency', barmode='group', color_discrete_sequence=px.colors.qualitative.Dark2)

# Line chart for vara and varb using secondary y-axis
line_trace_vara = go.Scatter(
    x=filtered_df['PublishMonth'],
    y=filtered_df['vara'],
    mode='lines+markers',
    name='congress',
    yaxis='y2',
    textposition='bottom right',
    line=dict(color='blue'),  # Customize the line color
    marker=dict(color='blue')
)

line_trace_varb = go.Scatter(
    x=filtered_df['PublishMonth'],
    y=filtered_df['varb'],
    mode='lines+markers',
    name='BJP',
    yaxis='y2',
    textposition='bottom right',
    line=dict(color='orange'),  # Customize the line color
    marker=dict(color='orange')
)

fig.add_trace(line_trace_vara)
fig.add_trace(line_trace_varb)

# Map numeric month values to abbreviated month names
month_name_map = {
    '1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr', '5': 'May', '6': 'Jun',
    '7': 'Jul', '8': 'Aug', '9': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'
}
# Update axis values
fig.update_xaxes(type='category', categoryorder='category ascending', title='Time Period (In Months)', tickvals=list(month_name_map.keys()), ticktext=list(month_name_map.values()))
fig.update_yaxes(title='Number of overall comments')
# Customize chart appearance
fig.update_layout(
    title='Monthly Frequency Distribution of Youtube comments',
    # title_x=0.5,  # Center-align the title
    title_y=0.95,  # Position title at the top
    plot_bgcolor='whitesmoke',
    paper_bgcolor='whitesmoke',
    title_font_color='black',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, tickformat=',d'),
    xaxis_title_font=dict(color='black'),
    yaxis_title_font=dict(color='black'),
    title_font=dict(color='black'),
    font=dict(color='black'),
    margin=dict(t=70, b=50, l=50, r=50),
    showlegend=True,
    # legend=dict(bgcolor='white'),
legend=dict(
        orientation='v',  # Vertical orientation for the legend
        x=1,  # Adjust the x position to move the legend to the right
        y=1.1  # Adjust the y position to center the legend vertically
    ),
    yaxis2=dict(title='Number of comments by Party', overlaying='y', side='right')
    )
#############################################################################################
#CHART-2: multi-selection box for years
data2 = df.groupby(['PublishMonth', 'PublishYear'])['video_id'].nunique().reset_index(name='videos_count')
data2['channel_count'] = df.groupby(['PublishMonth', 'PublishYear'])['yt_channelId'].nunique().values

# Filter the DataFrame based on selected years
filtered_df2 = data2[data2['PublishYear'].isin(selected_years)]
# Bar chart using Plotly Express
fig2 = px.bar(filtered_df2, x='PublishMonth', y='videos_count', barmode='group', color_discrete_sequence=px.colors.qualitative.Safe)

# Line chart for vara and varb using secondary y-axis
line_trace_vara2 = go.Scatter(
    x=filtered_df2['PublishMonth'],
    y=filtered_df2['channel_count'],
    mode='lines+markers',
    name='channels',
    yaxis='y2',
    textposition='bottom right',
    line=dict(color='blue'),  # Customize the line color
    marker=dict(color='blue')
)
fig2.add_trace(line_trace_vara2)

# Map numeric month values to abbreviated month names
month_name_map = {
    '1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr', '5': 'May', '6': 'Jun',
    '7': 'Jul', '8': 'Aug', '9': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'
}
# Update axis values
fig2.update_xaxes(type='category', categoryorder='category ascending', title='Time Period (In Months)', tickvals=list(month_name_map.keys()), ticktext=list(month_name_map.values()))
fig2.update_yaxes(title='Number of Videos')
# Customize chart appearance
fig2.update_layout(
    title='Monthly Frequency Distribution of Youtube Videos and Channels',
    # title_x=0.5,  # Center-align the title
    title_y=0.95,  # Position title at the top
    plot_bgcolor='whitesmoke',
    paper_bgcolor='whitesmoke',
    title_font_color='black',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, tickformat=',d'),
    xaxis_title_font=dict(color='black'),
    yaxis_title_font=dict(color='black'),
    title_font=dict(color='black'),
    font=dict(color='black'),
    margin=dict(t=70, b=50, l=50, r=50),
    showlegend=False,
    # legend=dict(bgcolor='white'),
legend=dict(
        orientation='v',  # Vertical orientation for the legend
        x=1,  # Adjust the x position to move the legend to the right
        y=1.1  # Adjust the y position to center the legend vertically
    ),
    yaxis2=dict(title='Number of Channels', overlaying='y', side='right')
    )

left_column, right_column = st.columns([2, 2])  # Adjust the widths as needed
with left_column:
    st.plotly_chart(fig, use_container_width=True)
with right_column:
    st.plotly_chart(fig2, use_container_width=True)

#############################################################################################
#CHART-3: Slider with multi-selection
chartdata3 = df.groupby(['PublishWeek']).size().reset_index(name='frequency')
chartdata3['vara'] = df.groupby(['PublishWeek'])['ing'].sum().values
chartdata3['varb'] = df.groupby(['PublishWeek'])['bjp'].sum().values
# Filter the DataFrame based on selected years
min_week = min(chartdata3['PublishWeek'])
max_week = max(chartdata3['PublishWeek'])
st.markdown('Section-2: Trend analysis on weekly/hourly youtube comments')
start_year, end_year = st.slider(
    "Select Week Range",
    min_value=min_week, max_value=max_week,
    value=(min_week, max_week))
filtered_df3 = chartdata3[(chartdata3['PublishWeek'] >= start_year) & (chartdata3['PublishWeek'] <= end_year)]
filtered_df3['PublishWeek'] = filtered_df3['PublishWeek'].astype(str).str.zfill(2)
# Bar chart using Plotly Express
fig3 = px.bar(filtered_df3, x='PublishWeek', y='frequency', barmode='group', color_discrete_sequence=px.colors.qualitative.Dark2)

# Line chart for vara and varb using secondary y-axis
line_trace_vara3= go.Scatter(
    x=filtered_df3['PublishWeek'],
    y=filtered_df3['vara'],
    mode='lines+markers',
    name='congress',
    yaxis='y2',
    textposition='bottom right',
    line=dict(color='blue'),  # Customize the line color
    marker=dict(color='blue')
)

line_trace_varb3 = go.Scatter(
    x=filtered_df3['PublishWeek'],
    y=filtered_df3['varb'],
    mode='lines+markers',
    name='BJP',
    yaxis='y2',
    textposition='bottom right',
    line=dict(color='orange'),  # Customize the line color
    marker=dict(color='orange')
)

fig3.add_trace(line_trace_vara3)
fig3.add_trace(line_trace_varb3)
fig3.update_xaxes(type='category', categoryorder='category ascending', title='Time Period (In Weeks)')
fig3.update_yaxes(title='Number of comments')
# Customize chart appearance
fig3.update_layout(
    title='Weekly Frequency Distribution of Youtube comments',
    # title_x=0.5,  # Center-align the title
    title_y=0.95,  # Position title at the top
    plot_bgcolor='whitesmoke',
    paper_bgcolor='whitesmoke',
    title_font_color='black',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, tickformat=',d'),
    xaxis_title_font=dict(color='black'),
    yaxis_title_font=dict(color='black'),
    title_font=dict(color='black'),
    font=dict(color='black'),
    margin=dict(t=70, b=50, l=50, r=50),
    showlegend=True,
legend=dict(
        orientation='v',  # Vertical orientation for the legend
        x=1,  # Adjust the x position to move the legend to the right
        y=1.1  # Adjust the y position to center the legend vertically
    ),
    yaxis2=dict(title='Number of comments by Party', overlaying='y', side='right')
    )

#############################################################################################
#CHART-4: Sidebar with multi-selection box for years
chartdata4 = df.groupby(['PublishHour']).size().reset_index(name='frequency')
chartdata4['vara'] = df.groupby(['PublishHour'])['ing'].sum().values
chartdata4['varb'] = df.groupby(['PublishHour'])['bjp'].sum().values
# Filter the DataFrame based on selected years
# min_week = min(chartdata4['PublishHour'])
# max_week = max(chartdata4['PublishHour'])
# filtered_df4 = chartdata4[(chartdata3['PublishHour'] >= start_year) & (chartdata4['PublishHour'] <= end_year)]
chartdata4['PublishHour'] = chartdata4['PublishHour'].astype(str).str.zfill(2)
# Bar chart using Plotly Express
fig4 = px.bar(chartdata4, x='PublishHour', y='frequency', barmode='group', color_discrete_sequence=px.colors.qualitative.Dark2)

# Line chart for vara and varb using secondary y-axis
line_trace_vara4= go.Scatter(
    x=chartdata4['PublishHour'],
    y=chartdata4['vara'],
    mode='lines+markers',
    name='congress',
    yaxis='y2',
    textposition='bottom right',
    line=dict(color='blue'),  # Customize the line color
    marker=dict(color='blue')
)

line_trace_varb4 = go.Scatter(
    x=chartdata4['PublishHour'],
    y=chartdata4['varb'],
    mode='lines+markers',
    name='BJP',
    yaxis='y2',
    textposition='bottom right',
    line=dict(color='orange'),  # Customize the line color
    marker=dict(color='orange')
)

fig4.add_trace(line_trace_vara4)
fig4.add_trace(line_trace_varb4)
fig4.update_xaxes(type='category', categoryorder='category ascending', title='Time Period (In Hours)')
fig4.update_yaxes(title='Number of comments')
# Customize chart appearance
fig4.update_layout(
    title='Hourly Frequency Distribution of Youtube comments',
    # title_x=0.5,  # Center-align the title
    title_y=0.95,  # Position title at the top
    plot_bgcolor='whitesmoke',
    paper_bgcolor='whitesmoke',
    title_font_color='black',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, tickformat=',d'),
    xaxis_title_font=dict(color='black'),
    yaxis_title_font=dict(color='black'),
    title_font=dict(color='black'),
    font=dict(color='black'),
    margin=dict(t=70, b=50, l=50, r=50),
    showlegend=True,
legend=dict(
        orientation='v',  # Vertical orientation for the legend
        x=1,  # Adjust the x position to move the legend to the right
        y=1.1  # Adjust the y position to center the legend vertically
    ),
    yaxis2=dict(title='Number of comments by Party', overlaying='y', side='right')
    )

left_column1, right_column1 = st.columns([2, 2])  # Adjust the widths as needed
with left_column1:
    st.plotly_chart(fig3, use_container_width=True)
with right_column1:
    st.plotly_chart(fig4, use_container_width=True)

#############################################################################################
#CHART-5: Sidebar with multi-selection box for years
chartdata5 = df.groupby(['language']).size().reset_index(name='frequency')
chartdata5['vara'] = df.groupby(['language'])['ing'].sum().values
chartdata5['varb'] = df.groupby(['language'])['bjp'].sum().values
st.markdown("Section-3: Trend analysis youtube comments by language")
selected_lang = st.multiselect('Select languages', chartdata5['language'].unique())
# Filter the DataFrame based on selected years
filtered_df5 = chartdata5[chartdata5['language'].isin(selected_lang)]
# Bar chart using Plotly Express
fig5 = px.bar(filtered_df5, x='language', y='frequency', barmode='group', color_discrete_sequence=px.colors.qualitative.Dark2)

# Line chart for vara and varb using secondary y-axis
line_trace_vara5 = go.Scatter(
    x=filtered_df5['language'],
    y=filtered_df5['vara'],
    mode='lines+markers',
    name='congress',
    yaxis='y2',
    textposition='bottom right',
    line=dict(color='blue'),  # Customize the line color
    marker=dict(color='blue')
)

line_trace_varb5 = go.Scatter(
    x=filtered_df5['language'],
    y=filtered_df5['varb'],
    mode='lines+markers',
    name='BJP',
    yaxis='y2',
    textposition='bottom right',
    line=dict(color='orange'),  # Customize the line color
    marker=dict(color='orange')
)

fig5.add_trace(line_trace_vara5)
fig5.add_trace(line_trace_varb5)

# Update axis values
fig5.update_xaxes(type='category', categoryorder='category ascending', title='Languages')
fig5.update_yaxes(title='Number of comments')
# Customize chart appearance
fig5.update_layout(
    title='Frequency Distribution of Youtube comments by languages',
    # title_x=0.5,  # Center-align the title
    title_y=0.95,  # Position title at the top
    plot_bgcolor='whitesmoke',
    paper_bgcolor='whitesmoke',
    title_font_color='black',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, tickformat=',d'),
    xaxis_title_font=dict(color='black'),
    yaxis_title_font=dict(color='black'),
    title_font=dict(color='black'),
    font=dict(color='black'),
    margin=dict(t=70, b=50, l=50, r=50),
    showlegend=True,
    # legend=dict(bgcolor='white'),
legend=dict(
        orientation='v',  # Vertical orientation for the legend
        x=1,  # Adjust the x position to move the legend to the right
        y=1.1  # Adjust the y position to center the legend vertically
    ),
    yaxis2=dict(title='Number of comments by Party', overlaying='y', side='right')
    )

#########################################################################################
#CHART-6: Overall distribution of Party comments for analysis
bjp = df['bjp'].sum()
ing = df['ing'].sum()
# Create a new DataFrame with the sum and name columns
result_df1 = pd.DataFrame({'Name': ['bjp'], 'Sum_Value': [bjp]})
result_df2 = pd.DataFrame({'Name': ['ing'], 'Sum_Value': [ing]})
df_pie = pd.concat([result_df1, result_df2], ignore_index=True)
total_count = df_pie['Sum_Value'].sum()
df_pie['Percentage'] = (df_pie['Sum_Value'] / total_count) * 100

# Create a pie chart with % values and labels
fig_pie = px.pie(df_pie, values='Percentage', names='Name', title='Overall distribution of Party comments for analysis',
                 labels={'Percentage': '%'})

# Customize chart appearance
fig_pie.update_traces(textinfo='percent+label')  # Display % values and labels
fig_pie.update_layout(
    title_x=0.0,  # Center-align the title
    plot_bgcolor='white',
    paper_bgcolor='white',
    title_font_color='black',
    font=dict(color='black'),
    margin=dict(t=70, b=50, l=50, r=50)
)

left_column2, mid_column2 = st.columns(2)  # Adjust the widths as needed
with left_column2:
    st.plotly_chart(fig5, use_container_width=True)
with mid_column2:
    st.plotly_chart(fig_pie, use_container_width=True)
# with right_column2:
#     st.plotly_chart(fig4, use_container_width=True)
#############################################################################################
# NLP sentiment analysis using TextBlob
df['TextBlob_polarity'] = df['comment_textDisplay'].apply(lambda x: TextBlob(x).sentiment[0])
df['TextBlob_subjectivity'] = df['comment_textDisplay'].apply(lambda x: TextBlob(x).sentiment[1])
df['TextBlob_sentiment'] = df['TextBlob_polarity'].apply(lambda TextBlob_polarity: 'Positive' if TextBlob_polarity > 0 else 'Negative' if TextBlob_polarity < 0 else 'Neutral')
# BJP
tb_df1 = df[df['bjp'] == 1]
tbresult_df1 = tb_df1['TextBlob_sentiment'].value_counts().reset_index()
tbresult_df1.columns = ['TextBlob_sentiment', 'count']
tbresult_df1['party'] = 'BJP'
tbtotal_count1 = tbresult_df1['count'].sum()
tbresult_df1['Percentage'] = (tbresult_df1['count'] / tbtotal_count1) * 100

# Congress
tb_df2 = df[df['ing'] == 1]
tbresult_df2 = tb_df2['TextBlob_sentiment'].value_counts().reset_index()
tbresult_df2.columns = ['TextBlob_sentiment', 'count']
tbresult_df2['party'] = 'Congress'
tbtotal_count2 = tbresult_df1['count'].sum()
tbresult_df2['Percentage'] = (tbresult_df2['count'] / tbtotal_count2) * 100
TB_bar = pd.concat([tbresult_df1, tbresult_df2], ignore_index=True)
st.markdown("Section-4: Sentiment analysis on youtube comments using Textblob")
# Bar chart using Plotly Express
fig_TB = px.bar(TB_bar, x='TextBlob_sentiment', y='count', color='party', barmode='group', color_discrete_sequence=px.colors.qualitative.Dark2)
# Update axis values
fig_TB.update_xaxes(type='category', categoryorder='category ascending', title='Sentiment Category')
fig_TB.update_yaxes(title='Number of comments')
# Customize chart appearance
fig_TB.update_layout(
    title='Sentiment Distribution of Youtube comments by parties',
    # title_x=0.5,  # Center-align the title
    title_y=0.95,  # Position title at the top
    plot_bgcolor='white',
    paper_bgcolor='white',
    title_font_color='black',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, tickformat=',d'),
    xaxis_title_font=dict(color='black'),
    yaxis_title_font=dict(color='black'),
    title_font=dict(color='black'),
    font=dict(color='black'),
    margin=dict(t=50, b=50, l=50, r=50),
    showlegend=True,
    legend=dict(bgcolor='white')
)

fig_pietb1 = px.pie(tbresult_df1, values='Percentage', names='TextBlob_sentiment', title='Sentiment Distribution of Parties BJP', labels={'Percentage': '%'})
fig_pietb2 = px.pie(tbresult_df2, values='Percentage', names='TextBlob_sentiment', title='Sentiment Distribution of Parties Congress', labels={'Percentage': '%'})
# Customize chart appearance
fig_pietb1.update_traces(textinfo='percent+label')  # Display % values and labels
fig_pietb2.update_traces(textinfo='percent+label')

fig_pietb1.update_layout(
    title_x=0.0,  # Center-align the title
    plot_bgcolor='white',
    paper_bgcolor='white',
    title_font_color='black',
    font=dict(color='black'),
    margin=dict(t=70, b=50, l=50, r=50)
)
fig_pietb2.update_layout(
    title_x=0.0,  # Center-align the title
    plot_bgcolor='white',
    paper_bgcolor='white',
    title_font_color='black',
    font=dict(color='black'),
    margin=dict(t=70, b=50, l=50, r=50)
)

left_column3, mid_column3, right_column3 = st.columns(3)  # Adjust the widths as needed
with left_column3:
    # st.markdown('### Heatmap')
    st.plotly_chart(fig_TB, use_container_width=True)
with mid_column3:
    # st.markdown('### Heatmap')
    st.plotly_chart(fig_pietb1, use_container_width=True)
with right_column3:
    # st.markdown('### Heatmap')
    st.plotly_chart(fig_pietb2, use_container_width=True)












# template
#CHART-2: Sidebar with multi-selection box for years
# chartdata2 = df.groupby(['PublishMonth', 'PublishYear']).size().reset_index(name='frequency')
# selected_years = st.multiselect('Select years', chartdata1['PublishYear'].unique())
# # Filter the DataFrame based on selected years
# filtered_df = chartdata1[chartdata1['PublishYear'].isin(selected_years)]
# # Bar chart using Plotly Express
# fig = px.bar(filtered_df, x='PublishMonth', y='frequency', barmode='group')
# # Add value labels on top of bars
# fig.update_traces(
#     marker_color='#438add',
#     texttemplate='%{y}',  # Use the y value as the label
#     textposition='outside'  # Position the label outside the bar
# )
# # Map numeric month values to abbreviated month names
# month_name_map = {
#     '1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr', '5': 'May', '6': 'Jun',
#     '7': 'Jul', '8': 'Aug', '9': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'
# }
# # Update axis values
# fig.update_xaxes(type='category', categoryorder='category ascending', title='Time Period (In Months)', tickvals=list(month_name_map.keys()), ticktext=list(month_name_map.values()))
# fig.update_yaxes(title='Number of comments')
# # Customize chart appearance
# fig.update_layout(
#     title='Monthly Frequency Distribution of Youtube comments',
#     # title_x=0.5,  # Center-align the title
#     title_y=0.95,  # Position title at the top
#     plot_bgcolor='white',
#     paper_bgcolor='white',
#     title_font_color='black',
#     xaxis=dict(showgrid=False),
#     yaxis=dict(showgrid=False, tickformat=',d'),
#     xaxis_title_font=dict(color='black'),
#     yaxis_title_font=dict(color='black'),
#     title_font=dict(color='black'),
#     font=dict(color='black'),
#     margin=dict(t=50, b=50, l=50, r=50),
#     showlegend=True,
#     legend=dict(bgcolor='white')
# )
# # Display the bar chart
# st.plotly_chart(fig)
#############################################################################################




