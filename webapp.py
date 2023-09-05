"""
Sentiment Analysis Dashboard using YouTube Comments - Prime Minister Election 2019
Author: Jagadeesan Rajalakshmi Vellaichamy
Reviewer: Dani Papamaximou
Created At: 20/08/2023
"""

#Import the necessary python libraries
# pip install pandas
# pip install plotly
# pip install streamlit

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

##########################################################################################
#Step1: The streamlit application page  layout should be set
st.set_page_config(layout="wide")

#Step2: The streamlit application title
st.title('Indian General Election 2019 Youtube Sentiment Dashboard')

#Step3: Read the file from
ReadFilepath = "https://raw.githubusercontent.com/JagadeesanRajalakshmiVellaichamy/Dissertation2023/main/Youtube_Clean_dataframe.csv"
df = pd.read_csv(ReadFilepath, sep=',')

#Step4: Plotting the graphs for the dashboard (Analysis period from Jan to Apr 2019 is considered)
#########################################----SECTION-1----#################################################
#CHART-1.1: ANALYSIS PERIOD - OVERALL PERCENTAGE OF COMMENTS ABOUT BJP AND CONGRESS
bjp = df['bjp'].sum()
ing = df['ing'].sum()
result_df1 = pd.DataFrame({'Name': ['BJP'], 'Sum_Value': [bjp]})
result_df2 = pd.DataFrame({'Name': ['CONGRESS'], 'Sum_Value': [ing]})
df_pie = pd.concat([result_df1, result_df2], ignore_index=True)
total_count = df_pie['Sum_Value'].sum()
df_pie['Percentage'] = (df_pie['Sum_Value'] / total_count) * 100

color_mapping = {
    'BJP': 'orange',
    'CONGRESS': 'blue'
}
df_pie['Color'] = df_pie['Name'].map(color_mapping)

PieChart1_1 = px.pie(df_pie, values='Percentage', names='Name', title='OVERALL DISTRIBUTION - YOUTUBE COMMENTS BY PARTY (01JAN2019 - 10APR2019)', labels={'Percentage': '%'}, color='Name', color_discrete_map=color_mapping)
PieChart1_1.update_traces(textinfo='percent+label')
PieChart1_1.update_layout(title_x=0.0,plot_bgcolor='white',paper_bgcolor='white',title_font_color='black',font=dict(color='black'),margin=dict(t=70, b=50, l=50, r=50))

#CHART-1.2: ANALYSIS PERIOD - MONTHLY YOUTUBE COMMENTS DISTRIBUTION
chartdata1 = df.groupby(['PublishMonth', 'PublishYear']).size().reset_index(name='Frequency')
chartdata1['vara'] = df.groupby(['PublishMonth', 'PublishYear'])['ing'].sum().values
chartdata1['varb'] = df.groupby(['PublishMonth', 'PublishYear'])['bjp'].sum().values
st.markdown("SECTION-1: TREND ANALYSIS OVERALL AND MONTHLY COMMENTS DISTRIBUTION")
selected_years = st.multiselect('Please select analysis time period (Year)', chartdata1['PublishYear'].unique())
final_df = chartdata1[chartdata1['PublishYear'].isin(selected_years)]
BarPlot1_2 = px.bar(final_df, x='PublishMonth', y='Frequency', barmode='group', color_discrete_sequence=['mediumaquamarine'])

line_trace_vara = go.Scatter(x=final_df['PublishMonth'], y=final_df['vara'], mode='lines+markers', name='CONGRESS', yaxis='y2', textposition='bottom right', line=dict(color='blue'), marker=dict(color='blue'))
line_trace_varb = go.Scatter(x=final_df['PublishMonth'], y=final_df['varb'], mode='lines+markers', name='BJP', yaxis='y2', textposition='bottom right', line=dict(color='orange'), marker=dict(color='orange'))
BarPlot1_2.add_trace(line_trace_vara)
BarPlot1_2.add_trace(line_trace_varb)
monname_map = {'1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr', '5': 'May', '6': 'Jun', '7': 'Jul', '8': 'Aug', '9': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
BarPlot1_2.update_xaxes(type='category', categoryorder='category ascending', title='Time Period (In Months)', tickvals=list(monname_map.keys()), ticktext=list(monname_map.values()))
BarPlot1_2.update_yaxes(title='Number of overall comments')

BarPlot1_2.update_layout( title='MONTHLY DISTRIBUTION OF YOUTUBE COMMENTS',
# title_x=0.5,
title_y=0.95,
plot_bgcolor='whitesmoke',
paper_bgcolor='whitesmoke',
title_font_color='grey',
xaxis=dict(showgrid=False),
yaxis=dict(showgrid=False, tickformat=',d'),
xaxis_title_font=dict(color='dark grey'),
yaxis_title_font=dict(color='dark grey'),
title_font=dict(color='dark grey'),
font=dict(color='dark grey'),
margin=dict(t=70, b=50, l=50, r=50),
showlegend=True,
legend=dict(orientation='v', x=1, y=1.1),
yaxis2=dict(title='Number of comments by Party', overlaying='y', side='right', title_font=dict(color='dark grey')))

#SECTION-1 DISPLAY CHARTS
left_column1, right_column1 = st.columns([2, 2])
with left_column1:
    st.plotly_chart(PieChart1_1, use_container_width=True)
with right_column1:
    st.plotly_chart(BarPlot1_2, use_container_width=True)

#########################################----SECTION-2----#################################################
#CHART-2.1: TREND ANALYSIS - MONTHLY YOUTUBE VIDEOS VS CHANNELS
plot2 = df.groupby(['PublishMonth', 'PublishYear'])['video_id'].nunique().reset_index(name='videos_count')
plot2['channel_count'] = df.groupby(['PublishMonth', 'PublishYear'])['yt_channelId'].nunique().values
final_df2 = plot2[plot2['PublishYear'].isin(selected_years)]

BarPlot2_1 = px.bar(final_df2, x='PublishMonth', y='videos_count', barmode='group', color_discrete_sequence=['burlywood'])
line_trace_vara2 = go.Scatter(x=final_df2['PublishMonth'], y=final_df2['channel_count'], mode='lines+markers', name='CHANNELS', yaxis='y2', textposition='bottom right', line=dict(color='green'), marker=dict(color='green'))
BarPlot2_1.add_trace(line_trace_vara2)

monname_map = {'1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr', '5': 'May', '6': 'Jun', '7': 'Jul', '8': 'Aug', '9': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}

BarPlot2_1.update_xaxes(type='category', categoryorder='category ascending', title='Time Period (In Months)', tickvals=list(monname_map.keys()), ticktext=list(monname_map.values()))
BarPlot2_1.update_yaxes(title='Number of Videos')

BarPlot2_1.update_layout(
title='MONTHLY DISTRIBUTION OF YOUTUBE VIDEOS VS CHANNELS',
# title_x=0.5,
title_y=0.95,
plot_bgcolor='whitesmoke',
paper_bgcolor='whitesmoke',
title_font_color='dark grey',
xaxis=dict(showgrid=False),
yaxis=dict(showgrid=False, tickformat=',d'),
xaxis_title_font=dict(color='dark grey'),
yaxis_title_font=dict(color='dark grey'),
title_font=dict(color='dark grey'),
font=dict(color='dark grey'),
margin=dict(t=70, b=50, l=50, r=50),
showlegend=True,
# legend=dict(bgcolor='white'),
legend=dict(orientation='v', x=1, y=1.1), yaxis2=dict(title='Number of Channels', overlaying='y', side='right', title_font=dict(color='dark grey')))


#CHART2.2 - MONTHLY DISTRIBUTION OF YOUTUBE RATING VS LIKES
# BJP
BJP_Filter = df[df['bjp'] == 1]
BJP_Likes = BJP_Filter.groupby('PublishMonth')['comment_likeCount'].sum().reset_index()
BJP_Likes['Type'] = 'BJP Comment Likes'
BJP_Likes = BJP_Likes.rename(columns={'comment_likeCount': 'count'})
BJP_Replies = BJP_Filter.groupby('PublishMonth')['comment_totalReplyCount'].sum().reset_index()
BJP_Replies['Type'] = 'BJP Comment Replies'
BJP_Replies = BJP_Replies.rename(columns={'comment_totalReplyCount': 'count'})

# CONGRESS
INC_Filter = df[df['ing'] == 1]
INC_Likes = INC_Filter.groupby('PublishMonth')['comment_likeCount'].sum().reset_index()
INC_Likes['Type'] = 'CONGRESS Comment Likes'
INC_Likes = INC_Likes.rename(columns={'comment_likeCount': 'count'})
INC_Replies = INC_Filter.groupby('PublishMonth')['comment_totalReplyCount'].sum().reset_index()
INC_Replies['Type'] = 'CONGRESS Comment Replies'
INC_Replies = INC_Replies.rename(columns={'comment_totalReplyCount': 'count'})

TB_bar2_2 = pd.concat([BJP_Likes, BJP_Replies, INC_Likes, INC_Replies], ignore_index=True)

BarPlot2_2 = px.bar(TB_bar2_2, x='PublishMonth', y='count', color='Type', barmode='group', color_discrete_sequence=['orange', 'moccasin', 'dodgerblue', 'deepskyblue'])

monname_map = {'1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr', '5': 'May', '6': 'Jun', '7': 'Jul', '8': 'Aug', '9': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
BarPlot2_2.update_xaxes(type='category', categoryorder='category ascending', title='Time Period (In Months)', tickvals=list(monname_map.keys()), ticktext=list(monname_map.values()))
BarPlot2_2.update_yaxes(title='Number of Likes & Replies')

BarPlot2_2.update_layout(
title='MONTHLY DISTRIBUTION OF YOUTUBE RATING VS LIKES PARTYWISE',
# title_x=0.5,
title_y=0.95,
plot_bgcolor='white',
paper_bgcolor='white',
title_font_color='dark grey',
xaxis=dict(showgrid=False),
yaxis=dict(showgrid=False, tickformat=',d'),
xaxis_title_font=dict(color='dark grey'),
yaxis_title_font=dict(color='dark grey'),
title_font=dict(color='dark grey'),
font=dict(color='dark grey'),
margin=dict(t=50, b=50, l=50, r=50),
showlegend=True,
legend=dict(bgcolor='white')
)

#SECTION-2 DISPLAY CHARTS
left_column2, right_column2 = st.columns([2, 2])
with left_column2:
    st.plotly_chart(BarPlot2_1, use_container_width=True)
with right_column2:
    st.plotly_chart(BarPlot2_2, use_container_width=True)

#########################################----SECTION-3----#################################################
#CHART-3.1: TREND ANALYSIS - WEEKLY DISTRIBUTION YOUTUBE COMMENTS
chartdata3 = df.groupby(['PublishWeek']).size().reset_index(name='frequency')
chartdata3['vara'] = df.groupby(['PublishWeek'])['ing'].sum().values
chartdata3['varb'] = df.groupby(['PublishWeek'])['bjp'].sum().values
min_week = min(chartdata3['PublishWeek'])
max_week = max(chartdata3['PublishWeek'])
st.markdown('SECTION-2: TREND ANALYSIS - WEEKLY/HOURLY COMMENT DISTRIBUTION (01JAN2019 - 10APR2019)')
start_week, end_week = st.slider("Please Select Week Range based on Analysis Window",min_value=min_week, max_value=max_week,value=(min_week, max_week))
filtered_df3 = chartdata3[(chartdata3['PublishWeek'] >= start_week) & (chartdata3['PublishWeek'] <= end_week)]
filtered_df3['PublishWeek'] = filtered_df3['PublishWeek'].astype(str).str.zfill(2)
BarPlot3_1 = px.bar(filtered_df3, x='PublishWeek', y='frequency', barmode='group', color_discrete_sequence=['mediumaquamarine'])

line_trace_vara3= go.Scatter(x=filtered_df3['PublishWeek'],y=filtered_df3['vara'],mode='lines+markers',name='CONGRESS',yaxis='y2',textposition='bottom right',line=dict(color='blue'),  marker=dict(color='blue'))
line_trace_varb3 = go.Scatter(x=filtered_df3['PublishWeek'],y=filtered_df3['varb'],mode='lines+markers',name='BJP',yaxis='y2',textposition='bottom right',line=dict(color='orange'),marker=dict(color='orange'))

BarPlot3_1.add_trace(line_trace_vara3)
BarPlot3_1.add_trace(line_trace_varb3)
BarPlot3_1.update_xaxes(type='category', categoryorder='category ascending', title='Time Period (In Weeks)')
BarPlot3_1.update_yaxes(title='Number of comments')
BarPlot3_1.update_layout(
title='WEEKLY YOUTUBE COMMENTS DISTRIBUTION',
# title_x=0.5,
title_y=0.95,
plot_bgcolor='whitesmoke',
paper_bgcolor='whitesmoke',
title_font_color='dark grey',
xaxis=dict(showgrid=False),
yaxis=dict(showgrid=False, tickformat=',d'),
xaxis_title_font=dict(color='dark grey'),
yaxis_title_font=dict(color='dark grey'),
title_font=dict(color='dark grey'),
font=dict(color='dark grey'),
margin=dict(t=70, b=50, l=50, r=50),
showlegend=True,
legend=dict(orientation='v', x=1,y=1.1),yaxis2=dict(title='Number of comments by Party', overlaying='y', side='right', title_font=dict(color='dark grey')))

#CHART-3.2: HOURLY FREQUENCY OF YOUTUBE COMMENTS
chartdata4 = df.groupby(['PublishWeek', 'PublishHour']).size().reset_index(name='frequency')
chartdata4['vara'] = df.groupby(['PublishWeek', 'PublishHour'])['ing'].sum().values
chartdata4['varb'] = df.groupby(['PublishWeek', 'PublishHour'])['bjp'].sum().values

#Filter from Slider
chartdata4 = chartdata4[(chartdata4['PublishWeek'] >= start_week) & (chartdata4['PublishWeek'] <= end_week)]
chartdata4['PublishWeek'] = filtered_df3['PublishWeek'].astype(str).str.zfill(2)

filtered_df3_2 = chartdata4.groupby('PublishHour').agg({'frequency': 'sum', 'vara': 'sum', 'varb': 'sum'}).reset_index()
filtered_df3_2['PublishHour'] = filtered_df3_2['PublishHour'].astype(str).str.zfill(2)

BarPlot3_2 = px.bar(filtered_df3_2, x='PublishHour', y='frequency', barmode='group', color_discrete_sequence=['mediumaquamarine'])
line_trace_vara4= go.Scatter(x=filtered_df3_2['PublishHour'],y=filtered_df3_2['vara'],mode='lines+markers',name='CONGRESS',yaxis='y2',textposition='bottom right',line=dict(color='blue'),marker=dict(color='blue'))
line_trace_varb4 = go.Scatter(x=filtered_df3_2['PublishHour'],y=filtered_df3_2['varb'],mode='lines+markers',name='BJP',yaxis='y2',textposition='bottom right',line=dict(color='orange'),marker=dict(color='orange'))

BarPlot3_2.add_trace(line_trace_vara4)
BarPlot3_2.add_trace(line_trace_varb4)
BarPlot3_2.update_xaxes(type='category', categoryorder='category ascending', title='Time Period (In Hours)')
BarPlot3_2.update_yaxes(title='Number of comments')
BarPlot3_2.update_layout(
    title='HOURLY YOUTUBE COMMENTS DISTRIBUTION',
    # title_x=0.5,
    title_y=0.95,
    plot_bgcolor='whitesmoke',
    paper_bgcolor='whitesmoke',
    title_font_color='dark grey',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, tickformat=',d'),
    xaxis_title_font=dict(color='dark grey'),
    yaxis_title_font=dict(color='dark grey'),
    title_font=dict(color='dark grey'),
    font=dict(color='dark grey'),
    margin=dict(t=70, b=50, l=50, r=50),
    showlegend=True,
legend=dict(orientation='v',  x=1,  y=1.1  ), yaxis2=dict(title='Number of comments by Party', overlaying='y', side='right'))

left_column3, right_column3 = st.columns([2, 2])
with left_column3:
    st.plotly_chart(BarPlot3_1, use_container_width=True)
with right_column3:
    st.plotly_chart(BarPlot3_2, use_container_width=True)

#########################################----SECTION-4----#################################################
#CHART-4.1: YOUTUBE COMMENTS DISTRIBUTION BY LANGAUGES
chartdata5 = df.groupby(['PublishMonthYear', 'language']).size().reset_index(name='frequency')
chartdata5['vara'] = df.groupby(['PublishMonthYear', 'language'])['ing'].sum().values
chartdata5['varb'] = df.groupby(['PublishMonthYear', 'language'])['bjp'].sum().values
st.markdown("SECTION-3: TREND ANALYSIS LANGUAGE BASED YOUTUBE COMMENTS")
#filter1
selected_lang = st.multiselect('Please select languages from the list', chartdata5['language'].unique())
#filter2
selected_month = st.multiselect('Please select Time Period (In Months)', chartdata5['PublishMonthYear'].unique())

chartdata5 = chartdata5[chartdata5['language'].isin(selected_lang) & chartdata5['PublishMonthYear'].isin(selected_month)]
filtered_df5 = chartdata5.groupby('language').agg({'frequency': 'sum', 'vara': 'sum', 'varb': 'sum'}).reset_index()

BarPlot4_1 = px.bar(filtered_df5, x='language', y='frequency', barmode='group', color_discrete_sequence=['mediumaquamarine'])

line_trace_vara5 = go.Scatter(x=filtered_df5['language'],y=filtered_df5['vara'],mode='lines+markers',name='CONGRESS',yaxis='y2',textposition='bottom right',line=dict(color='blue'),marker=dict(color='blue'))
line_trace_varb5 = go.Scatter(x=filtered_df5['language'],y=filtered_df5['varb'],mode='lines+markers',name='BJP',yaxis='y2',textposition='bottom right',line=dict(color='orange'),marker=dict(color='orange'))

BarPlot4_1.add_trace(line_trace_vara5)
BarPlot4_1.add_trace(line_trace_varb5)

BarPlot4_1.update_xaxes(type='category', categoryorder='category ascending', title='Languages')
BarPlot4_1.update_yaxes(title='Number of comments')
BarPlot4_1.update_layout(
    title='FREQUENCY DISTRIBUTION YOUTUBE COMMENTS LANGUAGES',
    # title_x=0.5,
    title_y=0.95,
    plot_bgcolor='whitesmoke',
    paper_bgcolor='whitesmoke',
    title_font_color='dark grey',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, tickformat=',d'),
    xaxis_title_font=dict(color='dark grey'),
    yaxis_title_font=dict(color='dark grey'),
    title_font=dict(color='dark grey'),
    font=dict(color='dark grey'),
    margin=dict(t=70, b=50, l=50, r=50),
    showlegend=True,
    # legend=dict(bgcolor='white'),
legend=dict(orientation='v',x=1,y=1.1), yaxis2=dict(title='Number of comments by Party', overlaying='y', side='right'))

#CHART-4.2: Overall distribution of Party comments for analysis
bjp = filtered_df5['varb'].sum()
ing = filtered_df5['vara'].sum()
result_df1 = pd.DataFrame({'Name': ['BJP'], 'Sum_Value': [bjp]})
result_df2 = pd.DataFrame({'Name': ['CONGRESS'], 'Sum_Value': [ing]})
df_pie2 = pd.concat([result_df1, result_df2], ignore_index=True)
total_count = df_pie2['Sum_Value'].sum()
df_pie2['Percentage'] = (df_pie2['Sum_Value'] / total_count) * 100

color_mapping = {
    'BJP': 'orange',
    'CONGRESS': 'blue'
}
df_pie2['Color'] = df_pie2['Name'].map(color_mapping)

PieChart4_2 = px.pie(df_pie2, values='Percentage', names='Name', title='PERCENTAGE DISTRIBUTION YOUTUBE COMMENTS LANGUAGES', labels={'Percentage': '%'}, color='Name', color_discrete_map=color_mapping)
PieChart4_2.update_traces(textinfo='percent+label')
PieChart4_2.update_layout(title_x=0.0,plot_bgcolor='white',paper_bgcolor='white',title_font_color='black',font=dict(color='black'),margin=dict(t=70, b=50, l=50, r=50))

left_column2, mid_column2 = st.columns(2)  # Adjust the widths as needed
with left_column2:
    st.plotly_chart(BarPlot4_1, use_container_width=True)
with mid_column2:
    st.plotly_chart(PieChart4_2, use_container_width=True)

#########################################----SECTION-5----#################################################
#CHART5.1: SENTIMENT DISTRIBUTION OF YOUTUBE COMMENTS BY PARTIES
# BJP
tb_df1 = df[df['bjp'] == 1]
tbresult_df1 = tb_df1['mBert_sentiment'].value_counts().reset_index()
tbresult_df1.columns = ['mBert_sentiment', 'count']
tbresult_df1['party'] = 'BJP'
tbtotal_count1 = tbresult_df1['count'].sum()
tbresult_df1['Percentage'] = (tbresult_df1['count'] / tbtotal_count1) * 100

# CONGRESS
tb_df2 = df[df['ing'] == 1]
tbresult_df2 = tb_df2['mBert_sentiment'].value_counts().reset_index()
tbresult_df2.columns = ['mBert_sentiment', 'count']
tbresult_df2['party'] = 'Congress'
tbtotal_count2 = tbresult_df2['count'].sum()
tbresult_df2['Percentage'] = (tbresult_df2['count'] / tbtotal_count2) * 100
TB_bar = pd.concat([tbresult_df1, tbresult_df2], ignore_index=True)
TB_bar = TB_bar.sort_values(by='mBert_sentiment')

st.markdown("SECTION-4: OVERALL SENTIMENTS BASED IN COMMENTS USING mBERT")

fig_TB = px.bar(TB_bar, x='mBert_sentiment', y='count', color='party', barmode='group', color_discrete_sequence=['orange', 'blue'])

fig_TB.update_xaxes(type='category', categoryorder='category ascending', title='Sentiment Category')
fig_TB.update_yaxes(title='Number of comments')

fig_TB.update_layout(
title='YOUTUBE COMMENTS SENTIMENT BY PARTIES',
# title_x=0.5,
title_y=0.95,
plot_bgcolor='white',
paper_bgcolor='white',
title_font_color='dark grey',
xaxis=dict(showgrid=False),
yaxis=dict(showgrid=False, tickformat=',d'),
xaxis_title_font=dict(color='dark grey'),
yaxis_title_font=dict(color='dark grey'),
title_font=dict(color='dark grey'),
font=dict(color='dark grey'),
margin=dict(t=50, b=50, l=50, r=50),
showlegend=True,
legend=dict(bgcolor='white')
)

color_mapping = {
    'Positive': 'forestgreen',
    'Neutral': 'whitesmoke',
    'Negative': 'red'
}

tbresult_df1['Color'] = tbresult_df1['mBert_sentiment'].map(color_mapping)
tbresult_df2['Color'] = tbresult_df2['mBert_sentiment'].map(color_mapping)

fig_pietb1 = px.pie(tbresult_df1, values='Percentage', names='mBert_sentiment', title='BJP SENTIMENT DISTRIBUTION', labels={'Percentage': '%'}, color='mBert_sentiment', color_discrete_map=color_mapping)
fig_pietb2 = px.pie(tbresult_df2, values='Percentage', names='mBert_sentiment', title='CONGRESS SENTIMENT DISTRIBUTION', labels={'Percentage': '%'}, color='mBert_sentiment', color_discrete_map=color_mapping)

fig_pietb1.update_traces(textinfo='percent+label')
fig_pietb2.update_traces(textinfo='percent+label')
fig_pietb1.update_layout(
title_x=0.0,
plot_bgcolor='white',
paper_bgcolor='white',
title_font_color='black',
font=dict(color='black'),
margin=dict(t=70, b=50, l=50, r=50)
)
fig_pietb2.update_layout(
title_x=0.0,
plot_bgcolor='white',
paper_bgcolor='white',
title_font_color='black',
font=dict(color='black'),
margin=dict(t=70, b=50, l=50, r=50)
)

left_column3, mid_column3, right_column3 = st.columns(3)
with left_column3:
    st.plotly_chart(fig_TB, use_container_width=True)
with mid_column3:
    st.plotly_chart(fig_pietb1, use_container_width=True)
with right_column3:
    st.plotly_chart(fig_pietb2, use_container_width=True)

#########################################----SECTION-6----#################################################
#CHART6.1: Sentiments based on youtube comments using mBert
st.markdown("SECTION-5: SENTIMENT DISTRUBUTION BASED IN TIME PERIOD AND LANGUAGES USING mBERT")
st.markdown("Note: Neutral Sentiment is removed, focus is on Positive and Negative Sentiments")

selected_timeperiod = st.multiselect("Select Analysis Timeperiod:", df['PublishMonthYear'].unique())
filtered_df6 = df[df['PublishMonthYear'].isin(selected_timeperiod)]
selected_language = st.multiselect("Select languages:", df['language'].unique())
filtered_df6 = filtered_df6[filtered_df6['language'].isin(selected_language)]

BJP6 = filtered_df6[filtered_df6['bjp'] == 1]
BJP6['PARTY'] = 'BJP'
BJP6 = BJP6.groupby(['PARTY','language', 'PublishMonth', 'mBert_sentiment']).size().reset_index(name='frequency')
BJP6 = BJP6[BJP6['mBert_sentiment'] != 'Neutral']

BJP6_neededcolumn = BJP6[['PARTY', 'mBert_sentiment', 'frequency']]
BJP6_pie = BJP6_neededcolumn.groupby(['PARTY', 'mBert_sentiment']).sum().reset_index()
BJP6_pie['sum'] = BJP6_pie.groupby(['PARTY'])['frequency'].transform('sum')
BJP6_pie['Percentage'] = (BJP6_pie['frequency'] / BJP6_pie['sum']) * 100

# Congress
CONGRESS6 = filtered_df6[filtered_df6['ing'] == 1]
CONGRESS6['PARTY'] = 'CONGRESS'
CONGRESS6 = CONGRESS6.groupby(['PARTY','language', 'PublishMonth', 'mBert_sentiment']).size().reset_index(name='frequency')
CONGRESS6 = CONGRESS6[CONGRESS6['mBert_sentiment'] != 'Neutral']

CONGRESS6_neededcolumn = CONGRESS6[['PARTY', 'mBert_sentiment', 'frequency']]
CONGRESS6_pie = CONGRESS6_neededcolumn.groupby(['PARTY', 'mBert_sentiment']).sum().reset_index()
CONGRESS6_pie['sum'] = CONGRESS6_pie.groupby(['PARTY'])['frequency'].transform('sum')
CONGRESS6_pie['Percentage'] = (CONGRESS6_pie['frequency'] / CONGRESS6_pie['sum']) * 100

ChartData6= pd.concat([BJP6_neededcolumn, CONGRESS6_neededcolumn], ignore_index=True)
ChartData6 = ChartData6.groupby(['PARTY', 'mBert_sentiment']).sum().reset_index()
ChartData6 = ChartData6.sort_values(by='mBert_sentiment')

fig_TB6 = px.bar(ChartData6, x='mBert_sentiment', y='frequency', color='PARTY', barmode='group', color_discrete_sequence=['orange','blue'])
fig_TB6.update_xaxes(type='category', categoryorder='category ascending', title='Sentiment Category')
fig_TB6.update_yaxes(title='Number of comments')
fig_TB6.update_layout(
title='SENTIMENT DISTRIBUTION BY PARTIES',
# title_x=0.5,
title_y=0.95,
plot_bgcolor='white',
paper_bgcolor='white',
title_font_color='dark grey',
xaxis=dict(showgrid=False),
yaxis=dict(showgrid=False, tickformat=',d'),
xaxis_title_font=dict(color='dark grey'),
yaxis_title_font=dict(color='dark grey'),
title_font=dict(color='dark grey'),
font=dict(color='dark grey'),
margin=dict(t=50, b=50, l=50, r=50),
showlegend=True,
legend=dict(bgcolor='white')
)

color_mapping = {
    'Positive': 'forestgreen',
    'Negative': 'red'
}

BJP6_pie['Color'] = BJP6_pie['mBert_sentiment'].map(color_mapping)
CONGRESS6_pie['Color'] = CONGRESS6_pie['mBert_sentiment'].map(color_mapping)

fig_pietb11 = px.pie(BJP6_pie, values='Percentage', names='mBert_sentiment', title='BJP SENTIMENT DISTRIBUTION', labels={'Percentage': '%'}, color='mBert_sentiment', color_discrete_map=color_mapping)
fig_pietb21 = px.pie(CONGRESS6_pie, values='Percentage', names='mBert_sentiment', title='CONGRESS SENTIMENT DISTRIBUTION', labels={'Percentage': '%'}, color='mBert_sentiment', color_discrete_map=color_mapping)
fig_pietb11.update_traces(textinfo='percent+label')
fig_pietb21.update_traces(textinfo='percent+label')

fig_pietb11.update_layout(
title_x=0.0,
plot_bgcolor='white',
paper_bgcolor='white',
title_font_color='black',
font=dict(color='black'),
margin=dict(t=70, b=50, l=50, r=50)
)
fig_pietb21.update_layout(
title_x=0.0,
plot_bgcolor='white',
paper_bgcolor='white',
title_font_color='black',
font=dict(color='black'),
margin=dict(t=70, b=50, l=50, r=50)
)

left_column3, mid_column3, right_column3 = st.columns(3)
with left_column3:
    st.plotly_chart(fig_TB6, use_container_width=True)
with mid_column3:
    st.plotly_chart(fig_pietb11, use_container_width=True)
with right_column3:
    st.plotly_chart(fig_pietb21, use_container_width=True)

#########################################----SECTION-7----#################################################
st.markdown("SECTION-6: mBert BASE VS FINE TUNED MODEL COMPARISON BY REGIONAL LANGUAGES")
ReadFilepath1 = "https://raw.githubusercontent.com/JagadeesanRajalakshmiVellaichamy/Dissertation2023/main/NLP_mBERT_Metrics.csv"
NLPmetrics = pd.read_csv(ReadFilepath1, sep=',')

#CHART7.1: Displaying the Trained model metrics - BAR plots
#filter1
NLPmetrics['LanguageCode'] = NLPmetrics['LanguageCode'].replace({'en': 'English', 'hi': 'Hindi','te': 'Telugu','ta': 'Tamil','ur': 'Urdu','mr': 'Marathi','bn': 'Bengali','or': 'Odia','gu': 'Gujarati','pa': 'Punjabi', 'kn': 'Kannada', 'ml': 'Malayalam'})

IndianLang = st.multiselect('Please select one or more language from the list', NLPmetrics['LanguageCode'].unique())

NLPmetrics_filtered = NLPmetrics[NLPmetrics['LanguageCode'].isin(IndianLang)]

BarPlot7_1 = px.bar(NLPmetrics_filtered, x='LanguageCode', y='Accuracy', color='ModelName', barmode='group', color_discrete_sequence=['royalblue','limegreen'])
BarPlot7_2 = px.bar(NLPmetrics_filtered, x='LanguageCode', y='Precision', color='ModelName', barmode='group', color_discrete_sequence=['lightcoral','limegreen'])
BarPlot7_3 = px.bar(NLPmetrics_filtered, x='LanguageCode', y='Recall', color='ModelName', barmode='group', color_discrete_sequence=['palevioletred','limegreen'])
BarPlot7_4 = px.bar(NLPmetrics_filtered, x='LanguageCode', y='F1Score', color='ModelName', barmode='group', color_discrete_sequence=['sandybrown','limegreen'])

BarPlot7_1.update_xaxes(type='category', categoryorder='category ascending', title='Languages')
BarPlot7_2.update_xaxes(type='category', categoryorder='category ascending', title='Languages')
BarPlot7_3.update_xaxes(type='category', categoryorder='category ascending', title='Languages')
BarPlot7_4.update_xaxes(type='category', categoryorder='category ascending', title='Languages')

BarPlot7_1.update_layout(title='mBERT BASE VS FINE TUNED MODEL - ACCURACY', title_y=0.95, plot_bgcolor='white', paper_bgcolor='white', title_font_color='dark grey', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), xaxis_title_font=dict(color='dark grey'), yaxis_title_font=dict(color='dark grey'), title_font=dict(color='dark grey'), font=dict(color='dark grey'), margin=dict(t=50, b=50, l=50, r=50), showlegend=True, legend=dict(bgcolor='white'))
BarPlot7_2.update_layout(title='mBERT BASE VS FINE TUNED MODEL - PRECISION', title_y=0.95, plot_bgcolor='white', paper_bgcolor='white', title_font_color='dark grey', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), xaxis_title_font=dict(color='dark grey'), yaxis_title_font=dict(color='dark grey'), title_font=dict(color='dark grey'), font=dict(color='dark grey'), margin=dict(t=50, b=50, l=50, r=50), showlegend=True, legend=dict(bgcolor='white'))
BarPlot7_3.update_layout(title='mBERT BASE VS FINE TUNED MODEL - RECALL', title_y=0.95, plot_bgcolor='white', paper_bgcolor='white', title_font_color='dark grey', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), xaxis_title_font=dict(color='dark grey'), yaxis_title_font=dict(color='dark grey'), title_font=dict(color='dark grey'), font=dict(color='dark grey'), margin=dict(t=50, b=50, l=50, r=50), showlegend=True, legend=dict(bgcolor='white'))
BarPlot7_4.update_layout(title='mBERT BASE VS FINE TUNED MODEL - F1SCORE', title_y=0.95, plot_bgcolor='white', paper_bgcolor='white', title_font_color='dark grey', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), xaxis_title_font=dict(color='dark grey'), yaxis_title_font=dict(color='dark grey'), title_font=dict(color='dark grey'), font=dict(color='dark grey'), margin=dict(t=50, b=50, l=50, r=50), showlegend=True, legend=dict(bgcolor='white'))

left_column7, right_column7 = st.columns(2)
with left_column7:
    st.plotly_chart(BarPlot7_1, use_container_width=True)
with right_column7:
    st.plotly_chart(BarPlot7_2, use_container_width=True)

left_column8, right_column8 = st.columns(2)
with left_column8:
    st.plotly_chart(BarPlot7_3, use_container_width=True)
with right_column8:
    st.plotly_chart(BarPlot7_4, use_container_width=True)

#CHART7.2: Displaying the Trained model metrics - TABULAR
basemodeldf=NLPmetrics[NLPmetrics['ModelName'] == 'mBERT Base Model']
basemodeldf=basemodeldf.drop('ModelName', axis=1)
Finetunedf=NLPmetrics[NLPmetrics['ModelName'] == 'mBERT Finetuned Model']
Finetunedf=Finetunedf.drop('ModelName', axis=1)

basemodelcolumn_suffixes = {'LanguageCode': 'LanguageCode',
                   'Accuracy': 'Accuracy_BaseModel',
                   'Precision': 'Precision_BaseModel',
                   'Recall': 'Recall_BaseModel',
                   'F1Score': 'F1Score_BaseModel'}
fitmodelcolumn_suffixes = {'LanguageCode': 'LanguageCode',
                   'Accuracy': 'Accuracy_FinetunedModel',
                   'Precision': 'Precision_FinetunedModel',
                   'Recall': 'Recall_FinetunedModel',
                   'F1Score': 'F1Score_FinetunedModel'}
basemodeldf = basemodeldf.rename(columns=basemodelcolumn_suffixes)
Finetunedf = Finetunedf.rename(columns=fitmodelcolumn_suffixes)
Final_metrics = pd.merge(basemodeldf, Finetunedf, on='LanguageCode', how='inner')
Final_metrics['LanguageCode'] = Final_metrics['LanguageCode'].replace({'en': 'English', 'hi': 'Hindi','te': 'Telugu','ta': 'Tamil','ur': 'Urdu','mr': 'Marathi','bn': 'Bengali','or': 'Odia','gu': 'Gujarati','pa': 'Punjabi', 'kn': 'Kannada', 'ml': 'Malayalam'})
Final_metrics_filtered = Final_metrics[Final_metrics['LanguageCode'].isin(IndianLang)]
st.markdown("SECTION-7: mBert BASE VS FINE TUNED MODEL COMPARISON BY REGIONAL LANGUAGES TABULAR VIEW")
st.write(Final_metrics_filtered)
