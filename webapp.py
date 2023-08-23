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
df = pd.read_csv("C:\\Dissertation_2023\\Youtube_Clean_dataframe.csv", sep=',')

#Step4: Plotting the graphs for the dashboard (Analysis period from Jan to Apr 2019 is considered)
#Plot1: Trend analysis on monthly youtube comments
chartdata1 = df.groupby(['PublishMonth', 'PublishYear']).size().reset_index(name='Frequency')
chartdata1['vara'] = df.groupby(['PublishMonth', 'PublishYear'])['ing'].sum().values
chartdata1['varb'] = df.groupby(['PublishMonth', 'PublishYear'])['bjp'].sum().values
st.markdown("Section-1: Trend analysis on monthly youtube comments")
selected_years = st.multiselect('Select Analysis Time Period (In Year)', chartdata1['PublishYear'].unique())
final_df = chartdata1[chartdata1['PublishYear'].isin(selected_years)]
fig = px.bar(final_df, x='PublishMonth', y='Frequency', barmode='group', color_discrete_sequence=[px.colors.qualitative.Dark2[0]])

line_trace_vara = go.Scatter(x=final_df['PublishMonth'], y=final_df['vara'], mode='lines+markers', name='CONGRESS', yaxis='y2', textposition='bottom right', line=dict(color='blue'), marker=dict(color='blue'))
line_trace_varb = go.Scatter(x=final_df['PublishMonth'], y=final_df['varb'], mode='lines+markers', name='BJP', yaxis='y2', textposition='bottom right', line=dict(color='orange'), marker=dict(color='orange'))
fig.add_trace(line_trace_vara)
fig.add_trace(line_trace_varb)
monname_map = {'1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr', '5': 'May', '6': 'Jun', '7': 'Jul', '8': 'Aug', '9': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
fig.update_xaxes(type='category', categoryorder='category ascending', title='Time Period (In Months)', tickvals=list(monname_map.keys()), ticktext=list(monname_map.values()))
fig.update_yaxes(title='Number of overall comments')

fig.update_layout( title='Monthly Frequency Distribution of Youtube comments',
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

#############################################################################################
#Step4: Plotting the graphs for the dashboard (Analysis period from Jan to Apr 2019 is considered)
#Plot2: Trend analysis on monthly youtube videos vs Channels
plot2 = df.groupby(['PublishMonth', 'PublishYear'])['video_id'].nunique().reset_index(name='videos_count')
plot2['channel_count'] = df.groupby(['PublishMonth', 'PublishYear'])['yt_channelId'].nunique().values
final_df2 = plot2[plot2['PublishYear'].isin(selected_years)]

fig2 = px.bar(final_df2, x='PublishMonth', y='videos_count', barmode='group', color_discrete_sequence=px.colors.qualitative.Safe)
line_trace_vara2 = go.Scatter(x=final_df2['PublishMonth'], y=final_df2['channel_count'], mode='lines+markers', name='CHANNELS', yaxis='y2', textposition='bottom right', line=dict(color='green'), marker=dict(color='green'))
fig2.add_trace(line_trace_vara2)

monname_map = {'1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr', '5': 'May', '6': 'Jun', '7': 'Jul', '8': 'Aug', '9': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}

fig2.update_xaxes(type='category', categoryorder='category ascending', title='Time Period (In Months)', tickvals=list(monname_map.keys()), ticktext=list(monname_map.values()))
fig2.update_yaxes(title='Number of Videos')

fig2.update_layout(
title='Monthly Frequency Distribution of Youtube Videos and Channels',
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

left_column, right_column = st.columns([2, 2])  # Adjust the widths as needed
with left_column:
    st.plotly_chart(fig, use_container_width=True)
with right_column:
    st.plotly_chart(fig2, use_container_width=True)

#############################################################################################
#CHART-3: Weekly Frequency Distribution of Youtube comments
chartdata3 = df.groupby(['PublishWeek']).size().reset_index(name='frequency')
chartdata3['vara'] = df.groupby(['PublishWeek'])['ing'].sum().values
chartdata3['varb'] = df.groupby(['PublishWeek'])['bjp'].sum().values
min_week = min(chartdata3['PublishWeek'])
max_week = max(chartdata3['PublishWeek'])
st.markdown('Section-2: Trend analysis on weekly/hourly youtube comments')
start_year, end_year = st.slider("Select Week Range based on Analysis Window",min_value=min_week, max_value=max_week,value=(min_week, max_week))
filtered_df3 = chartdata3[(chartdata3['PublishWeek'] >= start_year) & (chartdata3['PublishWeek'] <= end_year)]
filtered_df3['PublishWeek'] = filtered_df3['PublishWeek'].astype(str).str.zfill(2)
fig3 = px.bar(filtered_df3, x='PublishWeek', y='frequency', barmode='group', color_discrete_sequence=px.colors.qualitative.Dark2)

line_trace_vara3= go.Scatter(x=filtered_df3['PublishWeek'],y=filtered_df3['vara'],mode='lines+markers',name='CONGRESS',yaxis='y2',textposition='bottom right',line=dict(color='blue'),  marker=dict(color='blue'))
line_trace_varb3 = go.Scatter(x=filtered_df3['PublishWeek'],y=filtered_df3['varb'],mode='lines+markers',name='BJP',yaxis='y2',textposition='bottom right',line=dict(color='orange'),marker=dict(color='orange'))

fig3.add_trace(line_trace_vara3)
fig3.add_trace(line_trace_varb3)
fig3.update_xaxes(type='category', categoryorder='category ascending', title='Time Period (In Weeks)')
fig3.update_yaxes(title='Number of comments')
fig3.update_layout(
title='Weekly Frequency Distribution of Youtube comments',
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

#############################################################################################
#CHART-4: Hourly Frequency Distribution of Youtube comments
chartdata4 = df.groupby(['PublishHour']).size().reset_index(name='frequency')
chartdata4['vara'] = df.groupby(['PublishHour'])['ing'].sum().values
chartdata4['varb'] = df.groupby(['PublishHour'])['bjp'].sum().values
chartdata4['PublishHour'] = chartdata4['PublishHour'].astype(str).str.zfill(2)

fig4 = px.bar(chartdata4, x='PublishHour', y='frequency', barmode='group', color_discrete_sequence=px.colors.qualitative.Dark2)
line_trace_vara4= go.Scatter(x=chartdata4['PublishHour'],y=chartdata4['vara'],mode='lines+markers',name='CONGRESS',yaxis='y2',textposition='bottom right',line=dict(color='blue'),marker=dict(color='blue'))
line_trace_varb4 = go.Scatter(x=chartdata4['PublishHour'],y=chartdata4['varb'],mode='lines+markers',name='BJP',yaxis='y2',textposition='bottom right',line=dict(color='orange'),marker=dict(color='orange'))

fig4.add_trace(line_trace_vara4)
fig4.add_trace(line_trace_varb4)
fig4.update_xaxes(type='category', categoryorder='category ascending', title='Time Period (In Hours)')
fig4.update_yaxes(title='Number of comments')
fig4.update_layout(
    title='Hourly Frequency Distribution of Youtube comments',
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

left_column1, right_column1 = st.columns([2, 2])
with left_column1:
    st.plotly_chart(fig3, use_container_width=True)
with right_column1:
    st.plotly_chart(fig4, use_container_width=True)

#############################################################################################
#CHART-5: Frequency Distribution of Youtube comments by languages
chartdata5 = df.groupby(['language']).size().reset_index(name='frequency')
chartdata5['vara'] = df.groupby(['language'])['ing'].sum().values
chartdata5['varb'] = df.groupby(['language'])['bjp'].sum().values
st.markdown("Section-3: Trend analysis youtube comments by language")
selected_lang = st.multiselect('Select languages', chartdata5['language'].unique())
filtered_df5 = chartdata5[chartdata5['language'].isin(selected_lang)]
fig5 = px.bar(filtered_df5, x='language', y='frequency', barmode='group', color_discrete_sequence=px.colors.qualitative.Dark2)

line_trace_vara5 = go.Scatter(x=filtered_df5['language'],y=filtered_df5['vara'],mode='lines+markers',name='CONGRESS',yaxis='y2',textposition='bottom right',line=dict(color='blue'),marker=dict(color='blue'))
line_trace_varb5 = go.Scatter(x=filtered_df5['language'],y=filtered_df5['varb'],mode='lines+markers',name='BJP',yaxis='y2',textposition='bottom right',line=dict(color='orange'),marker=dict(color='orange'))

fig5.add_trace(line_trace_vara5)
fig5.add_trace(line_trace_varb5)

fig5.update_xaxes(type='category', categoryorder='category ascending', title='Languages')
fig5.update_yaxes(title='Number of comments')
fig5.update_layout(
    title='Frequency Distribution of Youtube comments by languages',
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

#########################################################################################
#CHART-6: Overall distribution of Party comments for analysis
bjp = df['bjp'].sum()
ing = df['ing'].sum()
result_df1 = pd.DataFrame({'Name': ['BJP'], 'Sum_Value': [bjp]})
result_df2 = pd.DataFrame({'Name': ['CONGRESS'], 'Sum_Value': [ing]})
df_pie = pd.concat([result_df1, result_df2], ignore_index=True)
total_count = df_pie['Sum_Value'].sum()
df_pie['Percentage'] = (df_pie['Sum_Value'] / total_count) * 100
fig_pie = px.pie(df_pie, values='Percentage', names='Name', title='Overall distribution of Party comments for analysis', labels={'Percentage': '%'}, color_discrete_sequence=['blue', 'orange'])

fig_pie.update_traces(textinfo='percent+label')
fig_pie.update_layout(title_x=0.0,plot_bgcolor='white',paper_bgcolor='white',title_font_color='black',font=dict(color='black'),margin=dict(t=70, b=50, l=50, r=50))

left_column2, mid_column2 = st.columns(2)  # Adjust the widths as needed
with left_column2:
    st.plotly_chart(fig5, use_container_width=True)
with mid_column2:
    st.plotly_chart(fig_pie, use_container_width=True)

#############################################################################################
#CHART7,8,9: Sentiment Distribution of Youtube comments by parties
# BJP
tb_df1 = df[df['bjp'] == 1]
tbresult_df1 = tb_df1['mBert_sentiment'].value_counts().reset_index()
tbresult_df1.columns = ['mBert_sentiment', 'count']
tbresult_df1['party'] = 'BJP'
tbtotal_count1 = tbresult_df1['count'].sum()
tbresult_df1['Percentage'] = (tbresult_df1['count'] / tbtotal_count1) * 100

# Congress
tb_df2 = df[df['ing'] == 1]
tbresult_df2 = tb_df2['mBert_sentiment'].value_counts().reset_index()
tbresult_df2.columns = ['mBert_sentiment', 'count']
tbresult_df2['party'] = 'Congress'
tbtotal_count2 = tbresult_df2['count'].sum()
tbresult_df2['Percentage'] = (tbresult_df2['count'] / tbtotal_count2) * 100
TB_bar = pd.concat([tbresult_df1, tbresult_df2], ignore_index=True)
TB_bar = TB_bar.sort_values(by='mBert_sentiment')

st.markdown("Section-4: Sentiments based on youtube comments using mBert")

fig_TB = px.bar(TB_bar, x='mBert_sentiment', y='count', color='party', barmode='group', color_discrete_sequence=['orange', 'blue'])

fig_TB.update_xaxes(type='category', categoryorder='category ascending', title='Sentiment Category')
fig_TB.update_yaxes(title='Number of comments')

fig_TB.update_layout(
title='Sentiment Distribution of Youtube comments by parties',
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

fig_pietb1 = px.pie(tbresult_df1, values='Percentage', names='mBert_sentiment', title='Sentiment Distribution of BJP', labels={'Percentage': '%'}, color_discrete_sequence=['red','forestgreen','gainsboro'])
fig_pietb2 = px.pie(tbresult_df2, values='Percentage', names='mBert_sentiment', title='Sentiment Distribution of CONGRESS', labels={'Percentage': '%'}, color_discrete_sequence=['red','forestgreen','gainsboro'])

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
#############################################################################################
#CHART-10, 11, 12: Sentiments based on youtube comments using mBert
st.markdown("Section-5: Sentiments based on youtube comments using mBert")
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
title='Sentiment Distribution of Youtube comments by parties',
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

fig_pietb11 = px.pie(BJP6_pie, values='Percentage', names='mBert_sentiment', title='Sentiment Distribution of BJP', labels={'Percentage': '%'}, color='mBert_sentiment', color_discrete_map=color_mapping)
fig_pietb21 = px.pie(CONGRESS6_pie, values='Percentage', names='mBert_sentiment', title='Sentiment Distribution of CONGRESS', labels={'Percentage': '%'}, color='mBert_sentiment', color_discrete_map=color_mapping)
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

##############################################################################
#CHART13: Displaying the Trained model metrics
st.markdown("Section-6: mBert model evaluation metrics")
NLPmetrics = pd.read_csv("C:\\Dissertation_2023\\NLP_mBERT_Metrics.csv", sep=',')

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
st.write(Final_metrics)
