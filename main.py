#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:09:34 2021

@author: kaustuv
"""
import streamlit as st 
import requests
import json
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize

def read_data_thingspeak():
    URL = 'https://api.thingspeak.com/channels/1097511/feeds.json?api_key='
    KEY = 'G8PPL9L3I2CSCRJ2'
    HEADER = '&results='
    POINTS = 180
    NEW_URL = URL+KEY+HEADER+str(POINTS)
#     print(NEW_URL)
    
    get_data = requests.get(NEW_URL).json()
#     print(get_data)
    
    channel_id = get_data['channel']['id']
    
    feed = get_data['feeds']
    
    t = []
    maxTemp = []
    minTemp = []
    relHum = []
    precip = []
    
    for x in feed:
        date_obj = dt.datetime.strptime(x['created_at'],'%Y-%m-%dT%H:%M:%SZ')
        
        t.append(date_obj.date())
        maxTemp.append(np.float(x['field1']))
        minTemp.append(np.float(x['field2']))
        relHum.append(np.float(x['field3']))
        precip.append(np.float(x['field4']))
    
    t = np.array(t)
    maxTemp = np.array(maxTemp)
    minTemp = np.array(minTemp)
    relHum = np.array(relHum)
    precip = np.array(precip)
    
    return t, maxTemp, minTemp, relHum, precip

# Fetch weather data
t, maxTemp, minTemp, relHum, precip = read_data_thingspeak()

# Calculate Heat Index
meanTemp = (minTemp + maxTemp)/2

T = (meanTemp*9/5)+32;
R = relHum;
c1 = -42.379;
c2 = 2.04901523;
c3 = 10.14333127;
c4 = -0.22475541;
c5 = -6.83783e-3;
c6 = -5.481717e-2;
c7 = 1.22874e-3;
c8 = 8.5282e-4;
c9 = -1.99e-6;
HI = c1+c2*T+c3*R+c4*T*R+c5*T**2+c6*R**2+c7*T**2*R+c8*T*R**2+c9*T**2.*R**2;

heatIndex = (HI-32)*5/9;

# Temperature Plot
tx = np.hstack((t,t[::-1]))
tempy = np.hstack((maxTemp,minTemp[::-1]))

fig1 = go.Figure()

fig1.add_trace(go.Scatter(x=tx,
                          y=tempy,
                          fill='toself',
                          fillcolor='rgba(0,100,80,0.2)',
                          line=dict(color='rgba(255,255,255,0)'),
                          hoverinfo="skip",
                          showlegend=False,
                         ))
fig1.add_trace(go.Scatter(x=t,y=meanTemp, mode="lines", name="Mean Temp",line={'dash': 'solid', 'color': 'midnightblue'}))
fig1.add_trace(go.Scatter(x=t,y=heatIndex, mode="lines", name="Heat Index",line={'dash': 'solid', 'color': 'orangered'}))



yTitle = 'Temperature (&#176;C)'
fig1.update_layout(title_text = 'Temperature',
                xaxis_title='Date',
                yaxis_title=yTitle,
                width = 740, height=480,
                margin=dict(r=20, b=10, l=10, t=30),
                showlegend = True,
                template = 'plotly_white'
                )
fig1.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01,
    bgcolor = 'rgba(255,255,255,0.8)'
))

# Precipitation Plot
fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(go.Scatter(x=t,y=precip, mode="lines", name="Precipitation",line={'dash': 'solid', 'color': 'dodgerblue'}),
               secondary_y=False)
fig2.add_trace(go.Scatter(x=t,y=relHum, mode="lines", name="Rel Humidity",line={'dash': 'solid', 'color': 'lightseagreen'}),
               secondary_y=True)

fig2.update_layout(title_text = 'Precipitation & Relative Humidity',
                xaxis_title='$Date$',
                width = 740, height=480,
                margin=dict(r=20, b=10, l=10, t=30),
                showlegend = True,
                template = 'plotly_white'
                )
fig2.update_yaxes(title_text="Precipitation (mm)", 
                  range = [0,100],
                  secondary_y=False)
fig2.update_yaxes(title_text="Relative Humidity (%)", 
                  range = [0,100],
                  secondary_y=True)

fig2.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01,
    bgcolor = 'rgba(255,255,255,0.8)'
))

# Doppler Radar Plot
url = 'https://mausam.imd.gov.in/Radar/caz_mum.gif'
img1 = io.imread(url)
# print(np.shape(img1))
url = 'https://mausam.imd.gov.in/Satellite/3Dasiasec_wv.jpg'
img2 = io.imread(url)

# Crop

img1 = img1[302:900,102:700,0:3]

s = np.shape(img2)
# print(s)
y1=int(np.ceil(s[1]/2.409));
y2=int(np.ceil(s[1]/2.023));
x1=int(np.ceil(s[0]/2.09));
x2=int(np.ceil(s[0]/1.82));

# print(x1,x2,y1,y2)

img2 = img2[x1:x2,y1:y2,:]
img2_gray = rgb2gray(img2)
img2_gray = resize(img2_gray,(np.shape(img1)[0],np.shape(img1)[1]),anti_aliasing=True)

# print(np.shape(img2_gray))
m = np.min(img2_gray[:])
normA = img2_gray - m
# normA = np.float(normA);
normA = normA / 255;

fig3, ax = plt.subplots(figsize=[15,15])
ax.set(xticks=[], yticks=[], title="Mumbai Doppler radar Image Overlayed with Satellite Image")
plt.imshow(img1)
plt.imshow(img2_gray, cmap='gray', alpha=img2_gray)


# Streamlit

st.plotly_chart(fig1)

st.plotly_chart(fig2)

st.pyplot(fig3)
