#!/usr/bin/env python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
import numpy as np
import re
import pandas as pd
from datetime import datetime, timedelta


# http://www.imdmumbai.gov.in/scripts/cur_met_data_print.asp
url = "http://www.imdmumbai.gov.in/"
r = requests.get(url)


soup = BeautifulSoup(r.text,"html.parser")
data_table = soup.find('table',width='700px')

data = data_table.find_all('tr')
for i in range(len(data)):
    if data[i].find_all('td'):
        station = data[i].find_all('td')[0].text.strip()

        if station == 'Mumbai (CLB)':
            idx = i
            break

if idx:
    maxTemp = data[idx].find_all('td')[1].text.strip()
    minTemp = data[idx].find_all('td')[2].text.strip()
    relHum = data[idx].find_all('td')[3].text.strip()
    rainFall = data[idx].find_all('td')[4].text.strip()


if not maxTemp.replace('.', '', 1).isdigit():
    maxTemp = np.nan
if not minTemp.replace('.', '', 1).isdigit():
    minTemp = np.nan
if not relHum.replace('.', '', 1).isdigit():
    relHum = np.nan
if not rainFall.replace('.', '', 1).isdigit():
    rainFall = np.nan

    
str = data[1].text
x = re.search('\son\s',str)
start = x.end()
x = re.search('\dat',str)
end = x.start()+1
obsDate =str[start:end]
obsDate = datetime.strptime(obsDate,'%B %d,%Y')
obsDate = obsDate-timedelta(days=1)
obsDate = obsDate.strftime('%Y-%m-%d')


df = pd.read_csv('weatherdata.csv')

if obsDate not in df.values:
    dct = {'date':obsDate, 'maxTemp':maxTemp, 'minTemp':minTemp, 'relHum':relHum, 'rainFall':rainFall}
    df = df.append(dct, ignore_index=True)
    df.to_csv('weatherdata.csv')
