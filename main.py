#%%
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly._subplots import make_subplots
import pymysql
from sqlalchemy import create_engine
from dbmodule import uloaddb, dloaddb
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from keras.layers import Dense, LSTM, Dropout
from keras.models import Model, Sequential
# %%
host = 'localhost'
id = 'root'
pw = 276018
dbname = 'cocacola'
tbname = 'cocacoladb'    
# %%
df = dloaddb(host, id, pw, dbname, tbname)
df.head()
# %%
# 시계열을 할때 date 컬럼은 datetime으로 타입 변경
df['Date']=pd.to_datetime(df['Date'])
# %%
df.info()
# %%
# 컬럼명 소문자로 변환
cols = df.columns
df.columns = cols.str.lower()
# %%
# x축에 시간을 넣기 위해 인덱스 고정
df.set_index('date', inplace=True)
# %%
df.plot()
# %%
df.plot(subplots=True)
# %%
fig = make_subplots(rows=4, cols=1, subplot_titles=("open", "high", "low", "close"))
fig.add_trace(go.Line(x=df.index, y=df["open"], name="open"), row=1, col=1)
fig.add_trace(go.Line(x=df.index, y=df["high"], name="high"), row=2, col=1)
fig.add_trace(go.Line(x=df.index, y=df["low"], name="low"), row=3, col=1)
fig.add_trace(go.Line(x=df.index, y=df["close"], name="close"), row=4, col=1)
fig.update_layout(title=dict(text="CoCaCola Stock"))
fig.show()
# %%
fig = go.Figure(data = [go.Candlestick(x=df.iloc[:30,:].index, open=df['open'], high=df['high'], low= df['low'], close=df['close'])])
fig.show()
# %%
# px.line(df['close'], title= 'CocaCola Close stock') # plotly 기본 문법

# fig = go.Figure() # plotly 그래프 여러개 그릴 때 문법
# fig.show()

fig = go.Figure()
fig.add_trace(go.Line(x=df.index, y=df['close'], name='close'))
fig.update_layout(title=dict(text='CocaCola Close Stock'))
fig.show()
# %%
# 추이를 알기 위해서 이동평균법을 사용
# 300, 500, 700, 900일 평균인 그래프 그려보기
# rolling 함수
df = df.drop(['open','high','low', 'volume', 'dividends','stock splits'], axis = 1)
df
# %%
df['ma300days'] = df['close'].rolling(300).mean()
df['ma500days'] = df['close'].rolling(500).mean()
df['ma700days'] = df['close'].rolling(700).mean()
df['ma900days'] = df['close'].rolling(900).mean()
# %%
print(df.isna().sum()) # ex) 300일의 평균 > 299일이 지나고 300일째부터 값 생성
#%%
# 추이만 확인하는 것이기 때문에 결측값은 상관이 없다
fig = go.Figure()
fig.add_trace(go.Line(x=df.index, y=df['close'], name='close'))
fig.add_trace(go.Line(x=df.index, y=df['ma300days'], name='ma300days'))
fig.add_trace(go.Line(x=df.index, y=df['ma500days'], name='ma500days'))
fig.add_trace(go.Line(x=df.index, y=df['ma700days'], name='ma700days'))
fig.add_trace(go.Line(x=df.index, y=df['ma900days'], name='ma900days'))
fig.update_layout(title=dict(text='Close & MA Graph'))
fig.show()
# %%
# 1일 수익률 체크
df['percent_change'] = df['close'].pct_change()
df.head(30)
#%%
# 히스토그램 그릴 때는 x값만 필요(수익률 분포 확인)
fig = go.Figure()
fig.add_trace(go.Histogram(x = df['percent_change'], name = 'percent change'))
fig.update_layout(title=dict(text='percent change distribution'))
fig.show()
# %%
fig = go.Figure()
fig.add_trace(go.Line(x=df.index, y=df['percent_change'], name = 'percent change'))
fig.update_layout(title=dict(text='percent change'))
fig.show()
# %%
# 리스크 = 퍼센테이지 구한 것에 표준편차
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.array(np.mean(df['percent_change'])), y=np.array(np.std(df['percent_change']))))
fig.update_layout(title=dict(text='Risk'))
fig.show()
# %%
# 종가만 있는 데이터프레임 가져오기
df1 = df.drop(['ma300days', 'ma500days', 'ma700days', 'ma900days','percent_change'], axis=1)
# %%
# 종가를 스케일링(0과 1사이로 맞추기 위함)
mmscale = MinMaxScaler()
df1['close'] = mmscale.fit_transform(df1[['close']])
df1
# %%
# 데이터 분리(전체 데이터의 85%를 학습 데이터로 지정)
train_data = df1.iloc[:int(np.round(df1.shape[0] * 0.85)),:]
test_data = df1.iloc[int(np.round(df1.shape[0]*0.85)-60):,:]
print(f'train data shape: {train_data.shape}')
print(f'test data shape: {test_data.shape}')
# %%
# LSTM 특성상 전 데이터가 현 데이터에 영향을 끼치기 때문에
# 통상적인 주가 결산 기준인 60일을 기준으로 데이터를 맞물리게 분리하여야함(ex) 0~59 / 1~60 ...
def maketrainArray(data):
    array_list=[]
    y_array=[]
    for i in range(int(len(data))):
        if i+60 < int(len(data)):
            array60 = data.iloc[i:i+60,0]
            array_y = data.iloc[i+60,0]
            array_list.append(array60)
            y_array.append(array_y)
        else:
            break
    return array_list, y_array

def maketestArray(data):
    array_list=[]
    for i in range(int(len(data))):
        if i+60 < int(len(data)):
            array60 = data.iloc[i:i+60,0]
            array_list.append(array60)
        else:
            break
    return array_list
# %%
array_list, y_array = maketrainArray(train_data)
X_train = np.array(array_list)
y_train = np.array(y_array)
print(X_train.shape)
print(y_train.shape)
# %%
array_list1 = maketestArray(test_data)
X_test = np.array(array_list1)
y_test = np.array(df1.iloc[int(np.round(df1.shape[0]*0.85)):,:])
print(X_test.shape)
print(y_test.shape)
#%%
# input_shape에 모양을 맞춰줌
X_train = X_train.reshape(12794,60,1)
# %%
# 모델 정의 (sequantial형)
model = Sequential()
model.add(LSTM(256, input_shape=(60,1), return_sequences=True)) #노드 수는 2의 제곱으로 적는다
# input_shape의 첫번째 인자는 batchsize(최적의 값을 모르기 때문에 비워 놓음), 두번째 인자는 시점(과거의 몇개의 데이터로 한개를 추론하냐), 세번째 인자는 feature 수
# LSTM 층을 여러 개 쌓을 때 return_sequences=True를 설정해주어야 전체 값을 다음 층에 받을 수 있음
model.add(LSTM(128, activation='relu', return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1)) # 마지막에 값 한 개를 맞추고 싶기 때문에 마지막 덴스층은 1로 한다. (10개 값을 맞추고 싶으면 10을 넣는다)

model.compile(optimizer='Adam', loss='mean_squared_error')
model.summary()
# %%
# 모델 학습
history = model.fit(X_train, y_train, epochs=10, verbose=1)
# verbose는 진행 상황을 보여주는 파라미터
#%%
history.history
# %%
# loss값으로 그래프 그려보기
epoch_list = list(range(0,30))
fig = go.Figure()
fig.add_trace(go.Line(x = epoch_list, y = history.history['loss']))
fig.show()
# %%
# 예측값과 실제값 비교
pred = model.predict(X_test)
invPred = mmscale.inverse_transform(pred)
invY_test = mmscale.inverse_transform(y_test)
print(invPred)
print(invY_test)
# %%
final_df = pd.DataFrame({'close': invY_test.reshape(2268,), 'pred': invPred.reshape(2268,)})
final_df = final_df.set_index(df1.index[-2268:])
print(mean_squared_error(invY_test, invPred))
# %%
fig = go.Figure()
fig.add_trace(go.Line(x=final_df.index, y= final_df['close'], name='Actual'))
fig.add_trace(go.Line(x=final_df.index, y= final_df['pred'], name='predict'))
fig.update_layout(title = dict(text='Compare actual and predicted values'))
fig.show()
# %%
