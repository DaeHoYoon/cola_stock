# %%
import pymysql
from sqlalchemy import create_engine
import pandas as pd


def uloaddb(df, id, pw, host, dbname, tbname):
    dbpath = f'mysql+pymysql://{id}:{pw}@{host}/{dbname}'
    conn = create_engine(dbpath)
    df.to_sql(name=f'{tbname}', con=conn, if_exists='fail', index=False)

    return df

def dloaddb(host, id, pw, dbname, tbname):
    conn = pymysql.connect(host=host, user=id, passwd=str(pw), database=dbname,charset='utf8')
    df = pd.read_sql(f'select * from {tbname}', con=conn)
    
    return df