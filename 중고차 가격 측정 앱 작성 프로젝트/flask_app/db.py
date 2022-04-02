import pandas as pd
import sqlite3

df = pd.read_csv('tfile.csv', encoding='cp949')

con = sqlite3.connect("df.db")

df.to_sql("df", con, if_exists="append", index=False)
con.commit()