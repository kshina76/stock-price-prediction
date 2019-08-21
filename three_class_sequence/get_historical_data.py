from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
import pandas as pd

accountID = "101-009-11764804-001"
access_token = "2826adb0783333f6150483c0aa5f1a59-79127121efa78112f7578530295a5e38"
client = API(access_token=access_token)

def cnv(r, h):
    for candle in r.get('candles'):
        ctime = candle.get('time')[0:19]
        try:
            rec = "{time},{complete},{o},{h},{l},{v},{c}".format(
                time=ctime,
                complete=candle['complete'],
                o=candle['mid']['o'],
                h=candle['mid']['h'],
                l=candle['mid']['l'],
                v=candle['volume'],
                c=candle['mid']['c'],
            )
        except Exception as e:
            print(e, r)
        else:
            h.write(rec+"\n")

_from = '2005-05-01T00:00:00Z' 
_to = '2019-07-17T00:00:00Z'
gran = 'D'
instr = 'USD_CAD'  

params = {
    "granularity": gran,
    "from": _from,
    "to": _to
}

with open("./{}.{}.csv".format(instr, gran), "w") as O:
    for r in InstrumentsCandlesFactory(instrument=instr, params=params):
        print("REQUEST: {} {} {}".format(r, r.__class__.__name__, r.params))
        rv = client.request(r)
        cnv(r.response, O)

df = pd.read_csv("./USD_CAD.D.csv" ,header=None,usecols=[0,2,3,4,5,6])
df.columns = ['time','open', 'high', 'low', 'volume','close']
#df.index = 'time'
df = df.set_index('time')
df.head()
