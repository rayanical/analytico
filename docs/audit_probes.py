"""Offline audit probes. Run from repo root with backend/venv/bin/python docs/audit_probes.py.
Uses synthetic data and mocked schema enrichment. Generated-code execution is no longer probed.
Outputs observations rather than asserting the current defects are desirable.
"""
import os, sys, json
from pathlib import Path
from unittest.mock import patch
os.environ['OPENAI_API_KEY'] = 'audit-placeholder'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import pandas as pd
from modules.data_janitor import clean_dataframe, legacy_normalize_header
from modules.aggregation import aggregate_data, smart_group_top_n, smart_resample_dates
from services.aggregation_service import run_aggregate
from storage import DatasetInfo, DATASETS, store_dataset
from models import AggregateRequest, FilterConfig
from utils.filtering import apply_filters
from fastapi.testclient import TestClient
from main import app

def emit(name, result):
    print('PROBE', name, json.dumps(result, default=str))
def store(df, types=None):
    ds=DatasetInfo(df,'fixture.csv',[],{},types or {c:'metric' for c in df.columns},{},{},None,[])
    store_dataset(ds)
    return ds

df=pd.DataFrame({'g':['A']*3+['B']*2+['C'], 'v':[10,20,30,40,50,60]})
g=smart_group_top_n(df,'g',['v'],'count',n=2)
actual,_=aggregate_data(g,'g',['v'],'count',limit=2)
emit('count_twice', {'intermediate':g.to_dict('records'),'actual':actual.to_dict('records')})
dates=pd.DataFrame({'date':pd.date_range('2024-01-01',periods=120),'v':1})
g,bucket=smart_resample_dates(dates,'date',['v'],'count')
r,_=aggregate_data(g,'date',['v'],'count')
emit('date_count', {'expected_total':120,'actual_total':int(r.v.sum()),'periods':len(r)})
rank=pd.DataFrame({'g':['A']*10+['B','C'],'v':[1]*10+[9,8]})
g=smart_group_top_n(rank,'g',['v'],'mean',n=2)
emit('mean_top_by_sum',g.to_dict('records'))
with patch('modules.data_janitor.llm_enrich_columns',side_effect=lambda headers,df:[{'original':h,'clean':legacy_normalize_header(h),'semantic_type':None} for h in headers]):
    for name,df in [('imputation',pd.DataFrame({'sensor_value':[1.1,2.4,3.8,None,5.2,6.1,None,8.0]})),('ratio',pd.DataFrame({'ratio':[0.2,0.8,None]})),('parse_loss',pd.DataFrame({'x':['1','2','bad']})),('locale',pd.DataFrame({'amount':['1.234,56','2.345,67']})),('identifier',pd.DataFrame({'id':['001','002',None]})),('duplicate_headers',pd.DataFrame([[1,2,3]],columns=['A!','A?','a_2']))]:
        raw=df.copy()
        try:
            cleaned,actions,missing,formats,types=clean_dataframe(df)
            emit(name,{'before':raw.to_dict('list'),'after':cleaned.to_dict('list'),'columns':cleaned.columns.tolist(),'actions':actions,'missing':missing})
        except Exception as e:
            emit(name,{'error':str(e),'columns_after':df.columns.tolist()})
store_df=pd.DataFrame({'g':['A','B','C'],'v':[1,2,100]})
ds=store(store_df)
r=run_aggregate(AggregateRequest(dataset_id=ds.id,x_axis_key='g',y_axis_keys=['v'],limit=2,group_others=False))
emit('topn_control',r.data)
try:
    run_aggregate(AggregateRequest(dataset_id=ds.id,x_axis_key='g',y_axis_keys=['v'],sort_by=None))
except Exception as e: emit('sort_none',str(e))
client=TestClient(app)
r=client.post('/drilldown',json={'dataset_id':ds.id,'filters':[{'column':'date_week','values':['one']}],'limit':50})
emit('missing_filter_column',{'status':r.status_code,'response':r.json()})
r=client.post('/drilldown',json={'dataset_id':ds.id,'limit':-1})
emit('negative_limit',{'status':r.status_code,'response':r.json()})
floatdf=pd.DataFrame({'g':['A','B'],'v':[1.0,2.0]})
f,_=apply_filters(floatdf,[FilterConfig(column='v',operator='eq',value=1)])
emit('float_equality',len(f))
nulls=pd.DataFrame({'g':['A',None,'B'],'v':[None,5.,2.]})
g,_=aggregate_data(nulls,'g',['v'],'sum')
emit('null_sum_group',g.to_dict('records'))
emit('execution_helper_present', (Path(__file__).resolve().parents[1] / 'backend/utils/execution.py').exists())
DATASETS.clear()
