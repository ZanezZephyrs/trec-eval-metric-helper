from collections import defaultdict
from dataclasses import dataclass
from typing import List
import streamlit as st

import numpy as np
import pandas as pd
import os
from pathlib import Path
import re
from constants import available_metrics


@dataclass(init=True)
class FileInfo: 
    filename:str
    filepath:Path


METRIC_SET_DIR="data"


def list_files_in_dir(path):
    _,_,files=next(os.walk(path))
    
    files_info=[
        FileInfo(filename=file.split(".")[0],filepath=Path(path,file))
        for file in files
    ]
    
    return files_info
    


def build_metric_set_data(desired_metrics,selected_files:List[FileInfo]):
    data=defaultdict(lambda: {})
    for file in selected_files:
        with open(file.filepath,"r") as fin:
            for line in fin:
                metric,_,value=re.split('\s+', line.strip())
                if metric in desired_metrics:
                    data[file.filename][metric]=float(value)
    return data

def get_metric_plot(data,metric):
    values=[]
    labels=[]
    for source_file in data:
        labels.append(source_file)
        values.append(data[source_file][metric])

    
    df= pd.DataFrame(
     np.array(values).reshape((-1,1)),
     columns=[metric],
     index=labels
     )

    print(df)
    return df

def create_all_data_dataframe(data,desired_metrics):
    values=[]
    labels=[]
    for source_file in data:
        labels.append(source_file)
        line_data=[]
        for metric in desired_metrics:
            line_data.append(data[source_file][metric])
        values.append(line_data)
    
    df= pd.DataFrame(
     values,
     columns=desired_metrics,
     index=labels
     )
    return df


def get_colormap_data(data_dict,desired_metrics, relative=True):
    values=[]
    labels=[source for source in data_dict]        

    for metric in desired_metrics:
        line_data=np.array([data_dict[label][metric] for label in labels])
        
        if relative:
            line_data=line_data / max(line_data)

        values.append(line_data)
        
    
    df= pd.DataFrame(
     values,
     columns=labels,
     index=desired_metrics
     )
    return df.style.highlight_max(axis=1,color="green")


metric_set_files=list_files_in_dir(METRIC_SET_DIR)

# metric_set_names=[fileinfo.filename for fileinfo in metric_set_files]

selected_metric_sets=st.multiselect(
    "Select the metrics that should be considered",
    metric_set_files,
    [],
    format_func=lambda x: x.filename
)

at_least_one_metric_set_selected=len(selected_metric_sets)>0

desired_metrics = st.sidebar.multiselect(
     'Select the metrics you wish to inspect',
     available_metrics,
     ['recip_rank',"ndcg_cut_5","recall_5","P_5"]
     )

metric_set_data=build_metric_set_data(desired_metrics,selected_metric_sets)


st.markdown("# Trec eval Results helper")
st.write("Put all your trec eval results in the data folder and select the desired metric to inspect in the sidebar.\n")


with st.expander("Table view of the metrics"):

    st.markdown("## Table view of the metrics")

    if at_least_one_metric_set_selected:
        st.dataframe(create_all_data_dataframe(metric_set_data,desired_metrics))
    else:
        st.markdown("Please, select at least one metric set")


with st.expander("Compare metrics sets"):

    st.markdown("## Compare metrics sets")

    colormap_scale = st.radio(
        "Please select the scale for the results of the comparison",
        ('Absolute scale', 'Relative scale'))
    
    if at_least_one_metric_set_selected:
        st.table(get_colormap_data(metric_set_data, desired_metrics,relative=(colormap_scale=="Relative scale")))
    else:
        st.markdown("Please, select at least one metric set")



with st.expander("Plot an specific metric"):
    st.markdown("## Plot an specific metric")

    bar_chart_selected_metric = st.selectbox(
        'Select the metric to plot',
        desired_metrics)

    if at_least_one_metric_set_selected:
        st.bar_chart(get_metric_plot(metric_set_data,bar_chart_selected_metric))
    else:
        st.markdown("Please, select at least one metric set")

