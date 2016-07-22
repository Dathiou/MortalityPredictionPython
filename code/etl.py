import utils
import time
import pandas as pd
from datetime import timedelta
from datetime import date
import numpy as np
import math
import sklearn

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = ''
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = ''

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = ''
    
    events, mortality, feature_map=utils.read_csv(filepath)
    

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.
    
    

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    A=pd.merge(events,mortality, how='outer',on='patient_id') 
    list_dead=A[A.label==1]
    list_alive=A[A.label<>1]
    #print list_alive
    #print list_alive 
    event_alive=list_alive.groupby('patient_id')['timestamp_x'].apply(lambda x: utils.date_convert(x.max()).date())
    #print event_alive
    event_dead=list_dead.groupby('patient_id')['timestamp_y'].apply(lambda x: utils.date_offset(x.max(),-30).date())
    #print event_dead
    
    df_alive=pd.DataFrame({'patient_id':event_alive.index, 'indx_date':event_alive.values})
    df_dead=pd.DataFrame({'patient_id':event_dead.index, 'indx_date':event_dead.values})
    indx_date = pd.concat([df_alive,df_dead])
    
    indx_date.sort(['patient_id','indx_date'], ascending=[True,True])
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)
    #print indx_date
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    
    #indx_date_day=indx_date['indx_date'].apply(lambda x: x.days)
    A=pd.merge(events,indx_date, how='outer',on='patient_id') 
    
    #for line in A:
        
    #[x for x in X if P(x)]
    #list=A[A.indx_date for x in X if utils.date_convert(x)>A.timestamp>utils.date_convert(A.indx_date)-timedelta(days=2000)]
   
    #filtered_events = pd.DataFrame()

    #C=[]
    #for r in zip(A['patient_id'], A['timestamp'], A['indx_date'],A['event_id'],A['value']):
    #    if r[2]-timedelta(days=2000)<utils.date_convert(r[1]).date()<r[2]:
    #        filtered_events = filtered_events.append({'patient_id': r[0], 'event_id': r[3], 'value': r[4]},ignore_index=True)
            #C.append((r[0], r[2]))
            
   
    A.timestamp=A.timestamp.apply(lambda x: utils.date_convert(x).date())
    filtered_events=A[(A['indx_date'] > A['timestamp']) & (A['timestamp'] > A['indx_date']-timedelta(days=2000))]
    
    
    #A["filter"] = A.apply(lambda row: 1 if row['indx_date'] > row['timestamp'] > row['indx_date']-timedelta(days=2000) else np.nan,axis=1)
    
    #A['timestamp']=A.apply(lambda row: utils.date_convert(row['timestamp']).date(),axis=1)
    #A["filter"] = A.apply(lambda row: row['indx_date'] if row['indx_date'] > row['timestamp'] > row['indx_date']-timedelta(days=2000) else np.nan,axis=1)
  
   
    filtered_events.sort(['patient_id'], ascending=[True])
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and mean to calculate feature value 
    4. Normalize the values obtained above using min-max normalization
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    filtered_events_df1=filtered_events_df[pd.notnull(filtered_events_df['value'])]
    
    events_lab1=filtered_events_df1[filtered_events_df1.event_id.str.startswith("L")]
    events_diagmed1=filtered_events_df1[~filtered_events_df1.event_id.str.startswith("L")]
    
    events_diagmed=events_diagmed1.groupby(['event_id','patient_id'])['value'].sum()
    events_lab=events_lab1.groupby(['event_id','patient_id'])['value'].count()
    
    #df_diagmed=pd.DataFrame({'event_id':events_diagmed.index, 'value':events_diagmed.values})
    #df_lab=pd.DataFrame({'event_id':events_lab.index, 'value':events_lab.values})
    computed = pd.concat([pd.DataFrame(events_diagmed),pd.DataFrame(events_lab)])
    feature_map_df=feature_map_df.rename(columns = {'idx':'feature_id'})
    #computed=computed.rename(columns = {'value':'feature_value'})
    computed['event_id'] = computed.index.get_level_values('event_id') 
    computed['patient_id'] = computed.index.get_level_values('patient_id')
    aggregated_events=pd.merge(computed,feature_map_df, how='inner',on='event_id')
    '''
    aggregated_events_un.value_normalized=
    for feature_id in aggregated_events_un.feature_id.nunique():
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    minmax = [(x_i - min(x)) / (min(x) - max(x)) for x_i in x]os = [0 for i in range(len(x))]
    '''
    aggregated_events['feature_value']=aggregated_events.groupby('feature_id')['value'].transform(lambda x: x / x.max())

    
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)
    
    
    
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    
    
    patient_features = aggregated_events.groupby('patient_id')[['feature_id','feature_value']].apply(lambda x: sorted([tuple(x) for x in x.values],key=lambda x: x[0])).to_dict()
    mortality=pd.merge(aggregated_events,mortality, how='left',on='patient_id') 
    mortality.loc[mortality.label<>1,'label']=0
    mortality = mortality.set_index('patient_id')['label'].to_dict()

    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed
    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    
    patient_features_sort=sorted(patient_features.items())
    
    deliverable1 = open(op_file, 'w')    
    deliverable2 = open(op_deliverable, 'w')
    for x in patient_features_sort: 
        label = '%i'%mortality[x[0]] 
        pairs = ['%i:%f'%(x[1][i]) for i in xrange(len(x[1]))] 
        sep_line1 = [label] 
        sep_line1.extend(pairs) 
        sep_line1.append(' \n') 
        line1 = ' '.join(sep_line1) 
        deliverable1.write(line1)
        
        patient = '%i'%x[0]
        sep_line2=['%i'%x[0]]
        sep_line2.extend([label])
        sep_line2.extend(pairs)
        sep_line2.append(' \n')
        line2 = ' '.join(sep_line2)
        deliverable2.write(line2)
        
    deliverable1.close()
    deliverable2.close()
    
    '''
    deliverable2 = open(op_deliverable, 'w')
    for x in patient_features_sort: 
        patient = '%i'%x[0]
        label = "{0:0.1f}".format(mortality[x[0]])
        pairs = ['%i:%f'%(x[1][i]) for i in xrange(len(x[1]))] 
        sep_line=['%i'%x[0]]
        sep_line.extend([label])
        sep_line.extend(pairs)
        sep_line.append(' \n')
        line = ' '.join(sep_line)
    
        deliverable2.write(line)
    deliverable2.close()
    '''

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', 'D:/Damien/Documents/GT/Spring 2016/CSE8803 Big Data Analytics For Healthcare/HM1/deliverables/features.train')

if __name__ == "__main__":
    main()




