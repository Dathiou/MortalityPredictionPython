import time
import numpy
import pandas as pd
import csv
import utils
from numpy import average
from pandas.stats.tests.common import DATE_RANGE
from pandas.tseries.index import date_range
from datetime import timedelta


def read_csv(filepath):
    '''
    Read the events.csv and mortality_events.csv files. Variables returned from this function are passed as input to the metric functions.
    This function needs to be completed.
    '''
    events,mortality,_=utils.read_csv(filepath)
    #events = pd.read_csv(open(filepath +'/events.csv'))
    

    #mortality = pd.read_csv(open(filepath +'/mortality_events.csv'))
    
    return events, mortality

def event_count_metrics(events, mortality):
    '''
    
    Event count is defined as the number of events recorded for a given patient.
    This function needs to be completed.
    '''
    
    A=pd.merge(events,mortality, how='outer',on='patient_id') 
    list_dead=A[A.label==1]
    list_alive=A[A.label<>1]
    #print list_alive 
    event_alive=list_alive.groupby('patient_id')['event_id'].count()
    #print event_alive
    event_dead=list_dead.groupby('patient_id')['event_id'].count()
    #print event_dead
    
    
    avg_dead_event_count = event_dead.mean()

    max_dead_event_count = event_dead.max()

    min_dead_event_count = event_dead.min()

    avg_alive_event_count = event_alive.mean()

    max_alive_event_count = event_alive.max()

    min_alive_event_count = event_alive.min()

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    This function needs to be completed.
    '''
    A=pd.merge(events,mortality, how='outer',on='patient_id') 
    list_dead=A[A.label==1]
    list_alive=A[A.label<>1]
    
    unique_date_alive=list_alive.groupby('patient_id')['timestamp_x'].nunique()
   
    unique_date_dead=list_dead.groupby('patient_id')['timestamp_x'].nunique()
  
    
    avg_dead_encounter_count = unique_date_dead.mean()

    max_dead_encounter_count = unique_date_dead.max()

    min_dead_encounter_count = unique_date_dead.min()

    avg_alive_encounter_count = unique_date_alive.mean()

    max_alive_encounter_count = unique_date_alive.max()

    min_alive_encounter_count = unique_date_alive.min()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    Record length is the duration between the first event and the last event for a given patient. 
    This function needs to be completed.
    '''
    
    A=pd.merge(events,mortality, how='outer',on='patient_id') 
    list_dead=A[A.label==1]
    list_alive=A[A.label<>1]
    
    length_alive=list_alive.groupby('patient_id').agg({'timestamp_x' : lambda x: utils.date_convert(max(x)) - utils.date_convert(min(x))+timedelta(days=1)})
    #length_alive=list_alive.groupby('patient_id').agg({'timestamp_x' : lambda x: date_range(max(x),min(x),freq='D')})
    #length_alive=list_alive.groupby('patient_id').agg({'timestamp_x' : lambda x: timedelta(days= utils.date_convert(max(x)) - utils.date_convert(min(x)))})
    #print length_alive
    length_dead=list_dead.groupby('patient_id').agg({'timestamp_x' : lambda x: utils.date_convert(max(x)) - utils.date_convert(min(x))+timedelta(days=1)})
   # print length_dead
    
    #deltas = [x-dt_min for x in series]
    avg_dead_rec_len = numpy.mean(length_dead.timestamp_x).days

    max_dead_rec_len = numpy.max(length_dead.timestamp_x).days

    min_dead_rec_len = numpy.min(length_dead.timestamp_x).days

    avg_alive_rec_len = numpy.mean(length_alive.timestamp_x).days

    max_alive_rec_len = numpy.max(length_alive.timestamp_x).days

    min_alive_rec_len = numpy.min(length_alive.timestamp_x).days

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    DONOT MODIFY THIS FUNCTION. 
    Just update the train_path variable to point to your train data directory.
    '''
    #Modify the filepath to point to the CSV files in train_data
    train_path = '../data/train/'
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute event count metrics: " + str(end_time - start_time) + "s")
    print event_count

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute encounter count metrics: " + str(end_time - start_time) + "s")
    print encounter_count

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute record length metrics: " + str(end_time - start_time) + "s")
    print record_length
    
if __name__ == "__main__":
    main()


