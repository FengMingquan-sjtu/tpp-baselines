# this file should be excuted with python3
import pickle
import numpy as np

def convert(input_path, output_path):
    # input: {sample_ID:{pred_ID:{"time":[1,2,3], "state":[0,1,0]}},..,}
    # output: {'test1': [], 'args': None, 'dim_process': N_events, 'dev': [[{'time_since_start': 118.0, 'time_since_last_event': 118.0, 'type_event': 0}],[sample_2],..] }}
    #- train : Training the model
    #- dev : Tuning hyper parameters
    #- test : Final test and numbers in the paper

    input_dataset = np.load(input_path, allow_pickle='TRUE').item()
    N_events = len(input_dataset[0].keys())
    N_samples = len(input_dataset)
    train_ratio = 0.6
    dev_ratio = 0.2
    output_train = {'test1': [], 'args': None, 'dim_process': N_events, "train":[]}
    output_test = {'test1': [], 'args': None, 'dim_process': N_events, "test":[]}
    output_dev = {'test1': [], 'args': None, 'dim_process': N_events, "dev":[]}
    
    
    for ID, sample in input_dataset.items():
        new_sample = []
        event_list = []
        for pred_ID, event in sample.items():
            for time,state in zip(event['time'], event["state"]):
                if state == 1:
                    event_list.append((time, pred_ID))
        event_list.sort(key=lambda x:x[0])
        new_sample.append({"time_since_start":event_list[0][0], 'time_since_last_event':event_list[0][0], 'type_event': event_list[0][1]})
        for idx in range(len(event_list[1:])):
            new_sample.append({"time_since_start":event_list[idx+1][0], 'time_since_last_event':event_list[idx+1][0]-event_list[idx][0], 'type_event': event_list[idx+1][1]})
        
        if ID <= train_ratio * N_samples:
            output_train["train"].append(new_sample)
        elif ID <= (train_ratio + dev_ratio) * N_samples:
            output_dev["dev"].append(new_sample)
        else:
            output_test["test"].append(new_sample)
        
    with open(output_path+"train.pkl", "wb") as f:
        pickle.dump(output_train, f, protocol=2) #convert to python2 pickle
    with open(output_path+"test.pkl", "wb") as f:
        pickle.dump(output_test, f, protocol=2) #convert to python2 pickle
    with open(output_path+"dev.pkl", "wb") as f:
        pickle.dump(output_dev, f, protocol=2) #convert to python2 pickle

        
                

if __name__ == "__main__":
    #input_crime = "../data/crime_all_day_scaled.npy"
    #output_crime = "./data/crime/"
    #convert(input_crime, output_crime)

    input_mimic = "../data/mimic_3_clip_scaled.npy"
    output_mimic = "./data/mimic_clip/"
    convert(input_mimic, output_mimic)
    
    #input_mimic = "../data/mimic_3_scaled.npy"
    #output_mimic = "./data/mimic/"
    #convert(input_mimic, output_mimic)

    