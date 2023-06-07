import os
import torch
import tempfile
import json
import random
from dosed.utils import Compose
from dosed.datasets import BalancedEventDataset as dataset
from dosed.models import DOSED3 as model
from dosed.datasets import get_train_validation_test
from dosed.trainers import trainers
from dosed.preprocessing import GaussianNoise, RescaleNormal, Invert
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple



def main():
    seed = 2019
    #h5_directory = 'D:/dd2'
    #h5_directory = 'D:/dosed'
    #h5_directory = "C:/Users/Nullerh/Documents/DTU_SCHOOL_WORK/dosed_no_change/data/h5"
    h5_directory = '/scratch/s194277/mros/h5_full/'
    train, validation, test = get_train_validation_test(h5_directory,
                                                        percent_test=10,
                                                        percent_validation=20,
                                                        seed=seed)

    print("Number of records train:", len(train))
    print("Number of records validation:", len(validation))
    print("Number of records test:", len(test))
    batch_size = 128
    window = 120  # window duration in seconds
    ratio_positive = 1. # When creating the batch, sample containing at least one spindle will be drawn with that probability
    torch.cuda.empty_cache()
    fs = 128

    signals = [
        {
            'h5_path': 'c3',
            'fs': fs,
            'processing': {
                "type": "clip_and_normalize",
                "args": {
                    "min_value": -150,
                    "max_value": 150,
                }
            }
        },
        {
            'h5_path': 'c4',
            'fs': fs,
            'processing': {
                "type": "clip_and_normalize",
                "args": {
                    "min_value": -150,
                    "max_value": 150,
                }
            }
        },
        {
            'h5_path': 'eogl',
            'fs': fs,
            'processing': {
                "type": "clip_and_normalize",
                "args": {
                    "min_value": -150,
                    "max_value": 150,
                }
            }
        },
        {
            'h5_path': 'eogr',
            'fs': fs,
            'processing': {
                "type": "clip_and_normalize",
                "args": {
                    "min_value": -150,
                    "max_value": 150,
                }
            }
        },
        {
            'h5_path': 'chin',
            'fs': fs,
            'processing': {
                "type": "clip_and_normalize",
                "args": {
                    "min_value": -150,
                    "max_value": 150,
                }
            }
        },
        {
            'h5_path': 'legl',
            'fs': fs,
            'processing': {
                "type": "clip_and_normalize",
                "args": {
                    "min_value": -150,
                    "max_value": 150,
                }
            }
        },
        {
            'h5_path': 'legr',
            'fs': fs,
            'processing': {
                "type": "clip_and_normalize",
                "args": {
                    "min_value": -150,
                    "max_value": 150,
                }
            }
        },
        {
            'h5_path': 'nasal',
            'fs': fs,
            'processing': {
                "type": "clip_and_normalize",
                "args": {
                    "min_value": -150,
                    "max_value": 150,
                }
            }
        },
        {
            'h5_path': 'thor',
            'fs': fs,
            'processing': {
                "type": "clip_and_normalize",
                "args": {
                    "min_value": -150,
                    "max_value": 150,
                }
            }
        },
        {
            'h5_path': 'abdo',
            'fs': fs,
            'processing': {
                "type": "clip_and_normalize",
                "args": {
                    "min_value": -150,
                    "max_value": 150,
                }
            }
        },
    ]
    events = [
        {
            "name": "Arousal",
            "h5_path": "ar",
        },
        {
            "name": "Leg Movement",
            "h5_path": "lm",
        },
        {
            "name": "Sleep-disordered breathing",
            "h5_path": "sdb",
        },
    ]

    if h5_directory == "C:/Users/Nullerh/Documents/DTU_SCHOOL_WORK/dosed_no_change/data/h5":
        signals = [
            {
                'h5_path': '/eeg_0',
                'fs': 128,
                'processing': {
                    "type": "clip_and_normalize",
                    "args": {
                        "min_value": -150,
                        "max_value": 150,
                    }
                }
            },
            {
                'h5_path': '/eeg_1',
                'fs': 128,
                'processing': {
                    "type": "clip_and_normalize",
                    "args": {
                        "min_value": -150,
                        "max_value": 150,
                    }
                }
            }
        ]

        events = [
            {
                "name": "spindle",
                "h5_path": "spindle",
            },
        ]


    dataset_parameters = {
        "h5_directory": h5_directory,
        "signals": signals,
        "events": events,
        "window": window,
        "fs": fs,
        "ratio_positive": ratio_positive,
        "n_jobs": -1,  # Make use of parallel computing to extract and normalize signals from h5
        "cache_data": True,
        # by default will store normalized signals extracted from h5 in h5_directory + "/.cache" directory
    }

    dataset_validation = dataset(records=validation, **dataset_parameters)
    dataset_test = dataset(records=test, **dataset_parameters)

    # for training add data augmentation
    dataset_parameters_train = {
        "transformations": Compose([
            GaussianNoise(),
            RescaleNormal(),
            Invert(),
        ])
    }
    dataset_parameters_train.update(dataset_parameters)
    dataset_train = dataset(records=train, **dataset_parameters_train)

    default_event_sizes = [3, 10, 25]
    k_max = 8
    kernel_size = 3
    probability_dropout = 0.1
    device = torch.device("cuda")

    sampling_frequency = dataset_train.fs

    net_parameters = {
        "detection_parameters": {
            "overlap_non_maximum_suppression": 0.5,
            "classification_threshold": 0.7
        },
        "default_event_sizes": [
            default_event_size * sampling_frequency
            for default_event_size in default_event_sizes
        ],
        "k_max": k_max,
        "kernel_size": kernel_size,
        "pdrop": probability_dropout,
        "fs": 128,  # just used to print architecture info with right time
        "input_shape": dataset_train.input_shape,
        "number_of_classes": dataset_train.number_of_classes,
    }
    net = model(**net_parameters)
    net = net.to(device)

    optimizer_parameters = {
        "lr": 1e-4,
        "weight_decay": 1e-4,
    }
    loss_specs = {
        "type": "focal",
        "parameters": {
            "number_of_classes": len(events),
            "alpha": 0.25,
            "gamma": 2,
            "device": torch.device("cuda"),
        }
    }
    epochs = 100

    trainer = trainers["adam"](
        net,
        optimizer_parameters=optimizer_parameters,
        loss_specs=loss_specs,
        epochs=epochs,
        logger_parameters={
            "num_events": 3,
            "output_dir": os.getcwd(),
            "output_fname": 'train_history.json',
            "metrics": ["f1", "precision", "recall"],
            "name_events": ["AR","LM","SDB"]
        },
        metrics=["f1", "precision", "recall"],
        metric_to_maximize="f1",
        matching_overlap=0.5,
    )

    best_net_train, best_metrics_train, best_threshold_train = trainer.train(
        dataset_train,
        dataset_validation,
        batch_size=batch_size
    )

    test_predictions = best_net_train.predict_dataset(
        dataset_test,
        best_threshold_train,
    )

    record = dataset_test.records[1]

    index_spindle = 30
    window_duration = 5

    # retrive spindle at the right index
    spindle_start = float(test_predictions[record][0][index_spindle][0]) / sampling_frequency
    spindle_end = float(test_predictions[record][0][index_spindle][1]) / sampling_frequency

    # center data window on annotated spindle
    start_window = spindle_start + (spindle_end - spindle_start) / 2 - window_duration
    stop_window = spindle_start + (spindle_end - spindle_start) / 2 + window_duration

    # Retrieve EEG data at right index
    index_start = int(start_window * sampling_frequency)
    index_stop = int(stop_window * sampling_frequency)
    y = dataset_test.signals[record]["data"][0][index_start:index_stop]

    # Build corresponding time support
    t = start_window + np.cumsum(np.ones(index_stop - index_start) * 1 / sampling_frequency)

    plt.figure(figsize=(16, 5))
    plt.plot(t, y)
    plt.axvline(spindle_end)
    plt.axvline(spindle_start)
    plt.ylim([-1, 1])
    plt.show()


if __name__ == '__main__':
    main()