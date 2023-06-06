""" Trainer class basic with SGD optimizer """

import copy
import os

import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ..datasets import collate
from ..functions import (loss_functions, available_score_functions, compute_metrics_dataset)
from ..utils import (match_events_localization_to_default_localizations, Logger)
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class TrainerBase:
    """Trainer class basic """

    def __init__(
            self,
            net,
            optimizer_parameters={
                "lr": 0.001,
                "weight_decay": 1e-8,
            },
            loss_specs={
                "type": "focal",
                "parameters": {
                    "number_of_classes": 3,
                    "alpha": 0.25,
                    "gamma": 2,
                    "device": torch.device("cuda"),
                }
            },
            metrics=["precision", "recall", "f1"],
            epochs=100,
            metric_to_maximize="f1",
            patience=None,
            save_folder=None,
            logger_parameters={
                "num_events": 3,
                "output_dir": os.getcwd(),
                "output_fname": 'train_history.json',
                "metrics": ["precision", "recall", "f1"],
                "name_events": ["arousal", "leg movement", "Sleep-disordered breathing"]
            },
            threshold_space={
                "upper_bound": 0.85,
                "lower_bound": 0.55,
                "num_samples": 5,
                "zoom_in": False,
            },
            matching_overlap=0.5,
    ):

        self.net = net
        print("Device: ", net.device)
        self.loss_function = loss_functions[loss_specs["type"]](
            **loss_specs["parameters"])
        self.optimizer = optim.SGD(net.parameters(), **optimizer_parameters)
        self.metrics = {
            score: score_function for score, score_function in
            available_score_functions.items()
            if score in metrics + [metric_to_maximize]
        }
        self.epochs = epochs
        self.threshold_space = threshold_space
        self.metric_to_maximize = metric_to_maximize
        self.patience = patience if patience else epochs
        self.save_folder = save_folder
        self.matching_overlap = matching_overlap
        self.matching = match_events_localization_to_default_localizations
        if logger_parameters is not None:
            self.train_logger = Logger(**logger_parameters)

    def on_batch_start(self):
        pass

    def on_epoch_end(self):
        pass

    def validate(self, validation_dataset, threshold_space):
        """
        Compute metrics on validation_dataset net for test_dataset and
        select best classification threshold
        """

        best_thresh = -1
        best_metrics_epoch = {
            metric: -1
            for metric in self.metrics.keys()
        }

        # Compute predicted_events
        '''thresholds = np.sort(
            np.random.uniform(threshold_space["upper_bound"],
                              threshold_space["lower_bound"],
                              threshold_space["num_samples"]))'''
        thresholds = np.array([0.0, 0.3, 0.5, 0.7, 0.9])

        for threshold in thresholds:
            metrics_thresh = compute_metrics_dataset(
                self.net,
                validation_dataset,
                threshold,
            )

            # If 0 events predicted, all superiors thresh's will also predict 0
            if metrics_thresh == -1:
                if best_thresh in (self.threshold_space["upper_bound"],
                                   self.threshold_space["lower_bound"]):
                    print(
                        "Best classification threshold is " +
                        "in the boundary ({})! ".format(best_thresh) +
                        "Consider extending threshold range")
                return best_metrics_epoch, best_thresh

            # Add to logger
            if "train_logger" in vars(self):
                self.train_logger.add_new_metrics((metrics_thresh, threshold))

            # Compute mean metric to maximize across events
            mean_metric_to_maximize = np.nanmean(
                [m[self.metric_to_maximize] for m in metrics_thresh])

            if mean_metric_to_maximize >= best_metrics_epoch[
                    self.metric_to_maximize]:
                best_metrics_epoch = {
                    metric: np.nanmean(
                        [m[metric] for m in metrics_thresh])
                    for metric in self.metrics.keys()
                }

                best_thresh = threshold

        if best_thresh in (threshold_space["upper_bound"],
                           threshold_space["lower_bound"]):
            print("Best classification threshold is " +
                  "in the boundary ({})! ".format(best_thresh) +
                  "Consider extending threshold range")

        return best_metrics_epoch, best_thresh

    def get_batch_loss(self, data):
        """ Single forward and backward pass """

        # Get signals and labels
        signals, events = data
        x = signals.to(self.net.device)

        # Forward
        localizations, classifications, localizations_default = self.net.forward(x)

        # Matching
        localizations_target, classifications_target = self.matching(
            localizations_default=localizations_default,
            events=events,
            threshold_overlap=self.matching_overlap)
        localizations_target = localizations_target.to(self.net.device)
        classifications_target = classifications_target.to(self.net.device)

        # Loss
        (loss_classification_positive,
         loss_classification_negative,
         loss_localization) = (
             self.loss_function(localizations,
                                classifications,
                                localizations_target,
                                classifications_target))

        return loss_classification_positive, \
            loss_classification_negative, \
            loss_localization

    def train(self, train_dataset, validation_dataset, batch_size=128):
        """ Metwork training with backprop """

        dataloader_parameters = {
            "num_workers": 0,
            "shuffle": True,
            "collate_fn": collate,
            "pin_memory": True,
            "batch_size": 128,
            'drop_last': True,
        }
        print("batch_size:", batch_size)

        dataloader_train = DataLoader(train_dataset, **dataloader_parameters)
        dataloader_val = DataLoader(validation_dataset, **dataloader_parameters)


        metrics_final = {
            metric: 0
            for metric in self.metrics.keys()
        }

        best_value = -np.inf
        best_threshold = None
        best_net = None
        counter_patience = 0
        last_update = None
        t = tqdm.tqdm(range(self.epochs,))
        for epoch, _ in enumerate(t):
            if epoch != 0:
                t.set_postfix(
                    best_metric_score=best_value,
                    threshold=best_threshold,
                    last_update=last_update,
                )

            epoch_loss_classification_positive_train = 0.0
            epoch_loss_classification_negative_train = 0.0
            epoch_loss_localization_train = 0.0

            epoch_loss_classification_positive_val = 0.0
            epoch_loss_classification_negative_val = 0.0
            epoch_loss_localization_val = 0.0

            for i, data in enumerate(dataloader_train, 0):
                '''s, e = data
                plot_data(s[0], e[0], fs = 128,
                    channel_names=['c3', 'c4', 'eogl', 'eogr', 'chin', 'legl', 'legr', 'abdo', 'thor', 'nasal'])
                '''
                # On batch start
                self.on_batch_start()

                self.optimizer.zero_grad()

                # Set network to train mode
                self.net.train()

                (loss_classification_positive,
                 loss_classification_negative,
                 loss_localization) = self.get_batch_loss(data)

                epoch_loss_classification_positive_train += \
                    loss_classification_positive
                epoch_loss_classification_negative_train += \
                    loss_classification_negative
                epoch_loss_localization_train += loss_localization

                loss = loss_classification_positive \
                    + loss_classification_negative \
                    + loss_localization
                loss.backward()
                # gradient descent
                self.optimizer.step()

            epoch_loss_classification_positive_train /= (i + 1)
            epoch_loss_classification_negative_train /= (i + 1)
            epoch_loss_localization_train /= (i + 1)

            with torch.no_grad():
                self.net.eval()
                for i, data in enumerate(dataloader_val, 0):
                    (loss_classification_positive,
                     loss_classification_negative,
                     loss_localization) = self.get_batch_loss(data)

                    epoch_loss_classification_positive_val += \
                        loss_classification_positive
                    epoch_loss_classification_negative_val += \
                        loss_classification_negative
                    epoch_loss_localization_val += loss_localization

            epoch_loss_classification_positive_val /= (i + 1)
            epoch_loss_classification_negative_val /= (i + 1)
            epoch_loss_localization_val /= (i + 1)

            metrics_epoch, threshold = self.validate(
                validation_dataset=validation_dataset,
                threshold_space=self.threshold_space,
            )

            if self.threshold_space["zoom_in"] and threshold != -1:
                threshold_space_size = self.threshold_space["upper_bound"] - \
                    self.threshold_space["lower_bound"]
                zoom_metrics_epoch, zoom_threshold = self.validate(
                    validation_dataset=validation_dataset,
                    threshold_space={
                        "upper_bound": threshold + 0.1 * threshold_space_size,
                        "lower_bound": threshold - 0.1 * threshold_space_size,
                        "num_samples": self.threshold_space["num_samples"],
                    })
                if zoom_metrics_epoch[self.metric_to_maximize] > metrics_epoch[
                        self.metric_to_maximize]:
                    metrics_epoch = zoom_metrics_epoch
                    threshold = zoom_threshold

            if self.save_folder:
                self.net.save(self.save_folder + str(epoch) + "_net")

            if metrics_epoch[self.metric_to_maximize] > best_value:
                best_value = metrics_epoch[self.metric_to_maximize]
                best_threshold = threshold
                last_update = epoch
                best_net = copy.deepcopy(self.net)
                metrics_final = {
                    metric: metrics_epoch[metric]
                    for metric in self.metrics.keys()
                }
                counter_patience = 0
            else:
                counter_patience += 1

            if counter_patience > self.patience:
                break

            self.on_epoch_end()
            if "train_logger" in vars(self):
                self.train_logger.add_new_loss(
                    epoch_loss_localization_train.item(),
                    epoch_loss_classification_positive_train.item(),
                    epoch_loss_classification_negative_train.item(),
                    mode="train"
                )
                self.train_logger.add_new_loss(
                    epoch_loss_localization_val.item(),
                    epoch_loss_classification_positive_val.item(),
                    epoch_loss_classification_negative_val.item(),
                    mode="validation"
                )
                self.train_logger.add_current_metrics_to_history()
                self.train_logger.dump_train_history()

        return best_net, metrics_final, best_threshold


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc='upper right')


def plot_data(
        data: np.ndarray,
        events: np.ndarray,
        fs: int,
        predictions: Optional[np.ndarray] = None,
        channel_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        ax: Optional[Axes] = None,
        ind: Optional[int] = None,
) -> Tuple[Figure, Axes]:
    # Get current axes or create new
    if ax is None:
        fig, ax = plt.subplots(figsize=((24, 9)))
        ax.set_xlabel("Time (s)")
    else:
        fig = ax.get_figure()
        ax.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off

    font = {'family': 'Times new roman',
            'weight': 'bold',
            'size': 20}

    plt.rc('font', **font)

    C, T = data.shape
    time_vector = np.arange(T) / fs

    assert (
            len(channel_names) == C
    ), f"Channel names are inconsistent with number of channels in data. Received {channel_names=} and {data.shape=}"

    event_label_dict = {0.0: 'Arousal', 1.0: 'Leg Movement', 2.0: 'Sleep-disordered Breathing'}
    # Plot events
    if True:
        for event_label in np.unique(events[:, -1]):
            if event_label == 0.0:
                color = "r"
            elif event_label == 1.0:
                color = "yellow"
            elif event_label == 2.0:
                color = "cornflowerblue"
            class_events = events[events[:, -1] == event_label, :-1] * T / fs
            for evt_start, evt_stop in class_events:
                label = np.unique(event_label_dict[event_label])
                ax.axvspan(evt_start, evt_stop, facecolor=color, edgecolor=None, alpha=0.6, label=label)
                legend_without_duplicate_labels(ax)
    # Calculate the offset between signals
    data = data.cpu().detach().numpy()

    data = (
        2
        * (data - data.min(axis=-1, keepdims=True))
        / (data.max(axis=-1, keepdims=True) - data.min(axis=-1, keepdims=True))
        - 1
    )
    offset = np.zeros((C, T))
    for idx in range(C - 1):
        # offset[idx + 1] = -(np.abs(np.min(data[idx])) + np.abs(np.max(data[idx + 1])))
        offset[idx + 1] = -2 * (idx + 1)
    ax.plot(time_vector, data.T + offset.T, color="gray", linewidth=1)
    ax.set_xlim(time_vector[0], time_vector[-1])
    ax.set_yticks(ticks=offset[:, 0], labels=channel_names)
    plt.show()