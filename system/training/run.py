# coding=utf-8
#!/usr/bin/env python3
import datetime
import torch
from pprint import pformat
import glob
import models
from dataset import create_dataloader
import fire
import losses
import logging
import pandas as pd
import kaldi_io
import yaml
import os
import numpy as np
from sklearn import metrics
import sklearn.preprocessing as pre
import uuid
from tabulate import tabulate
import sys
from ignite.contrib.handlers import ProgressBar
from ignite.engine import (Engine, Events)
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix, MeanAbsoluteError, Precision, Recall
from ignite.contrib.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import StepLR

device = 'cpu'
if torch.cuda.is_available(
) and 'SLURM_JOB_PARTITION' in os.environ and 'gpu' in os.environ[
        'SLURM_JOB_PARTITION']:
    device = 'cuda'
    # Without results are slightly inconsistent
    torch.backends.cudnn.deterministic = True
DEVICE = torch.device(device)


class Runner(object):
    """docstring for Runner"""
    def __init__(self, seed=0):
        super(Runner, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)

    @staticmethod
    def _forward(model, batch, poolingfunction):
        inputs, targets = batch
        inputs, targets = inputs.float().to(DEVICE), targets.float().to(DEVICE)
        return poolingfunction(model(inputs), 1), targets

    def train(self, config, **kwargs):
        config_parameters = parse_config_or_kwargs(config, **kwargs)
        outputdir = os.path.join(
            config_parameters['outputpath'], config_parameters['model'],
            "{}_{}".format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m'),
                uuid.uuid1().hex))
        checkpoint_handler = ModelCheckpoint(
            outputdir,
            'run',
            n_saved=1,
            require_empty=False,
            create_dir=True,
            score_function=lambda engine: -engine.state.metrics['Loss'],
            save_as_state_dict=False,
            score_name='loss')

        train_kaldi_string = parsecopyfeats(
            config_parameters['trainfeatures'],
            **config_parameters['feature_args'])
        dev_kaldi_string = parsecopyfeats(config_parameters['devfeatures'],
                                          **config_parameters['feature_args'])
        logger = genlogger(os.path.join(outputdir, 'train.log'))
        logger.info("Experiment is stored in {}".format(outputdir))
        for line in pformat(config_parameters).split('\n'):
            logger.info(line)
        scaler = getattr(
            pre,
            config_parameters['scaler'])(**config_parameters['scaler_args'])
        inputdim = -1
        logger.info("<== Estimating Scaler ({}) ==>".format(
            scaler.__class__.__name__))
        for _, feat in kaldi_io.read_mat_ark(train_kaldi_string):
            scaler.partial_fit(feat)
            inputdim = feat.shape[-1]
        assert inputdim > 0, "Reading inputstream failed"
        logger.info("Features: {} Input dimension: {}".format(
            config_parameters['trainfeatures'], inputdim))
        logger.info("<== Labels ==>")
        train_label_df = pd.read_csv(
            config_parameters['trainlabels']).set_index('Participant_ID')
        dev_label_df = pd.read_csv(
            config_parameters['devlabels']).set_index('Participant_ID')
        train_label_df.index = train_label_df.index.astype(str)
        dev_label_df.index = dev_label_df.index.astype(str)
        # target_type = ('PHQ8_Score', 'PHQ8_Binary')
        target_type = ('PHQ8_Score', 'PHQ8_Binary')
        n_labels = 3  # PHQ8 (1) + Cross_entropy  (2)
        # Scores and their respective PHQ8
        train_labels = train_label_df.loc[:, target_type].T.apply(
            tuple).to_dict()
        dev_labels = dev_label_df.loc[:, target_type].T.apply(tuple).to_dict()
        train_dataloader = create_dataloader(
            train_kaldi_string,
            train_labels,
            transform=scaler.transform,
            shuffle=True,
            **config_parameters['dataloader_args'])
        cv_dataloader = create_dataloader(
            dev_kaldi_string,
            dev_labels,
            transform=scaler.transform,
            shuffle=False,
            **config_parameters['dataloader_args'])
        model = getattr(models, config_parameters['model'])(
            inputdim=inputdim,
            output_size=n_labels,
            **config_parameters['model_args'])
        if 'pretrain' in config_parameters:
            logger.info("Loading pretrained model {}".format(
                config_parameters['pretrain']))
            pretrained_model = torch.load(config_parameters['pretrain'],
                                          map_location=lambda st, loc: st)
            if 'Attn' in pretrained_model.__class__.__name__:
                model.lstm.load_state_dict(pretrained_model.lstm.state_dict())
            else:
                model.net.load_state_dict(pretrained_model.net.state_dict())
        logger.info("<== Model ==>")
        for line in pformat(model).split('\n'):
            logger.info(line)
        criterion = getattr(
            losses,
            config_parameters['loss'])(**config_parameters['loss_args'])
        optimizer = getattr(torch.optim, config_parameters['optimizer'])(
            model.parameters(), **config_parameters['optimizer_args'])
        poolingfunction = parse_poolingfunction(
            config_parameters['poolingfunction'])
        criterion = criterion.to(device)
        model = model.to(device)

        def _train_batch(_, batch):
            model.train()
            with torch.enable_grad():
                optimizer.zero_grad()
                outputs, targets = Runner._forward(model, batch,
                                                   poolingfunction)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                return loss.item()

        def _inference(_, batch):
            model.eval()
            with torch.no_grad():
                return Runner._forward(model, batch, poolingfunction)

        def meter_transform(output):
            y_pred, y = output
            # y_pred is of shape [Bx3] (0 = MSE, 1+2 = Xent)
            # y = is of shape [Bx2] (0=Mse, 1 = Xent)
            return y_pred[:, 1:3], y[:, 1].long()

        precision = Precision(output_transform=meter_transform, average=False)
        recall = Recall(output_transform=meter_transform, average=False)
        F1 = (precision * recall * 2 / (precision + recall)).mean()
        metrics = {
            'Loss':
            Loss(criterion),
            'Accuracy':
            Accuracy(output_transform=meter_transform),
            'Recall':
            Recall(output_transform=meter_transform, average=True),
            'Precision':
            Precision(output_transform=meter_transform, average=True),
            'MAE':
            MeanAbsoluteError(
                output_transform=lambda out: (out[0][:, 0], out[1][:, 0])),
            'F1':
            F1
        }

        train_engine = Engine(_train_batch)
        inference_engine = Engine(_inference)
        for name, metric in metrics.items():
            metric.attach(inference_engine, name)
        RunningAverage(output_transform=lambda x: x).attach(
            train_engine, 'run_loss')
        pbar = ProgressBar(persist=False)
        pbar.attach(train_engine, ['run_loss'])

        scheduler = getattr(torch.optim.lr_scheduler,
                            config_parameters['scheduler'])(
                                optimizer,
                                **config_parameters['scheduler_args'])
        early_stop_handler = EarlyStopping(
            patience=5,
            score_function=lambda engine: -engine.state.metrics['Loss'],
            trainer=train_engine)
        inference_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                           early_stop_handler)
        inference_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                           checkpoint_handler, {
                                               'model': model,
                                               'scaler': scaler,
                                               'config': config_parameters
                                           })

        @train_engine.on(Events.EPOCH_COMPLETED)
        def compute_metrics(engine):
            inference_engine.run(cv_dataloader)
            validation_string_list = [
                "Validation Results - Epoch: {:<3}".format(engine.state.epoch)
            ]
            for metric in metrics:
                validation_string_list.append("{}: {:<5.2f}".format(
                    metric, inference_engine.state.metrics[metric]))
            logger.info(" ".join(validation_string_list))
            logger.info("\n")

            pbar.n = pbar.last_print_n = 0

        @inference_engine.on(Events.COMPLETED)
        def update_reduce_on_plateau(engine):
            val_loss = engine.state.metrics['Loss']
            if 'ReduceLROnPlateau' == scheduler.__class__.__name__:
                scheduler.step(val_loss)
            else:
                scheduler.step()

        train_engine.run(train_dataloader,
                         max_epochs=config_parameters['epochs'])
        # Return for further processing
        return outputdir

    def autoencoder(self, config, **kwargs):
        config_parameters = parse_config_or_kwargs(config, **kwargs)
        outputdir = os.path.join(
            config_parameters['outputpath'], config_parameters['model'],
            "{}_{}".format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m'),
                uuid.uuid1().hex))
        checkpoint_handler = ModelCheckpoint(
            outputdir,
            'run',
            n_saved=1,
            require_empty=False,
            create_dir=True,
            score_function=lambda engine: -engine.state.metrics['Loss'],
            save_as_state_dict=False,
            score_name='loss')

        train_kaldi_string = parsecopyfeats(
            config_parameters['trainfeatures'],
            **config_parameters['feature_args'])
        dev_kaldi_string = parsecopyfeats(config_parameters['devfeatures'],
                                          **config_parameters['feature_args'])
        logger = genlogger(os.path.join(outputdir, 'train.log'))
        logger.info("Experiment is stored in {}".format(outputdir))
        for line in pformat(config_parameters).split('\n'):
            logger.info(line)
        scaler = getattr(
            pre,
            config_parameters['scaler'])(**config_parameters['scaler_args'])
        inputdim = -1
        logger.info("<== Estimating Scaler ({}) ==>".format(
            scaler.__class__.__name__))
        train_labels_dummy = {}
        for k, feat in kaldi_io.read_mat_ark(train_kaldi_string):
            scaler.partial_fit(feat)
            inputdim = feat.shape[-1]
            train_labels_dummy[k] = np.empty(1)
        assert inputdim > 0, "Reading inputstream failed"
        logger.info("Features: {} Input dimension: {}".format(
            config_parameters['trainfeatures'], inputdim))
        train_dataloader = create_dataloader(
            train_kaldi_string,
            train_labels_dummy,
            transform=scaler.transform,
            shuffle=False,
            **config_parameters['dataloader_args'])
        model = models.AutoEncoderLSTM(inputdim)
        logger.info("<== Model ==>")
        for line in pformat(model).split('\n'):
            logger.info(line)
        criterion = losses.MSELoss()
        optimizer = getattr(torch.optim, config_parameters['optimizer'])(
            model.parameters(), **config_parameters['optimizer_args'])
        criterion = criterion.to(device)
        model = model.to(device)

        def _train_batch(_, batch):
            model.train()
            with torch.enable_grad():
                input_x, _ = batch
                optimizer.zero_grad()
                outputs = model(input_x)
                loss = criterion(outputs, input_x)
                loss.backward()
                optimizer.step()
                return loss.item()

        def _inference(_, batch):
            model.eval()
            with torch.no_grad():
                x, _ = batch
                o = model(x)
                return o, x

        metrics = {
            'Loss': Loss(criterion),
        }

        train_engine = Engine(_train_batch)
        inference_engine = Engine(_inference)
        for name, metric in metrics.items():
            metric.attach(inference_engine, name)
        RunningAverage(output_transform=lambda x: x).attach(
            train_engine, 'run_loss')
        pbar = ProgressBar(persist=False)
        pbar.attach(train_engine, ['run_loss'])

        step_scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
        scheduler = LRScheduler(step_scheduler)
        train_engine.add_event_handler(Events.ITERATION_STARTED, scheduler)
        early_stop_handler = EarlyStopping(
            patience=1,
            score_function=lambda engine: -engine.state.metrics['Loss'],
            trainer=train_engine)
        inference_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                           early_stop_handler)
        inference_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                           checkpoint_handler, {
                                               'model': model,
                                               'scaler': scaler,
                                               'config': config_parameters
                                           })

        @train_engine.on(Events.EPOCH_COMPLETED)
        def compute_metrics(engine):
            inference_engine.run(train_dataloader)
            validation_string_list = [
                "Train Results - Epoch: {:<3}".format(engine.state.epoch)
            ]
            for metric in metrics:
                validation_string_list.append("{}: {:<5.2f}".format(
                    metric, inference_engine.state.metrics[metric]))
            logger.info(" ".join(validation_string_list))
            logger.info("\n")

            pbar.n = pbar.last_print_n = 0

        train_engine.run(train_dataloader,
                         max_epochs=config_parameters['epochs'])
        # Return for further processing
        return outputdir

    def evaluate(self,
                 experiment_path: str,
                 outputfile: str = 'results.csv',
                 **kwargs):
        """Prints out the stats for the given model ( MAE, RMSE, F1, Pre, Rec)


        """
        config = torch.load(glob.glob(
            "{}/run_config*".format(experiment_path))[0],
                            map_location=lambda storage, loc: storage)
        model = torch.load(glob.glob(
            "{}/run_model*".format(experiment_path))[0],
                           map_location=lambda storage, loc: storage)
        scaler = torch.load(glob.glob(
            "{}/run_scaler*".format(experiment_path))[0],
                            map_location=lambda storage, loc: storage)
        config_parameters = dict(config, **kwargs)
        dev_features = config_parameters['devfeatures']
        dev_label_df = pd.read_csv(
            config_parameters['devlabels']).set_index('Participant_ID')
        dev_label_df.index = dev_label_df.index.astype(str)

        dev_labels = dev_label_df.loc[:, ['PHQ8_Score', 'PHQ8_Binary'
                                          ]].T.apply(tuple).to_dict()
        outputfile = os.path.join(experiment_path, outputfile)
        y_score_true, y_score_pred, y_binary_pred, y_binary_true = [], [], [], []

        poolingfunction = parse_poolingfunction(
            config_parameters['poolingfunction'])
        dataloader = create_dataloader(dev_features,
                                       dev_labels,
                                       transform=scaler.transform,
                                       batch_size=1,
                                       num_workers=2,
                                       shuffle=False)

        model = model.to(device)
        with torch.no_grad():
            for batch in dataloader:
                output, target = Runner._forward(model, batch, poolingfunction)
                y_score_pred.append(output[:, 0].cpu().numpy())
                y_score_true.append(target[:, 0].cpu().numpy())
                y_binary_pred.append(
                    torch.argmax(output[:, 1:3], dim=1).cpu().numpy())
                y_binary_true.append(target[:, 1].cpu().numpy())
        y_score_true = np.concatenate(y_score_true)
        y_score_pred = np.concatenate(y_score_pred)
        y_binary_pred = np.concatenate(y_binary_pred)
        y_binary_true = np.concatenate(y_binary_true)

        with open(outputfile, 'w') as wp:
            pre = metrics.precision_score(y_binary_true,
                                          y_binary_pred,
                                          average='macro')
            rec = metrics.recall_score(y_binary_true,
                                       y_binary_pred,
                                       average='macro')
            f1 = 2 * pre * rec / (pre + rec)
            rmse = np.sqrt(
                metrics.mean_squared_error(y_score_true, y_score_pred))
            mae = metrics.mean_absolute_error(y_score_true, y_score_pred)
            df = pd.DataFrame(
                {
                    'precision': pre,
                    'recall': rec,
                    'F1': f1,
                    'MAE': mae,
                    'RMSE': rmse
                },
                index=["Macro"])
            df.to_csv(wp, index=False)
            print(tabulate(df, headers='keys'))
        return df

    def evaluates(
            self,
            *experiment_paths: str,
            outputfile: str = 'scores.csv',
            num_workers: int = 2,
    ):
        result_dfs = []
        for exp_path in experiment_paths:
            print("Evaluating {}".format(exp_path))
            result_df = self.evaluate(exp_path)
            exp_config = torch.load(glob.glob(
                "{}/run_config*".format(exp_path))[0],
                                    map_location=lambda storage, loc: storage)
            result_df['exp'] = os.path.basename(exp_path)
            result_df['model'] = exp_config['model']
            result_df['optimizer'] = exp_config['optimizer']
            result_df['batch_size'] = exp_config['dataloader_args'][
                'batch_size']
            result_df['poolingfunction'] = exp_config['poolingfunction']
            result_df['loss'] = exp_config['loss']
            result_dfs.append(result_df)
        df = pd.concat(result_dfs)
        df.sort_values(by='F1', ascending=False, inplace=True)

        with open(outputfile, 'w') as wp:
            df.to_csv(wp, index=False)
            print(tabulate(df, headers='keys', tablefmt="pipe"))

    def check_dataloader(self, config, **kwargs):
        config_parameters = parse_config_or_kwargs(config, **kwargs)
        train_label_df = pd.read_csv(
            config_parameters['trainlabels']).set_index('Participant_ID')
        train_label_df.index = train_label_df.index.astype(str)
        # target_type = ('PHQ8_Score', 'PHQ8_Binary')
        target_type = ('PHQ8_Score', 'PHQ8_Binary')
        # Scores and their respective PHQ8
        train_labels = train_label_df.loc[:, target_type].T.apply(
            tuple).to_dict()
        train_kaldi_string = parsecopyfeats(
            config_parameters['trainfeatures'],
            **config_parameters['feature_args'])
        train_dataloader = create_dataloader(
            train_kaldi_string,
            train_labels,
            transform=lambda x: x,
            shuffle=True,
            **config_parameters['dataloader_args'])
        stat = []
        for a, b in train_dataloader:
            stat.append(b.squeeze())
        stat = torch.stack(stat).numpy()
        print(stat)


def parsecopyfeats(feat, cmvn=False, delta=False, splice=None):
    # Check if user has kaldi installed, otherwise just use kaldi_io (without extra transformations)
    import shutil
    if shutil.which('copy-feats') is None:
        return feat
    else:
        outstr = "copy-feats ark:{} ark:- |".format(feat)
        if cmvn:
            outstr += "apply-cmvn-sliding --center ark:- ark:- |"
        if delta:
            outstr += "add-deltas ark:- ark:- |"
        if splice and splice > 0:
            outstr += "splice-feats --left-context={} --right-context={} ark:- ark:- |".format(
                splice, splice)
    return outstr


def genlogger(outputfile):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger(__name__ + "." + outputfile)
    logger.setLevel(logging.INFO)
    stdlog = logging.StreamHandler(sys.stdout)
    stdlog.setFormatter(formatter)
    file_handler = logging.FileHandler(outputfile)
    file_handler.setFormatter(formatter)
    # Log to stdout
    logger.addHandler(file_handler)
    logger.addHandler(stdlog)
    return logger


def parse_config_or_kwargs(config_file, **kwargs):
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    # passed kwargs will override yaml config
    # for key in kwargs.keys():
    # assert key in yaml_config, "Parameter {} invalid!".format(key)
    return dict(yaml_config, **kwargs)


def parse_poolingfunction(poolingfunction_name='mean'):
    if poolingfunction_name == 'mean':

        def pooling_function(x, d):
            return x.mean(d)
    elif poolingfunction_name == 'max':

        def pooling_function(x, d):
            return x.max(d)[0]
    elif poolingfunction_name == 'linear':

        def pooling_function(x, d):
            return (x**2).sum(d) / x.sum(d)
    elif poolingfunction_name == 'exp':

        def pooling_function(x, d):
            return (x.exp() * x).sum(d) / x.exp().sum(d)
    elif poolingfunction_name == 'last':  # Last timestep

        def pooling_function(x, d):
            return x.select(d, -1)
    elif poolingfunction_name == 'first':

        def pooling_function(x, d):
            return x.select(d, 0)
    else:
        raise ValueError(
            "Pooling function {} not available".format(poolingfunction_name))

    return pooling_function


def _extract_features_from_model(model, features, scaler=None):
    if model.__class__.__name__ == 'LSTM':
        fwdmodel = torch.nn.Sequential(model.net)
    elif model.__class__.__name__ == 'LSTMSimpleAttn':
        fwdmodel = torch.nn.Sequential(model)
    elif model.__class__.__name__ == 'TCN':
        fwdmodel = None
    else:
        assert False, "Model not prepared for extraction"
    ret = {}
    with torch.no_grad():
        model = model.to(device)
        for k, v in kaldi_io.read_mat_ark(features):
            if scaler:
                v = scaler.transform(v)
            v = torch.from_numpy(v).to(device).unsqueeze(0)
            out = fwdmodel(v)
            if isinstance(out, tuple):  # LSTM output, 2 values hidden,and x
                out = out[0]
            ret[k] = out.cpu().squeeze().numpy()
    return ret


def extract_features(model_path: str, features='trainfeatures'):
    modeldump = torch.load(model_path,
                           map_location=lambda storage, loc: storage)
    model_dir = os.path.dirname(model_path)
    config_parameters = modeldump['config']
    dev_features = config_parameters[features]
    scaler = modeldump['scaler']
    model = modeldump['model']

    outputfile = os.path.join(model_dir, features + '.ark')
    dev_features = parsecopyfeats(dev_features,
                                  **config_parameters['feature_args'])

    vectors = _extract_features_from_model(model, dev_features, scaler)
    with open(outputfile, 'wb') as wp:
        for key, vector in vectors.items():
            kaldi_io.write_mat(wp, vector, key=key)
    return outputfile


if __name__ == '__main__':
    fire.Fire(Runner)
