# coding=utf-8
#!/usr/bin/env python3
import datetime
import torch
from pprint import pformat
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
import tableprint as tp
import sklearn.preprocessing as pre
import torchnet as tnt


class BinarySimilarMeter(object):
    """Only counts ones, does not consider zeros as being correct"""
    def __init__(self, sigmoid_output=False):
        super(BinarySimilarMeter, self).__init__()
        self.sigmoid_output = sigmoid_output
        self.reset()

    def reset(self):
        self.correct = 0
        self.n = 0

    def add(self, output, target):
        if self.sigmoid_output:
            output = torch.sigmoid(output)
        target = target.float()
        output = output.round()
        self.correct += np.sum(np.logical_and(output, target).numpy())
        self.n += (target == 1).nonzero().shape[0]

    def value(self):
        if self.n == 0:
            return 0
        return (self.correct / self.n) * 100.


class BinaryAccuracyMeter(object):
    """Counts all outputs, including zero"""
    def __init__(self, sigmoid_output=False):
        super(BinaryAccuracyMeter, self).__init__()
        self.sigmoid_output = sigmoid_output
        self.reset()

    def reset(self):
        self.correct = 0
        self.n = 0

    def add(self, output, target):
        if self.sigmoid_output:
            output = torch.sigmoid(output)
        output = output.float()
        target = target.float()
        output = output.round()
        self.correct += int((output == target).sum())
        self.n += np.prod(output.shape)

    def value(self):
        if self.n == 0:
            return 0
        return (self.correct / self.n) * 100.


def parsecopyfeats(feat, cmvn=False, delta=False, splice=None):
    outstr = "copy-feats ark:{} ark:- |".format(feat)
    if cmvn:
        outstr += "apply-cmvn-sliding --center ark:- ark:- |"
    if delta:
        outstr += "add-deltas ark:- ark:- |"
    if splice and splice > 0:
        outstr += "splice-feats --left-context={} --right-context={} ark:- ark:- |".format(
            splice, splice)
    return outstr


def runepoch(dataloader,
             model,
             criterion,
             optimizer=None,
             dotrain=True,
             poolfun=lambda x, d: x.mean(d)):
    model = model.train() if dotrain else model.eval()
    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = BinaryAccuracyMeter(sigmoid_output=True)
    with torch.set_grad_enabled(dotrain):
        for i, (features, targets) in enumerate(dataloader):
            features = features.float().to(device)
            targets = targets.to(device)
            outputs = model(features)
            outputs = poolfun(outputs, 1)
            loss = criterion(outputs, targets).cpu()
            loss_meter.add(loss.item())
            acc_meter.add(outputs.cpu().data[:, 1], targets.cpu().data[:, 1])
            if dotrain:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return loss_meter.value(), acc_meter.value()


def genlogger(outdir, fname):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logging.basicConfig(level=logging.DEBUG,
                        format="[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger("Pyobj, f")
    # Dump log to file
    fh = logging.FileHandler(os.path.join(outdir, fname))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def parse_config_or_kwargs(config_file, **kwargs):
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read)
    # passed kwargs will override yaml config
    for key in kwargs.keys():
        assert key in yaml_config, "Parameter {} invalid!".format(key)
    return dict(yaml_config, **kwargs)


def criterion_improver(mode):
    """Returns a function to ascertain if criterion did improve

    :mode: can be ether 'loss' or 'acc'
    :returns: function that can be called, function returns true if criterion improved

    """
    assert mode in ('loss', 'acc')
    best_value = np.inf if mode == 'loss' else 0

    def comparator(x, best_x):
        return x < best_x if mode == 'loss' else x > best_x

    def inner(x):
        # rebind parent scope variable
        nonlocal best_value
        if comparator(x, best_value):
            best_value = x
            return True
        return False

    return inner


device = 'cpu'
if torch.cuda.is_available(
) and 'SLURM_JOB_PARTITION' in os.environ and 'gpu' in os.environ[
        'SLURM_JOB_PARTITION']:
    device = 'cuda'
    torch.cuda.manual_seed(0)
device = torch.device(device)
torch.manual_seed(0)
np.random.seed(0)


def train(config='config/audio_lstm.yaml', **kwargs):
    """Trains a model on the given features and vocab.

    :features: str: Input features. Needs to be kaldi formatted file
    :config: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
    :returns: None
    """

    config_parameters = parse_config_or_kwargs(config, **kwargs)
    outputdir = os.path.join(
        config_parameters['outputpath'], config_parameters['model'],
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%f'))
    try:
        os.makedirs(outputdir)
    except IOError:
        pass
    logger = genlogger(outputdir, 'train.log')
    logger.info("Storing data at: {}".format(outputdir))
    logger.info("<== Passed Arguments ==>")
    # Print arguments into logs
    for line in pformat(config_parameters).split('\n'):
        logger.info(line)

    train_kaldi_string = parsecopyfeats(config_parameters['trainfeatures'],
                                        **config_parameters['feature_args'])
    dev_kaldi_string = parsecopyfeats(config_parameters['devfeatures'],
                                      **config_parameters['feature_args'])

    scaler = getattr(
        pre, config_parameters['scaler'])(**config_parameters['scaler_args'])
    inputdim = -1
    logger.info("<== Estimating Scaler ({}) ==>".format(
        scaler.__class__.__name__))
    for kid, feat in kaldi_io.read_mat_ark(train_kaldi_string):
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

    target_type = ('PHQ8_Score', 'PHQ8_Binary')

    # Scores and their respective
    train_labels = train_label_df.loc[:, target_type].T.apply(tuple).to_dict()
    dev_labels = dev_label_df.loc[:, target_type].T.apply(tuple).to_dict()
    n_labels = len(target_type)

    train_dataloader = create_dataloader(
        train_kaldi_string,
        train_labels,
        transform=scaler.transform,
        **config_parameters['dataloader_args'])
    cv_dataloader = create_dataloader(dev_kaldi_string,
                                      dev_labels,
                                      transform=scaler.transform,
                                      **config_parameters['dataloader_args'])
    model = getattr(models, config_parameters['model'])(
        inputdim=inputdim,
        output_size=n_labels,
        **config_parameters['model_args'])
    logger.info("<== Model ==>")
    for line in pformat(model).split('\n'):
        logger.info(line)
    model = model.to(device)
    optimizer = getattr(torch.optim, config_parameters['optimizer'])(
        model.parameters(), **config_parameters['optimizer_args'])

    scheduler = getattr(torch.optim.lr_scheduler,
                        config_parameters['scheduler'])(
                            optimizer, **config_parameters['scheduler_args'])
    criterion = getattr(
        losses, config_parameters['loss'])(**config_parameters['loss_args'])
    criterion.to(device)

    trainedmodelpath = os.path.join(outputdir, 'model.th')

    criterion_improved = criterion_improver(
        config_parameters['improvecriterion'])
    header = [
        'Epoch',
        'Loss(T)',
        'Loss(CV)',
        "Acc(T)",
        "Acc(CV)",
    ]
    for line in tp.header(header, style='grid').split('\n'):
        logger.info(line)

    poolingfunction_name = config_parameters['poolingfunction']
    pooling_function = parse_poolingfunction(poolingfunction_name)
    for epoch in range(1, config_parameters['epochs'] + 1):
        train_utt_loss_mean_std, train_utt_acc = runepoch(
            train_dataloader,
            model,
            criterion,
            optimizer,
            dotrain=True,
            poolfun=pooling_function)
        cv_utt_loss_mean_std, cv_utt_acc = runepoch(cv_dataloader,
                                                    model,
                                                    criterion,
                                                    dotrain=False,
                                                    poolfun=pooling_function)
        logger.info(
            tp.row((epoch, ) +
                   (train_utt_loss_mean_std[0], cv_utt_loss_mean_std[0],
                    train_utt_acc, cv_utt_acc),
                   style='grid'))
        epoch_meanloss = cv_utt_loss_mean_std[0]
        if epoch % config_parameters['saveinterval'] == 0:
            torch.save(
                {
                    'model': model,
                    'scaler': scaler,
                    # 'encoder': many_hot_encoder,
                    'config': config_parameters
                },
                os.path.join(outputdir, 'model_{}.th'.format(epoch)))
        # ReduceOnPlateau needs a value to work
        schedarg = epoch_meanloss if scheduler.__class__.__name__ == 'ReduceLROnPlateau' else None
        scheduler.step(schedarg)
        if criterion_improved(epoch_meanloss):
            torch.save(
                {
                    'model': model,
                    'scaler': scaler,
                    # 'encoder': many_hot_encoder,
                    'config': config_parameters
                },
                trainedmodelpath)
        if optimizer.param_groups[0]['lr'] < 1e-7:
            break
    logger.info(tp.bottom(len(header), style='grid'))
    logger.info("Results are in: {}".format(outputdir))
    return outputdir


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
    elif poolingfunction_name == 'time':  # Last timestep

        def pooling_function(x, d):
            return x.select(d, -1)
    elif poolingfunction_name == 'first':

        def pooling_function(x, d):
            return x.select(d, 0)

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
    modeldump = torch.load(model_path, lambda storage, loc: storage)
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


def stats(model_path: str, outputfile: str = 'stats.txt', cutoff: int = None):
    """Prints out the stats for the given model ( MAE, RMSE, F1, Pre, Rec)

    :model_path:str: TODO
    :returns: TODO

    """
    from tabulate import tabulate
    modeldump = torch.load(model_path, lambda storage, loc: storage)
    model_dir = os.path.dirname(model_path)
    config_parameters = modeldump['config']
    dev_features = config_parameters['devfeatures']
    dev_label_df = pd.read_csv(
        config_parameters['devlabels']).set_index('Participant_ID')
    dev_label_df.index = dev_label_df.index.astype(str)

    dev_labels = dev_label_df.loc[:, ['PHQ8_Score', 'PHQ8_Binary']].T.apply(
        tuple).to_dict()
    outputfile = os.path.join(model_dir, outputfile)
    y_score_true, y_score_pred, y_binary_pred, y_binary_true = [], [], [], []
    scores = _forward_model(model_path, dev_features, cutoff=cutoff)
    for key, score in scores.items():
        score_pred, binary_pred = torch.chunk(score, 2, dim=-1)
        y_score_pred.append(score_pred.numpy())
        y_score_true.append(dev_labels[key][0])
        y_binary_pred.append(
            torch.sigmoid(binary_pred).round().numpy().astype(int).item())
        y_binary_true.append(dev_labels[key][1])

    with open(outputfile, 'w') as wp:
        pre = metrics.precision_score(y_binary_true,
                                      y_binary_pred,
                                      average='macro')
        rec = metrics.recall_score(y_binary_true,
                                   y_binary_pred,
                                   average='macro')
        f1 = 2 * pre * rec / (pre + rec)
        rmse = np.sqrt(metrics.mean_squared_error(y_score_true, y_score_pred))
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
        print(tabulate(df, headers='keys'), file=wp)
        print(tabulate(df, headers='keys'))


def fuse(model_paths: list, outputfile='scores.txt', cutoff: int = None):
    from tabulate import tabulate
    scores = []
    for model_path in model_paths:
        modeldump = torch.load(model_path, lambda storage, loc: storage)
        config_parameters = modeldump['config']
        dev_features = config_parameters['devfeatures']
        dev_label_df = pd.read_csv(
            config_parameters['devlabels']).set_index('Participant_ID')
        dev_label_df.index = dev_label_df.index.astype(str)
        score = _forward_model(model_path, dev_features, cutoff=cutoff)
        for speaker, pred_score in score.items():
            scores.append({
                'speaker':
                speaker,
                'MAE':
                float(pred_score[0].numpy()),
                'binary':
                float(torch.sigmoid(pred_score[1]).numpy()),
                'model':
                model_path,
                'binary_true':
                dev_label_df.loc[speaker, 'PHQ8_Binary'],
                'MAE_true':
                dev_label_df.loc[speaker, 'PHQ8_Score']
            })
    df = pd.DataFrame(scores)

    spkmeans = df.groupby('speaker')[[
        'MAE', 'MAE_true', 'binary', 'binary_true'
    ]].mean()
    spkmeans['binary'] = spkmeans['binary'] > 0.5

    with open(outputfile, 'w') as wp:
        pre = metrics.precision_score(spkmeans['binary_true'].values,
                                      spkmeans['binary'].values,
                                      average='macro')
        rec = metrics.recall_score(spkmeans['binary_true'].values,
                                   spkmeans['binary'].values,
                                   average='macro')
        f1 = 2 * pre * rec / (pre + rec)
        rmse = np.sqrt(
            metrics.mean_squared_error(spkmeans['MAE_true'].values,
                                       spkmeans['MAE'].values))
        mae = metrics.mean_absolute_error(spkmeans['MAE_true'].values,
                                          spkmeans['MAE'].values)
        df = pd.DataFrame(
            {
                'precision': pre,
                'recall': rec,
                'F1': f1,
                'MAE': mae,
                'RMSE': rmse
            },
            index=["Macro"])
        print(tabulate(df, headers='keys'), file=wp)
        print(tabulate(df, headers='keys'))


def _forward_model(model_path: str,
                   features: str,
                   dopooling: bool = True,
                   cutoff=None):
    modeldump = torch.load(model_path, lambda storage, loc: storage)
    scaler = modeldump['scaler']
    config_parameters = modeldump['config']
    pooling_function = parse_poolingfunction(
        config_parameters['poolingfunction'])
    kaldi_string = parsecopyfeats(features,
                                  **config_parameters['feature_args'])
    ret = {}

    with torch.no_grad():
        model = modeldump['model'].to(device).eval()
        for key, feat in kaldi_io.read_mat_ark(kaldi_string):
            feat = scaler.transform(feat)
            if cutoff:
                # Cut all after cutoff
                feat = feat[:cutoff]
            feat = torch.from_numpy(feat).to(device).unsqueeze(0)
            output = model(feat).cpu()
            if dopooling:
                output = pooling_function(output, 1).squeeze(0)
            ret[key] = output
    return ret


def trainstats(config: str = 'config/audio_lstm.yaml', **kwargs):
    """Runs training and then prints dev stats

    :config:str: config file
    :**kwargs: Extra overwrite configs
    :returns: None

    """
    output_model = train(config, **kwargs)
    best_model = os.path.join(output_model, 'model.th')
    stats(best_model)


def run_search(config: str = 'config/audio_lstm.yaml',
               lr=0.1,
               mom=0.9,
               nest=False,
               **kwargs):
    """Runs training and then prints dev stats

    :config:str: config file
    :**kwargs: Extra overwrite configs
    :returns: None

    """
    optimizer_args = {'lr': lr, 'momentum': mom, 'nesterov': nest}
    kwargs['optimizer_args'] = optimizer_args
    output_model = train(config, **kwargs)
    best_model = os.path.join(output_model, 'model.th')
    stats(best_model)


def run_search_adam(config: str = 'config/audio_lstm.yaml', lr=0.1, **kwargs):
    """Runs training and then prints dev stats

    :config:str: config file
    :**kwargs: Extra overwrite configs
    :returns: None

    """
    optimizer_args = {'lr': lr}
    kwargs['optimizer_args'] = optimizer_args
    output_model = train(config, **kwargs)
    best_model = os.path.join(output_model, 'model.th')
    stats(best_model)


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'stats': stats,
        'trainstats': trainstats,
        'search': run_search,
        'searchadam': run_search_adam,
        'ex': extract_features,
        'fwd': _forward_model,
        'fuse': fuse,
    })
