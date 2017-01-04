from pkg_resources import resource_string, resource_listdir
import time
from time import localtime, strftime
import json
import pprint
import os
import numpy as np


class Logger:
    """
    Interface for the Logger class
    """

    def __init__(self, log_every_step=20):
        self.log_every_step = log_every_step

    def new_model(self, conf, model_id=None):
        print('*** new model ***')
        print(pprint.pformat(conf, width=1, indent=0))

    def new_fold(self):
        print('*****\nnew fold\n*****')

    def new_epoch(self, iepoch):
        print('*** epoch:', iepoch)
        self.train_loss = [0, 0]
        self.test_loss = [0, 0]

    def train_step(self, loss, epochs):
        self.train_loss[0] += loss
        self.train_loss[1] += 1
        if(self.train_loss[1] % self.log_every_step == 0):
            print('epoch: %.2f, train loss: %.3f' % (epochs, self.train_loss[0] / self.train_loss[1]))
            self.train_loss = [0, 0]

    def test_step(self, loss, epochs):
        self.test_loss[0] += loss
        self.test_loss[1] += 1
        if(self.test_loss[1] % self.log_every_step == 0):
            print('epoch: %.2f, test loss: %.3f' % (epochs, self.test_loss[0] / self.test_loss[1]))
            self.test_loss = [0, 0]

    def train_epoch(self, train_ev, epoch):
        print('epoch:', epoch, 'train loss:', train_ev)

    def test_epoch(test_ev, epoch):
        print('epoch:', epoch, 'test loss:', test_ev)


class LoggerComposite(Logger):
    def __init__(self, loggers):
        self.loggers = loggers

    def new_model(self, conf):
        for l in self.loggers:
            l.new_model(conf)

    def new_fold(self):
        for l in self.loggers:
            l.new_fold()

    def new_epoch(self, iepoch):
        for l in self.loggers:
            l.new_epoch(iepoch)

    def append_step(self, step, loss, epochs):
        for l in self.loggers:
            l.new_epoch(self, step, loss, epochs)

    def set_current_epoch(self, epoch):
        for l in self.loggers:
            l.set_current_epoch(epoch)

    def append_train_ev(self, train_ev, epoch):
        for l in self.loggers:
            l.append_train_ev(train_ev, epoch)

    def append_test_ev(self, test_ev, epoch):
        for l in self.loggers:
            l.append_test_ev(test_ev, epoch)


class HtmlLogger(Logger):
    def __init__(self, log_path, ylabel, log_every_step=50, log_delta_sec=30):
        if(not os.path.exists(log_path)):
            os.mkdir(log_path)
        self.log_path = log_path
        self.template = resource_string('dlutils.resources', 'LogTemplate.html_template').decode('utf-8')
        self.ylabel = ylabel

        self.log_delta_sec = log_delta_sec
        self.log_time = time.time() - self.log_delta_sec
        self.log_every_step = log_every_step

    def new_model(self, conf, model_id=None):
        self.conf = conf
        self.model_id = model_id
        if(model_id is None):
            self.model_id = strftime("%Y%m%d%H%M%S", localtime())
        self.json_filename = os.path.join(self.log_path, self.model_id + '.json')
        self.html_filename = os.path.join(self.log_path, self.model_id + '.html')

        self.train_loss = []
        self.train_loss_step = []
        self.test_loss = []
        self.test_loss_step = []
        self.train_time = []

        self.i_fold = 0
        self.current_epoch = 0
        self.current_state_str = ''
        self.epoch_eta = 0
        self._write()

    def new_fold(self):
        self.train_loss.append([])
        self.train_loss_step.append([])
        self.test_loss.append([])
        self.test_loss_step.append([])
        self.train_time.append([])
        self.i_fold = len(self.train_loss) - 1

    def new_epoch(self, iepoch):
        self.current_epoch = iepoch

    def train_step(self, loss, epoch):
        self.train_loss_step[self.i_fold].append((float(epoch), float(loss)))
        self.current_epoch = epoch
        self._write()

    def test_step(self, loss, epoch):
        self.test_loss_step[self.i_fold].append((float(epoch), float(loss)))
        self._write()

    def train_epoch(self, loss, epoch):
        self.train_loss[self.i_fold].append((epoch, loss))
        self.train_time[self.i_fold].append(time.time())
        self.current_epoch = epoch
        self._write()

    def test_epoch(self, loss, epoch):
        self.test_loss[self.i_fold].append((epoch, loss))
        self._write()

    def _write(self):
        if(time.time() - self.log_time > self.log_delta_sec):
            self.log_time = time.time()
            if(len(self.train_time) > 0 and len(self.train_time[0]) >= 2):  # at least two records
                self.epoch_eta = np.mean(np.diff(np.concatenate(self.train_time)))

            self._write_json()
            self._write_html()

    def _write_json(self):
        data = {
            "model_id": self.model_id,
            "conf": self.conf,
            "train_loss": self.train_loss,
            "train_time": self.train_time,
            "test_loss": self.test_loss,
            "current_state": {
                "ifold": self.i_fold,
                "epoch": self.current_epoch,
                "epoch_eta": self.epoch_eta,
            }
        }
        json.dump(data, open(self.json_filename, 'w'))

    def _write_html(self):
        html_title = "dlutils HtmlLogger model " + self.model_id
        conf_string = pprint.pformat(self.conf, width=1, indent=0).replace("\n", "<br/>")
        conf = json.dumps(conf_string)
        current_ifold = json.dumps(self.i_fold)
        epoch_eta = json.dumps(self.epoch_eta)
        current_state = json.dumps(self.current_state_str)

        plotter = PlotlyPlotter()

        def invnest(x):
            return [[x_[0] for x_ in x], [x_[1] for x_ in x]]

        def aggregate_steps(x, steps=self.log_every_step):
            x_ = x[0]
            y_ = x[1]
            assert(len(x_) == len(y_))
            n = len(x_) // steps
            y_ = np.mean(np.reshape(y_[:n * steps], [n, steps]), axis=-1).tolist()
            return x_[steps - 1::steps], y_

        self.train_loss_step
        self.test_loss_step

        for i, (tr, trs, te, tes) in enumerate(zip(self.train_loss, self.train_loss_step, self.test_loss, self.test_loss_step)):
            tr, trs, te, tes = invnest(tr), invnest(trs), invnest(te), invnest(tes)
            trs, tes = aggregate_steps(trs), aggregate_steps(tes)
            plotter.add_trace(x=tr[0], y=tr[1], mode="lines+markers", name="train_" + str(i), color=(0, i))
            plotter.add_trace(x=trs[0], y=trs[1], name="train_step" + str(i), width=.5, color=(0, i))
            plotter.add_trace(x=te[0], y=te[1], mode="lines+markers", name="test_" + str(i), color=(1, i))
            plotter.add_trace(x=tes[0], y=tes[1], name="test_step" + str(i), width=.5, color=(1, i))

        plot_code = plotter.generate_js()
        ylabel = json.dumps(self.ylabel)
        xlabel = json.dumps("epochs")

        html_string = self.template
        html_string = html_string.replace("PYTHON_PLACEHOLDER_HTML_TITLE", html_title)
        html_string = html_string.replace("PYTHON_PLACEHOLDER_CONF", conf)
        html_string = html_string.replace("PYTHON_PLACEHOLDER_CURRENT_IFOLD", current_ifold)
        html_string = html_string.replace("PYTHON_PLACEHOLDER_CURRENT_STATE", current_state)
        html_string = html_string.replace("PYTHON_PLACEHOLDER_EPOCH_ETA", epoch_eta)
        html_string = html_string.replace("PYTHON_PLACEHOLDER_TRACES", plot_code)
        html_string = html_string.replace("PYTHON_PLACEHOLDER_YLABEL", ylabel)
        html_string = html_string.replace("PYTHON_PLACEHOLDER_XLABEL", xlabel)

        with open(self.html_filename, 'w') as htmlfile:
            htmlfile.write(html_string)


def generate_color_shades(hues, N):
    import colorsys
    import math
    out = []
    for i, hue in enumerate(hues):
        out.append([])
        hue_ = hue / 360
        sat_range = 0.5
        val_range = 0.5
        HSV_tuples = [(hue_, 1 - sat_range * x * 1.0 / N, (1 - val_range) + val_range * x * 1.0 / N) for x in range(N)]
        RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

        for r in RGB_tuples:
            r_ = [math.floor(255 * r_) for r_ in r]
            out[i].append("#%02X%02X%02X" % (r_[0], r_[1], r_[2]))
    return out


class PlotlyPlotter:
    def __init__(self, hues=[209, 29], N=5):
        self.traces = []
        self.color_shades = generate_color_shades(hues, N)

    def add_trace(self, x, y, mode='lines', name='', dash='solid', width=2, color=(0, 0)):
        trace = {
            "x": x,
            "y": y,
            "mode": mode,
            "name": name,
            "line": {
                "dash": dash,
                "width": width,
                "color": self.color_shades[color[0]][color[1]],
            }
        }
        self.traces.append(trace)

    def generate_js(self):
        out = ""
        out_data = []
        for i, t in enumerate(self.traces):
            var_name = "trace" + str(i)
            out += "var " + var_name + " = "
            out += json.dumps(t)
            out += ";\n\n"
            out_data.append(var_name)

        out += "var data = ["
        for d in out_data:
            out += d + ", "
        out += "];\n\n"

        return out
