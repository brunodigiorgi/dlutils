from pkg_resources import resource_string, resource_listdir
import time
from time import localtime, strftime
import json
import pprint
import os


class Logger:
    """
    Interface for the Logger class
    """

    def __init__(self):
        pass

    def new_model(self, conf):
        pass

    def append_train_ev(self, train_ev, epoch):
        pass

    def append_test_ev(test_ev, epoch):
        pass


class LoggerComposite (Logger):
    def __init__(self, loggers):
        self.loggers = loggers

    def new_model(self, conf):
        for l in self.loggers:
            l.new_model(conf)

    def new_fold(self):
        for l in self.loggers:
            l.new_fold()

    def set_current_epoch(self, epoch):
        for l in self.loggers:
            l.set_current_epoch(epoch)

    def append_train_ev(self, train_ev, epoch):
        for l in self.loggers:
            l.append_train_ev(train_ev, epoch)

    def append_test_ev(self, test_ev, epoch):
        for l in self.loggers:
            l.append_test_ev(test_ev, epoch)


class HtmlLogger (Logger):
    def __init__(self, log_path, ylabel):
        if(not os.path.exists(log_path)):
            os.mkdir(log_path)
        self.log_path = log_path
        self.template = resource_string('dlutils.resources', 'LogTemplate.html_template').decode('utf-8')
        self.ylabel = ylabel

    def new_model(self, conf, model_id=None):
        self.conf = conf
        self.model_id = model_id
        if(model_id is None):
            self.model_id = strftime("%Y%m%d%H%M%S", localtime())
        self.json_filename = os.path.join(self.log_path, self.model_id + '.json')
        self.html_filename = os.path.join(self.log_path, self.model_id + '.html')

        self.train_ev = []
        self.train_epochs = []
        self.test_ev = []
        self.test_epochs = []
        self.i_fold = 0
        self.current_epoch = 0

    def new_fold(self):
        self.train_ev.append([])
        self.train_epochs.append([])
        self.test_ev.append([])
        self.test_epochs.append([])
        self.i_fold = len(self.train_ev) - 1

    def append_train_ev(self, train_ev, epoch):
        self.train_ev[self.i_fold].append(train_ev)
        self.train_epochs[self.i_fold].append(epoch)
        self.current_epoch = epoch
        self._write_json()
        self._write_html()

    def append_test_ev(self, test_ev, epoch):
        self.test_ev[self.i_fold].append(test_ev)
        self.test_epochs[self.i_fold].append(epoch)
        self.current_epoch = epoch
        self._write_json()
        self._write_html()

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch
        self._write_json()
        self._write_html()

    def _write_json(self):
        data = {
            "model_id": self.model_id,
            "conf": self.conf,
            "train_ev": self.train_ev,
            "train_epochs": self.train_epochs,
            "test_ev": self.test_ev,
            "test_epochs": self.test_epochs,
            "current_state": {
                "ifold": self.i_fold,
                "epoch": self.current_epoch,
            }
        }
        json.dump(data, open(self.json_filename, 'w'))

    def _write_html(self):
        html_title = "dlutils HtmlLogger model " + self.model_id
        conf_string = pprint.pformat(self.conf, width=1, indent=0).replace("\n", "<br/>")
        conf = json.dumps(conf_string)
        current_ifold = json.dumps(self.i_fold)
        current_epoch = json.dumps(self.current_epoch)

        plotter = PlotlyPlotter()
        for i, (tr_ep, tr_ev, te_ep, te_ev) in enumerate(zip(self.train_epochs, self.train_ev, self.test_epochs, self.test_ev)):
            plotter.add_trace(x=tr_ep, y=tr_ev, name="train_" + str(i), color=(0, i))
            plotter.add_trace(x=te_ep, y=te_ev, name="test_" + str(i), color=(1, i))

        plot_code = plotter.generate_js()
        ylabel = json.dumps(self.ylabel)
        xlabel = json.dumps("epochs")

        html_string = self.template
        html_string = html_string.replace("PYTHON_PLACEHOLDER_HTML_TITLE", html_title)
        html_string = html_string.replace("PYTHON_PLACEHOLDER_CONF", conf)
        html_string = html_string.replace("PYTHON_PLACEHOLDER_CURRENT_IFOLD", current_ifold)
        html_string = html_string.replace("PYTHON_PLACEHOLDER_CURRENT_EPOCH", current_epoch)
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
