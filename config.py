import datetime
import pathlib
from configparser import ConfigParser


class Config:

    def __init__(self, config: ConfigParser):
        """Parse the config file and apply the correct typing, etc to it"""

        self.graph_mode = config['graphs']['mode']
        self.start_time = datetime.datetime.strptime(config['data']['start_time'], "%Y-%m-%dT%H:%M:%S")

        root_path = config['full_paths']['root']
        self.root_path = root_path

        self.product = config['data']['product']
        self.num_predictions = int(config['data']['num_predictions'])
        self.interval = int(config['data']['interval'])

        self.real_root = root_path + config['part_paths']['real_root']
        self.sim_root = root_path + config['part_paths']['sim_root']
        self.orderbook_input_root = root_path + config['part_paths']['orderbook_input_root']

        self.graphs_root = root_path + config['part_paths']['graphs_output_root']
        self.params_path = root_path + config['part_paths']['params_output_root']
        pathlib.Path(self.params_path).mkdir(parents=True, exist_ok=True)
        self.sim_logs_root = root_path + config['part_paths']['sim_logs_root']

        self.orderbook_root = root_path + config['part_paths']['orderbook_output_root']
        self.correlation_root = root_path + config['part_paths']['correlation_output_root']
        self.confidence_root = root_path + config['part_paths']['confidence_output_root']

        self.sim_config_path = root_path + config['part_paths']['sim_config_path']
        self.jar_path = root_path + config['part_paths']['jar_path']

        self.sampling_window = int(config['window']['sampling'])
        self.simulation_window = int(config['window']['simulation'])
        self.orderbook_window = int(config['window']['orderbook'])

        self.mode = config['behaviour']['mode']
        self.show_graphs = config['behaviour'].getboolean('show_graphs')
        self.save_graphs = config['graphs']['mode'] == "save"
        self.fit_distributions = config['behaviour'].getboolean('fit_distributions')
        self.use_cached_params = config['behaviour'].getboolean('use_cached_params')
        self.sim_timeout = int(config['behaviour']['sim_timeout'])
        self.num_simulators = int(config['behaviour']['num_simulators'])
        self.num_traders = int(config['behaviour']['num_traders'])
        self.run_simulation = config['behaviour'].getboolean('run_simulation')

        self.ywindow = int(config['graphs']['ywindow'])
        self.xinterval = int(config['graphs']['xinterval'])

        import matplotlib

        if self.graph_mode == "save":
            matplotlib.use('PS')

        import matplotlib as plt

        self.plt = plt


