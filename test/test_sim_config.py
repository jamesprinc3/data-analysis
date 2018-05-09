from unittest import TestCase

from sim_config import SimConfig


class TestGenerateConfigString(TestCase):
    def setUp(self):
        pass

    def test_generate_str(self):
        config = {'paths': {'testPath': "test/path"}}
        config_str = SimConfig.generate_config_string(config)

        assert config_str == "paths {\n\ttestPath = \"test/path\"\n}"

    def test_generate_int(self):
        config = {'execution': {'numSimulations': 10}}
        config_str = SimConfig.generate_config_string(config)

        assert config_str == "execution {\n\tnumSimulations = 10\n}"

    def test_generate_bool(self):
        config = {'execution': {'parallel': True}}
        config_str = SimConfig.generate_config_string(config)

        assert config_str == "execution {\n\tparallel = true\n}"

    def test_generate_two_params(self):
        config = {'execution': {'numSimulations': 10, 'parallel': True}}
        config_str = SimConfig.generate_config_string(config)

        assert config_str == "execution {\n\tnumSimulations = 10,\n\tparallel = true\n}"

    def test_generate_two_sections(self):
        config = {'execution': {'numSimulations': 10}, 'paths': {'testPath': "test/path"}}
        config_str = SimConfig.generate_config_string(config)

        assert config_str == "execution {\n\tnumSimulations = 10\n}\n\npaths {\n\ttestPath = \"test/path\"\n}"

# execution {
# 	numSimulations = 10
# }
#
# paths {
# 	testPath = "test/path"
# }