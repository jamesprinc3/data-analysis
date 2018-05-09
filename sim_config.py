

class SimConfig:

    @staticmethod
    def generate_config_string(config: dict):
        ret_string = ""
        for section in config.keys():
            ret_string += section
            ret_string += " {\n"
            for param in config[section].keys():
                ret_string += "\t"
                ret_string += param
                ret_string += " = "
                if type(config[section][param]) == str:
                    ret_string += "\""
                    ret_string += str(config[section][param])
                    ret_string += "\""
                elif type(config[section][param]) == int:
                    ret_string += str(config[section][param])
                elif type(config[section][param]) == bool:
                    if config[section][param]:
                        ret_string += "true"
                    else:
                        ret_string += "false"
                ret_string += ",\n"
            ret_string = ret_string.rstrip(",\n")
            ret_string += "\n}\n\n"
        ret_string = ret_string.rstrip("\n\n")
        return ret_string
