import json


class Writer:

    @staticmethod
    def json_to_file(params: dict, file_path: str):
        with open(file_path, 'w') as fp:
            json.dump(params, fp)