import json
import logging


class Writer:

    logger = logging.getLogger("Writer")

    @classmethod
    def json_to_file(cls, params: dict, file_path: str):
        try:
            with open(file_path, 'w') as fp:
                json.dump(params, fp)
        except Exception as e:
            cls.logger.error("Writing JSON to file failed, exception was " + str(e))
            raise e
