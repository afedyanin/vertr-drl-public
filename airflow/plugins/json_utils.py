import dataclasses, json


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def dataclass_to_json(dataclass):
    return json.dumps(
        dataclasses.asdict(dataclass),
        cls=EnhancedJSONEncoder,
        ensure_ascii=False,
        indent=4,
        sort_keys=True,
        default=str)
