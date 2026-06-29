import json


def hash_args(args: dict) -> str:
    return json.dumps(args, sort_keys=True, default=str)
