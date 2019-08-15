from typing import Dict, List, Union


class ScoreException(Exception):
    pass


ScoresType = Dict[str, Dict[str, Union[float, List[Dict[str, float]]]]]
