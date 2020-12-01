from dataclasses import dataclass
from typing import Dict, List, Optional, Union


class ScoreException(Exception):
    pass


SeriesDict = Dict[str, float]
DataFrameDict = Dict[str, SeriesDict]
RocDict = Dict[str, List[float]]
ScoreDict = Dict[str, Union[float, Optional[float], SeriesDict, DataFrameDict, Dict[str, RocDict]]]


@dataclass
class Score:
    overall: float
    validation: Optional[float]

    def to_string(self) -> str:
        output = f'Overall: {self.overall}\n'
        output += f'Validation: {self.validation}'
        return output

    def to_dict(self) -> ScoreDict:
        return {'overall': self.overall, 'validation': self.validation}
