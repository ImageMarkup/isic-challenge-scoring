from dataclasses import dataclass


class ScoreError(Exception):
    pass


SeriesDict = dict[str, float]
DataFrameDict = dict[str, SeriesDict]
RocDict = dict[str, list[float]]
ScoreDict = dict[str, float | None | SeriesDict | DataFrameDict | dict[str, RocDict]]


@dataclass
class Score:
    overall: float
    validation: float | None

    def to_string(self) -> str:
        output = f'Overall: {self.overall}\n'
        output += f'Validation: {self.validation}'
        return output

    def to_dict(self) -> ScoreDict:
        return {'overall': self.overall, 'validation': self.validation}
