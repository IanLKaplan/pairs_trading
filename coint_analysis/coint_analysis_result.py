class CointInfo:
    def __init__(self,
                 pair_str: str,
                 confidence: int,
                 weight: float,
                 has_intercept: bool,
                 intercept: float):
        self.pair_str = pair_str
        self.confidence = confidence
        self.weight = weight
        self.has_intercept = has_intercept
        self.intercept = intercept

    def __str__(self):
        s_true = f'pair: {self.pair_str} confidence: {self.confidence} weight: {self.weight} intercept: {self.intercept}'
        s_false = f'pair: {self.pair_str} confidence: {self.confidence} weight: {self.weight}'
        s = s_true if self.has_intercept else s_false
        return s


class CointAnalysisResult:
    def __init__(self,
                 granger_coint: CointInfo = None,
                 johansen_coint: CointInfo = None):
        self.granger_coint = granger_coint
        self.johansen_coint = johansen_coint
