class LangEvalAlgo:

    def __init__(self, juror_models: list, judge_model: str, possible_emotions: list):
        self._judge_model = judge_model
        self._juror_models = juror_models
        self._possible_emotions = possible_emotions

    async def run(self, example: str):
        # This is the main method which will call the judge and jurors methods
        # iterate through the dataset row by row (should it take the path to the
        # We also need a jinja template in which we can inject the example and the landID
        # Calls the juror method first
        # invokes the juror method after
        # update the cached output expeted, this output format needs to match the expected codabench format (csv, check columns)

        # Detemermine the language of the input text:
        classifier = TextClassify()
        self._example_lang = classifier.classify(example)
