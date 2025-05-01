from ollama import AsyncClient
from lang_detect.lang_detection_lib.lang_classify import TextClassify
from templates import (
    JUDGE_TEMPLATE,
    JUROR_TEMPLATE_1,
    JUROR_TEMPLATE_2,
    JUROR_TEMPLATE_3,
    JUROR_TEMPLATE_4,
)


class LangEvalAlgo:
    """Main Text Evaluation Class for the framework"""

    def __init__(
        self,
        juror_model: str,
        judge_model: str,
        possible_emotions: list,
    ):
        self._judge_model = judge_model
        self._possible_emotions = possible_emotions
        self.juror_model = juror_model
        self._example_lang = None
        self.lang_classifier = TextClassify()

    async def run_jurors(self, example: str):
        """Runs the example through all 4 jurors sequentially.

        Args:
            example (str): current example being classified.

        Returns:
            list: list containing the language of the example and the evaluations from the 4 jurors.
        """
        # Stores the results from all the jurors
        juror_results = []

        #  Get the language of current example
        self._example_lang = self.lang_classifier.classify(example)

        # add the language of the example to first index in juror_results
        juror_results.append(self._example_lang)

        juror1 = await self._juror(example=example, template=JUROR_TEMPLATE_1)
        juror2 = await self._juror(example=example, template=JUROR_TEMPLATE_2)
        juror3 = await self._juror(example=example, template=JUROR_TEMPLATE_3)
        juror4 = await self._juror(example=example, template=JUROR_TEMPLATE_4)

        juror_results.append(juror1)
        juror_results.append(juror2)
        juror_results.append(juror3)
        juror_results.append(juror4)

        return juror_results

    async def run_judge(
        self, example: str, language: str, juror_assessments: list
    ) -> str:
        """_summary_

        Args:
            example (str): example to be evaluate
            juror_assessments (list): list containing the language of examplea and evaluations from jurors from previous step.

        Returns:
            str: returns the final assessment from the juror.
        """
        # Create the judge prompt make final decision
        prompt = self._create_judge_template(
            example=example,
            example_lang=language,
            juror_assessments=juror_assessments,
        )
        judge_response = await self._query_model(self._judge_model, prompt)
        return judge_response

    async def _juror(self, example: str, template: str) -> str:
        """One instance of a juror

        Args:
            example (str): text to be evaluate.
            template (str): the template in which to inject the example into.

        Returns:
           str: string response from that particual juror (expert)
        """

        # Create a juror template using one of the four persona
        prompt = self._create_juror_template(example=example, template=template)

        # Query the model example injected into template
        response = await self._query_model(model=self.juror_model, prompt=prompt)

        return response

    def _create_juror_template(self, example: str, template: str) -> str:
        """Generate a string prompt (template) to pass to the jurors.

        Args:
            example (str): example to be evaluated
            template (str): template to inject example into for prompt creation

        Returns:
            str: final template for that particular juror.
        """
        # inject variables into the template
        juror_template = (
            template.replace("{{possible_langs}}", str(self._possible_emotions))
            .replace("{{lang_id}}", self._example_lang)
            .replace("{{text}}", example)
        )

        return juror_template

    def _create_judge_template(
        self, example: str, example_lang: str, juror_assessments: list
    ) -> str:
        """Generate a string prompt (template) to pass to the judge.

        Args:
            example (str): example to be evaluated
            example_lang (str): language of example
            template (str): template to inject example into for prompt creation

        Returns:
            str: final template for the judge.
        """
        # Parse the assessments from the jurors
        formatted_assessments = "\n".join(
            f"Juror{value}\nResponse: {str(emotions)}\n"
            for value, emotions in enumerate(juror_assessments)
        )

        # inject assessments into judge template
        judge_template = (
            JUDGE_TEMPLATE.replace("{{juror_assessment}}", formatted_assessments)
            .replace("{{lang_id}}", example_lang)
            .replace("{{possible_emotions}}", str(self._possible_emotions))
            .replace("{{text}}", example)
        )

        return judge_template

    async def _query_model(self, model: str, prompt: str) -> str:
        """Queries the LLM via Ollama given the model and prompt

        Args:
            model (str): name of the model to query
            prompt (str): prompt detailing what the model should do.

        Returns:
            str: parsed string output from the model
        """
        response = await AsyncClient().chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        # Parse the raw string returned from the LLM
        return response["message"]["content"]
