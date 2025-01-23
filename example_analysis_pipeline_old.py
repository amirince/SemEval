# Algorithm:

# Step 1: given input_sentence: Identify language of text.

# Step 2: If language is part of output #1 & 2, continue to step 3. Else, translate to English & continue to step 3.

# Step 3: Pass text + language name to a N LLMs. Ask each about emotion intensity (output #2). Use output of emotion intensity to label output #1.

# Step 4: Combine the results across LLMs and use a larger model to finalize results for Outputs #1, #2, and #3.

# Additional Steps for Improvement:
# Finetune models for each language separately for step 3.
# Finetune judge model across all results for step 4.


# Identify language here

import subprocess
import asyncio
import subprocess
import pandas as pd
from ollama import AsyncClient
from lang_detection_lib.lang_classify import TextClassify

# Some global variables are For each track
# we also need to parse tracks or have some way to determine which track we are taking.
# There are 3 tracks A,B,C


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

        # Get the responses from jurors:
        juror_responses = await self._juror(example=example)

        judge_response = await self._judge(example, juror_assessments=juror_responses)

        return judge_response

    async def _judge(self, example: str, juror_assessments: list):

        # Create the judge prompt make final decision
        prompt = self._create_judge_template(
            example=example,
            juror_assessments=juror_assessments,
        )
        judge_response = await self._query_model(self._judge_model, prompt)
        return judge_response

    async def _juror(self, example: str):

        results = []

        for model in self._juror_models:
            # Create the juror prompt for the current model.
            # TODO: Do we want the same prompt for all the jurors?
            prompt = self._create_juror_template(example=example)

            response = await self._query_model(model=model, prompt=prompt)
            results.append(
                (response, response)
            )  # TODO: Fix this, expeted append is tuple of classification and reasoning

        return results

    def _parse_output(model_output):
        # We need someway to pase the final output from the judge. It needs to be just one word or a catrgory depending on which track
        # We can potentially use another smaller superfast llm for parsing? How long will it take to respond?
        # we can also use some combination of regex??? Since we know the words we are looking for!!!
        # we also need to get a list of expected responses.

        # TODO update the parsing logic here
        return model_output

    def _create_juror_template(self, example):

        # inject variables into the template
        juror_template = (
            JUROR_TEMPLATE.replace("{{possible_langs}}", str(self._possible_emotions))
            .replace("{{lang_id}}", self._example_lang)
            .replace("{{text}}", example)
        )

        return juror_template

    def _create_judge_template(self, example, juror_assessments):

        # Parse the assessments from the jurors
        formatted_assessments = "\n".join(
            f"Juror{value}\nResponse: {str(emotions)}\n"
            for value, emotions in enumerate(juror_assessments)
        )

        # inject assessments into judge template
        judge_template = (
            JUDGE_TEMPLATE.replace("{{juror_assessment}}", formatted_assessments)
            .replace("{{lang_id}}", self._example_lang)
            .replace("{{possible_emotions}}", str(self._possible_emotions))
            .replace("{{text}}", example)
        )

        return judge_template

    async def _query_model(self, model, prompt):

        response = await AsyncClient().chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        # Terminate the Model after getting responsem(This save memory)
        # we do not want numerous instances of different models open at the same time
        # GPU go brrr....
        subprocess.Popen(
            ["ollama", "stop", model],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Parse the raw string return from the LLM
        return response["message"]["content"]


### TEMP Global Vars #TODO: Clean this up maybe refactor to another file

JUROR_TEMPLATE = """

## Task: Perceived Emotion Detection:

You are an expert in natural language processing and sentiment analysis. 
Your task is to analyze the given text and identify the emotions conveyed within it. 
You can select multiple emotions if the text evokes a range of feelings.

The possible emotions are: {{possible_langs}}

The language of the given text is: {{lang_id}}

Given the instructions analyse the following text and determine the emotions perceived:
"{{text}}"


Please provide a brief explanation for your emotion(s) choice(s).
Note: The text can exhibit multiple emotions.
"""

JUDGE_TEMPLATE = """

## Task: Final Emotion Determination

**Review the Juror Assessments:**

Carefully review the emotion assessments provided by the Jurors. 
Pay attention to the range of emotions identified, the frequency of specific emotions, and the level of confidence expressed by each Juror.

**Consider the Following:**

1. **Consensus:**
   * Identify emotions that have been consistently selected by multiple Jurors.
   * Prioritize emotions with strong consensus.

2. **Confidence Levels:**
   * Assess the confidence levels expressed by the Jurors.
   * Give more weight to emotions that have been identified with high confidence.

3. **Nuance and Complexity:**
   * Consider the possibility of multiple emotions or complex emotional states.
   * Look for subtle cues and underlying feelings that may not be explicitly stated.

** For context:**
1. This is the sample text the jurors were asked to classify: "{{text}}".
2. The languge of the above text is {{lang_id}}.
3. The possible emotions invoked by the above text are: {{possible_emotions}}.

**Juror Assesments:**

{{juror_assessment}}

**Make a Final Decision:**

Based on your analysis, determine the primary emotion(s) conveyed in the text. 

Please only provide the final emotion(s) in your response. You do not need to explain your thought process.
"""


# ### For testing purposes:
# judge_model = "llama3.2:1b"

# jurors = ["phi:latest"]

# poss_emo = ["Anger", "Fear", "Joy", "Sadness", "Surprise"]

# example = "I hate this world"

# evaluator = LangEvalAlgo(
#     juror_models=jurors, judge_model=judge_model, possible_emotions=poss_emo
# )

# resp = asyncio.run(evaluator.run(example=example))
# print(resp)
