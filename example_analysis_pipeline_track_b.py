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

    def __init__(
        self,
        juror_model: str,
        judge_model: str,
        possible_emotions: list,
    ):
        self._judge_model = judge_model
        self._possible_emotions = possible_emotions
        self.juror_model = juror_model
        self.lang_classifier = TextClassify()

    async def run_jurors(self, example):
        # This stores the results from all the jurors
        juror_results = []

        #  Get the language of current example
        self._example_lang = self.lang_classifier.classify(example)

        juror_results.append(self._example_lang)

        juror1 = await self._juror(example=example, template=JUROR_TEMPLATE_1)
        print("Juror 1 done")
        juror2 = await self._juror(example=example, template=JUROR_TEMPLATE_2)
        print("Juror 2 done")
        juror3 = await self._juror(example=example, template=JUROR_TEMPLATE_3)
        print("Juror 3 done")
        juror4 = await self._juror(example=example, template=JUROR_TEMPLATE_4)
        print("Juror 4 done")

        juror_results.append(juror1)
        juror_results.append(juror2)
        juror_results.append(juror3)
        juror_results.append(juror4)

        return juror_results

    async def _judge(self, example: str, juror_assessments: list):

        # Create the judge prompt make final decision
        prompt = self._create_judge_template(
            example=example,
            juror_assessments=juror_assessments,
        )
        judge_response = await self._query_model(self._judge_model, prompt)
        return judge_response

    async def _juror(self, example: str, template: str):
        """This is one instance of a juror."""

        # Create a juror template using one of the four persona
        prompt = self._create_juror_template(example=example, template=template)

        # Query the model example injected into template
        response = await self._query_model(model=self.juror_model, prompt=prompt)

        return response

    def _create_juror_template(self, example, template):

        # inject variables into the template
        juror_template = (
            template.replace("{{possible_langs}}", str(self._possible_emotions))
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

        return response["message"]["content"]


JUROR_TEMPLATE_1 = """

Task: Emotional Intensity Analysis with Cultural Sensitivity

You are a cultural and linguistic expert specializing in analyzing emotional intensity within language. Your role is to evaluate and explain the emotional intensity conveyed in the provided text, considering cultural nuances, idiomatic expressions, and sociolinguistic factors that may influence emotional perception.

## Instructions:
Analyze the text for emotional intensity, considering how cultural and linguistic contexts influence the expression of emotions.
Evaluate the intensity of the following emotions:
- Anger
- Fear
- Joy
- Sadness
- Surprise
### Rate the intensity of each emotion on the following scale:
0: No emotion
1: Low degree of emotion
2: Moderate degree of emotion
3: High degree of emotion
Note the language of the text: {{lang_id}}.
Text for Analysis:
"{{text}}"

Deliverable:
Provide a rating (0–3) for the intensity of each emotion in the list.
Explain your ratings for each emotion, referencing cultural factors, idiomatic expressions, or linguistic features (e.g., tone, syntax, word choice) that contributed to your evaluation.
Highlight any sociolinguistic nuances that influenced your analysis.
Note: The text may exhibit varying levels of intensity for different emotions. Your analysis should be detailed, context-aware, and culturally sensitive.

"""

JUROR_TEMPLATE_2 = """

You are a trained expert in psychology and cognitive science, specializing in the analysis of emotional tone, psychological responses, and cognitive processes that shape human perception. Your role is to assess the intensity of the emotions evoked in the given text, categorizing each emotion according to its degree of presence.

Instructions:
Analyze the emotional tone of the provided text, considering both overt and subtle cues.
Identify and categorize the emotions conveyed using the following scale:
0: No emotion
1: Low degree of emotion
2: Moderate degree of emotion
3: High degree of emotion
For each emotion, assess the intensity based on the cues provided in the text and assign the appropriate value from the scale (0-3).
Emotions to consider include:

Anger
Fear
Joy
Sadness
Surprise
The language of the given text is: {{lang_id}}

Given the above, please analyze the following text: "{{text}}"

In your response, explain:

The intensity score assigned to each emotion.
The psychological or cognitive mechanisms that contribute to the intensity of each emotion in the text.
Note: The text may evoke varying intensities of multiple emotions. Feel free to assign multiple emotions with corresponding intensity scores where applicable.


"""

JUROR_TEMPLATE_3 = """

Task: Emotional Intensity Analysis

You are an expert in communication, behavioral analysis, and natural language processing. Your task is to assess the emotional intensity present in the following text. Focus on determining the degree of emotional impact conveyed by the language, and evaluate how strongly the text evokes each emotion on the given scale.

Instructions: For each of the following emotions, assess the intensity on a scale from 0 to 3:

Emotions:

Anger
Fear
Joy
Sadness
Surprise
Scale:
0: No emotion
1: Low degree of emotion
2: Moderate degree of emotion
3: High degree of emotion

Text for Analysis:
"{{text}}"

Provide your evaluation in the following format:

Anger: [0, 1, 2, 3]
Fear: [0, 1, 2, 3]
Joy: [0, 1, 2, 3]
Sadness: [0, 1, 2, 3]
Surprise: [0, 1, 2, 3]
Explanation:

Provide a brief rationale for your assessment of each emotion's intensity.
Identify any nuances or subtext that may influence how the reader emotionally interprets the text.
If the text expresses a blend of emotions, explain how the intensity levels may shift or contrast.
"""

JUROR_TEMPLATE_4 = """

Role: Examine the intentionality, ethical implications, and broader societal effects of the text's emotional expression, while assessing the intensity of each emotion conveyed.

Task: Emotional Intensity and Ethical Implication Detection

You are an expert in philosophy, language, and ethics. Your task is to analyze the given text by identifying the emotions it conveys and evaluating the intensity of each emotion on the following scale:

Emotions to Consider:
- Anger
- Fear
- Joy
- Sadness
- Surprise

Scale for Intensity: 0: No emotion
1: Low degree of emotion
2: Moderate degree of emotion
3: High degree of emotion

Instructions:

Identify the emotional intensity for each emotion listed above based on the scale provided (0–3).
Consider the intentionality behind each emotion: What does the author aim to communicate with these emotions?
Explore the ethical implications of these emotions: Are they fair, just, and morally sound? Do they uphold standards of ethical communication?
Analyze the broader societal impact: How might these emotions influence the broader social context or shape the audience's perception?
Text for Analysis: "{{text}}"

Provide your analysis in the following format:

Anger: [0, 1, 2, 3]
Fear: [0, 1, 2, 3]
Joy: [0, 1, 2, 3]
Sadness: [0, 1, 2, 3]
Surprise: [0, 1, 2, 3]
Ethical Considerations:
For each emotion identified, provide a rationale focused on:

How the emotion aligns with moral standards (fairness, justice, etc.)
The possible impact this emotion could have on social fairness or bias
Note: The text may evoke multiple emotions with varying degrees of intensity. Explore the broader ethical context of each emotion and how its expression may affect the audience.
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
