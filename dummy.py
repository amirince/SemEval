import asyncio
from ollama import AsyncClient
import time
async def query_model(model, prompt):

    response = await AsyncClient().chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    print(response)

    

model = "llama3.2:3b"


prompt = """

Role: Examine the intentionality, ethical implications, and broader societal effects of the text's emotional expression.

Task: Perceived Emotion and Ethical Implication Detection:
You are an expert in philosophy, language, and ethics. Your task is to analyze the given text by identifying the emotions it conveys, but with a deeper focus on the ethical dimensions and potential societal effects of these emotions.

In addition to recognizing the emotions in the text, consider the following:

Intentionality: What might the author intend to communicate with these emotions?
Ethical Implications: Are the emotions expressed fair, just, and morally sound? Do they align with standards of ethical communication?
Broader Societal Impact: How might these emotions influence the broader social context or affect the audience's understanding?
The possible emotions are: {{possible_langs}}

The language of the given text is: {{lang_id}}

Given the instructions, analyze the following text and determine the emotions perceived: "{{text}}"

Ethical Considerations:
Please provide a rationale for each emotion identified, specifically focusing on:
How the emotion aligns with moral standards.
The possible impact this emotion could have on social fairness or bias.
Note: The text may evoke multiple emotions; please explore the broader ethical context of each.
"""


asyncio.run(query_model(model, prompt))
asyncio.run(query_model(model, prompt))
asyncio.run(query_model(model, prompt))
asyncio.run(query_model(model, prompt))