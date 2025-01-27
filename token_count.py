import tiktoken
import pandas as pd
import os
# Choose the model tokenizer (e.g., gpt-4, gpt-3.5-turbo)
encoding = tiktoken.encoding_for_model("gpt-4o")

dataset_list = [
    "afr",
    "amh",
    "arq",
    "ary",
    "chn",
    "deu",
    "eng",
    "esp",
    "hau",
    "hin",
    "ibo",
    "ind",
    "jav",
    "kin",
    "mar",
    "orm",
    "pcm",
    "ptbr",
    "ptmz",
    "ron",
    "rus",
    "som",
    "sun",
    "swa",
    "swe",
    "tat",
    "tir",
    "ukr",
    "vmw",
    "xho",
    "yor",
    "zul",
]

paths = ["track_a", "track_b", "track_c"]

juror_tokens = 0


JUROR_TEMPLATE_1 = """

## Task: Cultural and Linguistic Emotion Analysis:

You are a cultural and linguistic expert specializing in analyzing emotions through the lens of language, cultural context, and sociolinguistic nuances.  
Your role is to identify and explain the emotions conveyed in the provided text while considering cultural nuances, idiomatic expressions, and the sociolinguistic factors that may influence emotional interpretation.  

### Instructions:  
1. Analyze the text for emotional content, considering how cultural context and language usage shape emotional expression.  
2. Identify emotions from the following list: "Anger", "Fear", "Joy", "Sadness", "Surprise".  
3. Note the language of the text: {{lang_id}}.  

### Text for Analysis:  
"The weekend didn't live up to my storm standards, I was all prepared to watch lots of movies and spend time with the new kitty and it barely rained on and off all weekend and unfortunately I was too sick to make it to beerfest, damn tabs really upset my stomach but I was grateful for ladies night and surely think it should be mandatory every weekend."  

### Deliverable:  
- Identify the emotions perceived in the text.  
- Provide a culturally sensitive explanation for each emotion identified, referencing idiomatic expressions or cultural factors where applicable.  
- Highlight any linguistic features (e.g., tone, word choice, syntax) that influenced your interpretation.  

**Note**: The text may convey multiple emotions. Your analysis should be thorough and context-sensitive.

"""

JUROR_TEMPLATE_2 = """

## Task: Emotional Perception and Psychological Impact Assessment

You are a trained expert in psychology and cognitive science, specializing in the analysis of emotional tone, psychological responses, and cognitive processes that shape human perception. Your role is to assess the emotional tone of the given text, identify the emotions it evokes, and offer insights grounded in psychological theory.

Key Instructions:
1. Analyze the emotional tone of the provided text, considering both overt and subtle cues.
2. Identify and categorize the emotions conveyed, drawing on established psychological frameworks (e.g., the basic emotions theory, cognitive appraisal theory).
3. Explain the cognitive and psychological mechanisms that contribute to the perception of each identified emotion.

Emotions to consider include: {{possible_langs}} (select all applicable emotions that fit the text).

The language of the given text is: {{lang_id}}

Given the above, please analyze the following text:
"{{text}}"

In your response, explain:
- Why you selected each emotion(s).
- How the psychological or cognitive processes underlying these emotions might manifest in the text.

**Note:** The text may evoke a range of emotions. Feel free to identify and explain multiple emotions where applicable.


"""

JUROR_TEMPLATE_3 = """

Task: Emotional and Pragmatic Response Analysis
You are an expert in communication, behavioral analysis, and natural language processing. Your task is to assess the emotional and pragmatic impact of the following text. Focus on how the language may influence the reader's emotions, behavioral responses, and overall interpretation.

Goals:
Identify the emotions conveyed by the text.
Evaluate how the text's language and tone might affect the reader's emotional state or behavior.
Consider implied meanings, subtext, and the potential impact of the text on the audience.
Emotions to consider:
{{possible_langs}}

Language of the given text:
{{lang_id}}

Given the instructions, carefully analyze the following text and determine the emotions it is likely to evoke:

"{{text}}"

Explanation:

Provide a brief rationale for your choice(s) of emotion(s).
Highlight any subtext or implied meanings that influence emotional perception.
If the text has a mix of emotions, explain the potential shifts or contrasts in how a reader might emotionally react.
Note: The text can reflect multiple emotions or conflicting emotional cues.
"""

JUROR_TEMPLATE_4 = """

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

# prompt1_enc = len(encoding.encode(JUROR_TEMPLATE_1))
# prompt2_enc = len(encoding.encode(JUROR_TEMPLATE_2))
# prompt3_enc = len(encoding.encode(JUROR_TEMPLATE_3))
# prompt4_enc = len(encoding.encode(JUROR_TEMPLATE_4))


# # public_data_test/track_a/test
# for path in paths:

#     for dataset in dataset_list:

#         file_path = f"public_data_test/{path}/test/{dataset}.csv"

#         if not os.path.exists(file_path):
#             print(f"{dataset} found")
#             continue
        
#         data = pd.read_csv(file_path)

#         for index, row in data.iterrows():
            
#             text = row["text"]

#             # Calculate the number of tokens
#             text_enc = encoding.encode(text)
            
#             juror_tokens += len(text_enc) + prompt1_enc
#             juror_tokens += len(text_enc) + prompt2_enc
#             juror_tokens += len(text_enc) + prompt3_enc
#             juror_tokens += len(text_enc) + prompt4_enc
            
# print("Input Juror_text = ")



output = """Analyze the given text and identify the perceived emotions, considering their intentionality, ethical implications, and broader societal effects.

The language used in the text appears to be Amharic. After analyzing the text, I have identified the following emotions:

1. Anger:
The text expresses a sense of anger or frustration towards something, but its exact object is unclear due to the complexities of the language.
Intentionality: The author might intend to communicate their dissatisfaction with a particular situation or policy.
Ethical Implications: This emotion could be seen as morally sound if it is directed towards unjust actions or policies. However, if it is expressed without context or justification, it may lead to unfair accusations or mob mentality.
Broader Societal Impact: This emotion could have a negative impact on social cohesion, especially if it is exploited by politicians or media outlets for their own gain.

2. Fear:
The text also conveys a sense of fear or apprehension, possibly related to the consequences of certain actions or policies.
Intentionality: The author might intend to warn others about potential dangers or risks associated with a particular decision.
Ethical Implications: This emotion is morally sound if it is directed towards legitimate threats and used to inform people's decisions. However, if it is used to manipulate or control others, it becomes unethical.
Broader Societal Impact: This emotion could have a significant impact on public policy, especially in areas such as national security or environmental protection.

3. Surprise:
The text expresses surprise or astonishment at certain developments or events.
Intentionality: The author might intend to express their shock or amazement at a particular situation.
Ethical Implications: This emotion is morally sound if it is used to highlight unexpected consequences or injustices. However, if it is used to obscure the truth or avoid accountability, it becomes unethical.
Broader Societal Impact: This emotion could have an impact on public perception and understanding of certain issues, potentially leading to a more nuanced discussion.

4. Joy:
The text does not convey any sense of joy or happiness, which may be surprising given the context of Amharic language typically used in emotional expressions.
Intentionality: It is unclear what the author intends to communicate through this emotion, as it seems out of place given the tone and subject matter.
Ethical Implications: The absence of joy raises questions about the author's intentions or perspective on certain issues. If they are expressing a lack of enthusiasm due to moral concerns, that might be justifiable; however, if it is simply an error, it could mislead readers.
Broader Societal Impact: This emotion could lead to misunderstandings or misinterpretations about the author's stance or message.

5. Sadness:
The text does not convey any sense of sadness or sorrow, which may also seem unusual given the subject matter.
Intentionality: Similar to joy, it is unclear what the author intends to communicate through this emotion, as it seems out of place given the tone and subject matter.
Ethical Implications: The absence of sadness raises questions about the author's intentions or perspective on certain issues. If they are expressing a lack of emotional investment due to moral concerns, that might be justifiable; however, if it is simply an error, it could mislead readers.
Broader Societal Impact: This emotion could lead to misunderstandings or misinterpretations about the author's stance or message.

In conclusion, the text expresses emotions of anger and fear, with a possible intention to warn others about potential dangers or risks. The absence of joy and sadness is puzzling, but may indicate that the author is expressing moral concerns rather than emotional investment."""


# print(len(output)*2)


# output_len = len(encoding.encode(output))

# print(output_len*2)


print(len(JUROR_TEMPLATE_1))