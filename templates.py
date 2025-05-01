"""This File contains the string templates used for prompt generation."""

JUROR_TEMPLATE_1 = """

## Task: Cultural and Linguistic Emotion Analysis:

You are a cultural and linguistic expert specializing in analyzing emotions through the lens of language, cultural context, and sociolinguistic nuances.  
Your role is to identify and explain the emotions conveyed in the provided text while considering cultural nuances, idiomatic expressions, and the sociolinguistic factors that may influence emotional interpretation.  

### Instructions:  
1. Analyze the text for emotional content, considering how cultural context and language usage shape emotional expression.  
2. Identify emotions from the following list: {{possible_langs}}.  
3. Note the language of the text: {{lang_id}}.  

### Text for Analysis:  
"{{text}}"  

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
