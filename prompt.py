critic_simplified = '''Given the problem, image, image description, and reasoning steps, evaluate whether the reasoning steps are sufficient to solve the problem.
If there is no correct answer, never assign a score higher than 9.
Output format must be: "Score: ...", where ... is the decimal score.
Do not include any additional explanation or analysis. Follow the output format exactly.'''

complete_query_from_ans = """Given intermediate answer containing the facts about the original question, which is unknown, your task is to infer what the orginal question might have been.
Output the most likely original question directly and nothing else.

Here are some examples:

Example 1:
Intermediate answer:
Muhammad Ali was 74 years old when he died.
Alan Turing was 41 years old when he died.
The original question might be:
Who lived longer, Muhammad Ali or Alan Turing?

Example 2:
Intermediate answer:
Craigslist was founded by Craig Newmark.
Craig Newmark was born on December 6, 1952.
The original question might be:
When was the founder of craigslist born?

Intermediate answer:
{answer}
The original question might be:
"""


complete_query_from_subquery = """Given sub-question derived from the original question, which is unknown, your task is to infer what the original question might have been.
Output the most likely original question directly and nothing else.

Here are some examples:

Example 1:
Sub-question:
How old was Muhammad Ali when he died?
How old was Alan Turing when he died?
The original question might be:
Who lived longer, Muhammad Ali or Alan Turing?

Example 2:
Sub-question:
Who was the mother of George Washington?
The original question might be:
Who was the maternal grandfather of George Washington?

Example 3:
Sub-question:
Who is the director of Jaws?
Where is Steven Spielberg from?
Who is the director of Casino Royale?
Where is Martin Campbell from?
The original question might be:
Are both the directors of Jaws and Casino Royale from the same country?

Sub-question:
{query}
The original question might be:
"""

#prompts/GPQA/decompose/decompose_prompt.txt
'''Given a question, please decompose it into sub-questions. For each sub-question, please answer it in one complete sentence, ending with "The answer is ". When the original question is answerable, please start the subquestion with "Now we can answer the question: <original question>".

Question 1: Who was the president in 1980 of the country that has Azad Kashmir?
Question 1.1: Which country contains Azad Kashmir?
Answer 1.1: The answer is: Pakistan.
Question 1.2: Who was the president of Pakistan in 1980?
Answer 1.2: The answer is: Muhammad Zia-ul-Haq.
Question 1.3: Now we can answer the question: Who was the president in 1980 of the country that has Azad Kashmir?
Answer 1.3: The answer is: Muhammad Zia-ul-Haq.'''

#prompts/GPQA/decompose/decompose_template.json
{
    "question_prefix": "Question 5: ",
    "subquestion_prefix": "Question 5.{}:",
    "overall_question_prefix": "Question 5.{}: Now we can answer the question: {}\n",
    "answer_prefix": "Answer 5.{}: ",
    "index": 5
}