import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.prompts import complete_query_from_ans, complete_query_from_subquery
from utils.value_model import get_query_token_probabilities

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

def similarity_value(ori_query, query, answer, ans_weight=0.7):
    """
    Calculate weighted combination of:
    1. TF-IDF similarity between query-answer pair and original question
    2. TF-IDF similarity between knowledge and query to assess knowledge reliability

    Returns:
        float: Combined similarity score (smoothly mapped to [0,1])
    """
    try:
        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer()

        # Calculate query-question similarity
        query_matrix = vectorizer.fit_transform([query, ori_query])
        query_similarity = cosine_similarity(query_matrix[0:1], query_matrix[1:2])[0][0]

        # Calculate knowledge-query similarity if knowledge exists
        if answer:
            # Reinitialize vectorizer for knowledge similarity
            vectorizer = TfidfVectorizer()
            answer_matrix = vectorizer.fit_transform([answer, ori_query])
            answer_similarity = cosine_similarity(
                answer_matrix[0:1], answer_matrix[1:2]
            )[0][0]

            value = (1 - ans_weight) * query_similarity + ans_weight * answer_similarity
        else:
            value = query_similarity

        return float(value)

    except Exception as e:
        print(f"Error in similarity calculation: {str(e)}")
        return 0.0


def risk_value(ori_query, query, answer, ans_weight=0.75):
    """
    Calculate average log probability of tokens in the sequence.
    Formula: (1/|d|) * sum(log p(d_t|d_<t))

    Args:
        ori_query (str): Original query
        query (str): Current query
        answer (str): Answer text
        ans_weight (float): Weight for answer probability

    Returns:
        float: Average log probability score
    """
    try:
        # 计算answer条件下的原始query概率
        kl_ans_text_front = complete_query_from_ans.format(answer=answer)
        kl_ans_probs = get_query_token_probabilities(kl_ans_text_front, ori_query)
        if not kl_ans_probs:
            return 0.0
        kl_ans = -sum(kl_ans_probs) / len(kl_ans_probs)

        # 计算decomposed query条件下的原始query概率
        kl_dcp_text_front = complete_query_from_subquery.format(query=query)
        kl_dcp_probs = get_query_token_probabilities(kl_dcp_text_front, ori_query)
        if not kl_dcp_probs:
            return 0.0
        kl_dcp = -sum(kl_dcp_probs) / len(kl_dcp_probs)

        # 计算加权平均
        kl_loss = (1 - ans_weight) * kl_dcp + ans_weight * kl_ans

        # 映射到[0,1]区间
        value = np.exp(-1.8 * (kl_loss - 1.8))
        value = 1 - (1 / (1 + value))

        return float(value)

    except Exception as e:
        logging.error(f"Error in risk value calculation: {str(e)}")
        return 0.0


if __name__ == "__main__":
    print(risk_value("What is 2+2+2?", "What is 2+2?", "2+2=4"))