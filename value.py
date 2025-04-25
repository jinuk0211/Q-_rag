
from prompt import complete_query_from_ans, complete_query_from_subquery
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global variables for model caching
# VALUE_MODEL_DIR = "../hfmodels/Qwen/Qwen2.5-72B-Instruct"
VALUE_MODEL_DIR = "meta-llama/Llama-3.2-1B-Instruct"
global_value_model = None
global_tokenizer = None


def initialize_value_model():
    """Initialize the value model and tokenizer."""
    global global_value_model, global_tokenizer

    if global_value_model is not None and global_tokenizer is not None:
        return True  # Model already initialized

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(VALUE_MODEL_DIR)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            VALUE_MODEL_DIR,
            torch_dtype=torch.bfloat16,
            # device_map="auto",  # Automatically choose best device
        )

        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()

        global_value_model = model
        global_tokenizer = tokenizer

        print("Value model initialized successfully")
        return True

    except Exception as e:
        print(f"Error initializing value model: {str(e)}")
        return False


def cleanup_value_model():
    """Cleanup model resources."""
    global global_value_model, global_tokenizer

    if global_value_model is not None:
        del global_value_model
        global_value_model = None

    if global_tokenizer is not None:
        del global_tokenizer
        global_tokenizer = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Value model resources cleaned up")


def get_token_probabilities(text, idx, inputs=None):
    """
    Calculate log probabilities for tokens from idx onwards.
    Each probability p(d_t|d_<t) is conditioned only on previous tokens.

    Args:
        text (str): Input text sequence d
        idx (int): Starting index for probability calculation
        inputs (dict, optional): Pre-tokenized inputs, if None will tokenize text

    Returns:
        list: List of log probabilities [log p(d_t|d_<t)] for t >= idx
    """
    global global_value_model, global_tokenizer

    if (
        global_value_model is None or global_tokenizer is None
    ) and not initialize_value_model():
        return []

    try:
        # Use pre-tokenized inputs if provided, otherwise tokenize text
        if inputs is None:
            inputs = global_tokenizer(
                text, truncation=True, max_length=512, return_tensors="pt"
            )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Move to GPU if available
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        log_probs = []
        with torch.no_grad():
            # Get model outputs for the entire sequence
            outputs = global_value_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            logits = outputs.logits[0]  # Remove batch dimension

            # Calculate log probabilities for each position from idx
            for pos in range(
                idx - 1, input_ids.shape[1] - 1
            ):  # -1 because we predict next token
                # Get log probabilities for next token
                next_token_logits = logits[pos]
                log_probs_t = torch.log_softmax(next_token_logits, dim=-1)

                # Get log probability of the actual next token
                next_token_id = input_ids[0, pos + 1]
                log_prob = log_probs_t[next_token_id].item()
                log_probs.append(log_prob)

        return log_probs

    except Exception as e:
        print(f"Error calculating token probabilities: {str(e)}")
        return []



def get_query_token_probabilities(context, query):
    """
    获取query部分的token概率

    Args:
        context (str): 前文上下文
        query (str): 要计算概率的查询文本

    Returns:
        list: query部分token的log概率列表
    """
    global global_value_model, tokenizer

    if (
        global_value_model is None or global_tokenizer is None
    ) and not initialize_value_model():
        return []

    try:
        # 先对整个文本做tokenization
        full_text = context + query
        inputs = global_tokenizer(
            full_text, truncation=True, return_tensors="pt"
        )

        # 单独对前文做tokenization，找到query的起始位置
        context_tokens = global_tokenizer(
            context, padding=False, truncation=False, return_tensors="pt"
        )
        query_start_idx = context_tokens["input_ids"].shape[1]

        # 获取query部分的概率，传入已tokenized的inputs
        return get_token_probabilities(full_text, query_start_idx, inputs)

    except Exception as e:
        print(f"Error in get_query_token_probabilities: {str(e)}")
        return []


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
