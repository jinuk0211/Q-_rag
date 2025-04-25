        if self.node_type is Node_Type.USER_QUESTION:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                # 提交所有无依赖任务到线程池
                if not self.disable_a1:
                    do_action_generate_ost_step()
                if not self.disable_rag:
                    do_action_generate_rag_step()
                    do_action_generate_question_retrieve()
                # futures.append(executor.submit(do_action_generate_question_retrieve))
                do_action_generate_direct_answers()
                do_action_generate_subquestions()
                if not self.disable_a5:
                    do_action_generate_rephrased_user_question()
                
        elif self.node_type is Node_Type.REPHRASED_USER_QUESTION:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                # 提交所有无依赖任务到线程池
                if not self.disable_a1:
                    do_action_generate_ost_step()
                if not self.disable_rag:
                    do_action_generate_rag_step()
                do_action_generate_direct_answers()
                do_action_generate_subquestions()

        elif self.node_type is Node_Type.DIRECT_ANSWER:
            raise ValueError("DIRECT_ANSWER node cannot create children!!")

              
        elif self.node_type is Node_Type.RE_SUBANSWER:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                # 提交所有无依赖任务到线程池
                if not self.disable_a1:
                    do_action_generate_ost_step(True)
                if not self.disable_rag:
                    do_action_generate_rag_step(True)
                do_action_generate_direct_answers()
                do_action_generate_subquestions()
                
        elif self.node_type is Node_Type.OST_STEP:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                # 提交所有无依赖任务到线程池
                if not self.disable_rag:
                    do_action_generate_rag_step()
                if not self.disable_a1:
                    do_action_generate_ost_step()
                do_action_generate_direct_answers()
                

        assert self.children


def parse_args():
    parser = argparse.ArgumentParser(description="Run Naive RAG for various datasets and models.")

    # Dataset and split configuration
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        choices=['gpqa', 'math500', 'aime', 'amc', 'livecode', 'nq', 'triviaqa', 'hotpotqa', '2wiki', 'musique', 'bamboogle', 'medmcqa', 'pubhealth'],
        help="Name of the dataset to use."
    )

    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['test', 'diamond', 'main', 'extended'],
        help="Dataset split to use."
    )

    parser.add_argument(
        '--subset_num',
        type=int,
        default=None,
        help="Number of examples to process. Defaults to all if not specified."
    )

    # Search and document retrieval configuration
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help="Number of top search results to retrieve."
    )

    parser.add_argument(
        '--max_doc_len',
        type=int,
        default=3000,
        help="Maximum length of each searched document."
    )

    # Model configuration
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the pre-trained model."
    )

    parser.add_argument(
        '--use_jina',
        type=bool,
        default=True,
        help="Whether to use Jina API for document fetching."
    )

    parser.add_argument(
        '--jina_api_key',
        type=str,
        default='None',
        help="Your Jina API Key to Fetch URL Content."
    )

    # Sampling parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help="Sampling temperature."
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.8,
        help="Top-p sampling parameter."
    )

    parser.add_argument(
        '--top_k_sampling',
        type=int,
        default=20,
        help="Top-k sampling parameter."
    )

    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=None,
        help="Repetition penalty. If not set, defaults based on the model."
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=32768,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    # Bing API Configuration
    parser.add_argument(
        '--bing_subscription_key',
        type=str,
        required=True,
        help="Bing Search API subscription key."
    )

    parser.add_argument(
        '--bing_endpoint',
        type=str,
        default="https://api.bing.microsoft.com/v7.0/search",
        help="Bing Search API endpoint."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Extract arguments
    dataset_name = args.dataset_name
    split = args.split
    subset_num = args.subset_num
    top_k = args.top_k
    max_doc_len = args.max_doc_len
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k_sampling = args.top_k_sampling
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens
    bing_subscription_key = args.bing_subscription_key
    bing_endpoint = args.bing_endpoint
    use_jina = args.use_jina
    jina_api_key = args.jina_api_key

    # Set default repetition_penalty if not provided
    if repetition_penalty is None:
        repetition_penalty = 1.05 if 'qwq' in model_path.lower() else 1.0
    
    if args.jina_api_key == 'None':
        jina_api_key = None

    # Paths to datasets
    if dataset_name == 'livecode':
        data_path = f'./data/LiveCodeBench/{split}.json'
    elif dataset_name in ['math500', 'gpqa', 'aime', 'amc']:
        data_path = f'./data/{dataset_name.upper()}/{split}.json'
    else:
        data_path = f'./data/QA_Datasets/{dataset_name}.json'

    if 'qwq' in model_path.lower():
        if dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode']:
            output_dir = f'./outputs/{dataset_name}.qwq.naive_rag'
        else:
            output_dir = f'./outputs/runs.qa/{dataset_name}.qwq.naive_rag'
    else:
        model_short_name = model_path.split('/')[-1].lower().replace('-instruct', '')
        output_dir = f'./outputs/runs.baselines/{dataset_name}.{model_short_name}.naive_rag'
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------- Data Loading ----------------------
    with open(data_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        if subset_num is not None:
            data = data[:subset_num]

    for item in tqdm(data, desc="Searching"):
        question = item['Question']
        # Check if the question has already been searched and cached
        if question in search_cache:
            results = search_cache[question]
            # print(f"Using cached search results for question: {question}")
        else:
            if dataset_name == 'livecode':
                search_question = question[:500]
            else:
                search_question = question
            results = bing_web_search(search_question, bing_subscription_key, bing_endpoint, market='en-US', language='en')
            search_cache[question] = results            