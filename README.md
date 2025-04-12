# Q-_rag
```python
        instruction = get_naive_rag_instruction(question, formatted_documents)

        # Construct dataset and model-specific prompts
        if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_openqa(question)

        elif dataset_name in ['math500', 'aime', 'amc']:
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_math(question)

        elif dataset_name == 'gpqa':
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
            elif 'llama' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(question)

        elif dataset_name == 'livecode':
            question_title = item.get('question_title', '')
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
            else:
                user_prompt = get_task_instruction_code(question)
        else:
            user_prompt = ""  # Default to empty if dataset not matched

        # Combine instruction and user prompt
        full_prompt = instruction + "\n\n" + user_prompt

        # Combine instruction and user prompt
        full_prompt = instruction + "\n\n" + user_prompt

        # Apply tokenizer and chat template
        prompt = [{"role": "user", "content": full_prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_prompts.append(prompt)

    # ---------------------- Generation ----------------------
    # Initialize the LLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
    )

    print("Generating answers with LLM...")

    # Set default max_tokens if not provided
    if max_tokens is None:
        if 'qwq' in model_path.lower():
            max_tokens = 20480
        else:
            max_tokens = 10240

    start_time = time.time()
    # Generate model outputs
    output_list = llm.generate(
        input_prompts, 
        sampling_params=SamplingParams(
            max_tokens=max_tokens, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k_sampling, 
            repetition_penalty=repetition_penalty,
        )
    )

    total_time = time.time() - start_time

    # ---------------------- Evaluation ----------------------
    print("Evaluating generated answers...")
    run_evaluation(
        filtered_data=data,
        input_list=input_prompts,
        output_list=output_list,
        dataset_name=dataset_name,
        output_dir=output_dir,
        total_time=total_time,
        split=split,
    )
def run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split, apply_backoff=False):
    if dataset_name == 'livecode':
        # Prepare samples and generations for codegen_metrics
        samples_list = []
        generations_list = []

        # Collect difficulty levels for per-domain metrics
        difficulties = []
        per_difficulty_count = {}
        num_valid_answer = 0

        for item, input_prompt, result in zip(filtered_data, input_list, output_list):
            if type(result) == str:
                item['Output'] = result
            else:
                item['Output'] = result.outputs[0].text
            difficulty = item.get("difficulty", "Unknown")
            difficulties.append(difficulty)
            # Track metrics per domain
            if difficulty not in per_difficulty_count.keys():
                per_difficulty_count[difficulty] = 0

            pred_code = extract_answer(item['Output'], mode='codegen')
            if pred_code != '':
                num_valid_answer += 1
                per_difficulty_count[difficulty] += 1
            # Assuming each item has 'input_output' with 'inputs' and 'outputs'
            public_test_cases = json.loads(item.get("public_test_cases", "{}"))

            inputs, outputs = [], []
            for case in public_test_cases:
                inputs.append(case["input"])
                outputs.append(case["output"])

            sample = {
                "input_output": json.dumps({
                    "inputs": inputs,
                    "outputs": outputs
                }),
            }

            samples_list.append(sample)
            generations_list.append([pred_code])
            item['Pred_Answer'] = pred_code
            item['Question'] = input_prompt


        # Call codegen_metrics with pass@1
        metrics, results, final_metadata = codegen_metrics(
            samples_list,
            generations_list,
            k_list=[1],  # Evaluate the top 1 generated result
            num_process_evaluate=2,   # Parallel evaluation
            timeout=10,  # Set timeout to 10 seconds
            debug=False,  # Enable debug mode
        )
        # print('samples_list', samples_list)
        # print('generations_list', generations_list)
        # print('metrics', metrics)

        # Extract pass@1
        pass_at_1 = metrics.get('pass@1', 0.0)
        detail_pass_at_1 = metrics['detail']['pass@1']

        for item, pass1, res, meta in zip(filtered_data, detail_pass_at_1.values(), results.values(), final_metadata):
            item['Metrics'] = {'pass@1': pass1}
            item['Results'] = res
            item['Final_metadata'] = meta

        # Initialize per-difficulty metrics
        difficulty_metrics = defaultdict(list)
        for idx, difficulty in enumerate(difficulties):
            pass1 = detail_pass_at_1[idx]
            difficulty_metrics[difficulty].append(pass1)

        # Compute overall pass@1
        overall_metrics = {
            'pass@1': pass_at_1,  # / num_valid_answer * len(input_list),
            'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
            'query_latency': f'{(total_time / len(input_list) * 1000):.0f} ms',
        }

        # Compute per-difficulty pass@1
        per_difficulty_metrics = {}
        for difficulty, passes in difficulty_metrics.items():
            avg_pass = np.mean(passes) if len(passes) > 0 else 0.0
            num_valid_answer = per_difficulty_count[difficulty]
            per_difficulty_metrics[difficulty] = {
                'pass@1': avg_pass,
                'num_valid_answer': f'{num_valid_answer} of {len(passes)}'
            }

        # Save the metrics
        final_metrics = {
            'overall': overall_metrics,
            'per_domain': per_difficulty_metrics
        }

    else:
        # Existing evaluation for other datasets
        avg_em, avg_acc, avg_f1, avg_math = [], [], [], []
        num_valid_answer = 0

        # If the dataset is GPQA, track metrics per domain
        domain_metrics = {}

        for item, input_prompt, result in zip(filtered_data, input_list, output_list):
            if type(result) == str:
                item['Output'] = result
            else:
                item['Output'] = result.outputs[0].text
            if dataset_name in ['gpqa', 'medmcqa']:
                labeled_answer = item["Correct Choice"]
                # labeled_choice_answer = item["Correct Answer"]
                mode = 'choose'
            elif dataset_name in ['math500', 'aime', 'amc']:
                labeled_answer = item["answer"]
                mode = 'gen'
            elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
                labeled_answer = item["answer"]
                mode = 'qa'
            elif dataset_name in ['pubhealth']:
                labeled_answer = item["answer"]
                mode = 'choose'
            else:
                raise ValueError(f"Unknown dataset_name: {dataset_name}")

            metric, pred_answer = evaluate_predictions(output=item['Output'], labeled_answer=labeled_answer, mode=mode)
            item['Pred_Answer'] = pred_answer
            item['Metrics'] = metric
            item['Question'] = input_prompt

            # Determine the validity of the predicted answer
            my_method_valid = (pred_answer != '' and not (mode == 'choose' and dataset_name == 'gpqa' and len(pred_answer) > 1))

            avg_em.append(metric['em'])
            avg_acc.append(metric['acc'])
            avg_f1.append(metric['f1'])
            avg_math.append(metric['math_equal'])

            if my_method_valid:
                num_valid_answer += 1

            # If the dataset is GPQA, attempt to track metrics per domain
            if dataset_name == 'gpqa':
                domain = item.get("High-level domain", "Unknown")
                if domain not in domain_metrics:
                    domain_metrics[domain] = {'em': [], 'acc': [], 'f1': [], 'math_equal': [], 'num_valid_answer': 0, 'total_num': 0}
                domain_metrics[domain]['total_num'] += 1
                domain_metrics[domain]['em'].append(metric['em'])
                domain_metrics[domain]['acc'].append(metric['acc'])
                domain_metrics[domain]['f1'].append(metric['f1'])
                domain_metrics[domain]['math_equal'].append(metric['math_equal'])
                if my_method_valid:
                    domain_metrics[domain]['num_valid_answer'] += 1

        t = time.localtime()
        result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
        metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json'

        # Compute overall metrics
        overall_results = {
            'em': np.mean(avg_em) if len(avg_em) > 0 else 0.0,
            'acc': np.mean(avg_acc) if len(avg_acc) > 0 else 0.0,
            'f1': np.mean(avg_f1) if len(avg_f1) > 0 else 0.0,
            'math_equal': np.mean(avg_math) if len(avg_em) > 0 else 0.0,
            'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
            'query_latency': f'{(total_time / len(input_list) * 1000):.0f} ms',
        }

        # If the dataset is GPQA, output average metrics per domain
        domain_avg_metrics = {}
        if dataset_name == 'gpqa':
            for dm, m in domain_metrics.items():
                domain_avg_metrics[dm] = {
                    'em': np.mean(m['em']) if len(m['em']) > 0 else 0,
                    'acc': np.mean(m['acc']) if len(m['acc']) > 0 else 0,
                    'f1': np.mean(m['f1']) if len(m['f1']) > 0 else 0,
                    'math_equal': np.mean(m['math_equal']) if len(m['math_equal']) > 0 else 0,
                    'num_valid_answer': f'{m["num_valid_answer"]} of {m["total_num"]}'
                }

        # 保存总体和分domain的指标
        final_metrics = {'overall': overall_results}
        if dataset_name == 'gpqa':
            final_metrics['per_domain'] = domain_avg_metrics

    t = time.localtime()
    result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
    metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json'
    if apply_backoff:
        result_json_name = output_dir
        metrics_json_name = output_dir.replace('.json', '.metrics.backoff.json')

    # Save prediction results and metrics
    with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, metrics_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)
```
