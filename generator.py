
class Generator:
    """Generator generates children nodes"""

    def __init__(self, args, tokenizer, model, evaluator: Evaluator) -> None:
        self.io = IO_System(args, tokenizer, model)
        self.evaluator = evaluator
        if not args.disable_rag:
            self.retriever = Retriever()
            self.retriever.regist_io_system(self.io)

        self.num_subquestions = args.num_subquestions
        self.num_a1_steps = args.num_a1_steps
        self.num_votes = args.num_votes
        self.max_tokens = args.max_tokens
        self.enable_potential_score = args.enable_potential_score

        self.mcts_num_last_votes = args.mcts_num_last_votes

        with open(args.decompose_template_path, "r") as f:
            decompose_template = json.load(f)
            self.question_index = decompose_template["index"]

        self.decompose_prompt = read_txt(args.decompose_prompt_path)
        self.fewshot_cot_prompt = read_txt(args.fewshot_cot_prompt_path)
        self.fewshot_cot_config = read_json(args.fewshot_cot_config_path)

        self.fewshot_ost_prompt = read_txt(args.fewshot_ost_prompt_path)
        self.fewshot_ost_config = read_json(args.fewshot_ost_config_path)

        if not args.disable_a5:  # A5: Rephrase the question/sub-question.
            self.rephrasing_prompt_template = read_txt(args.rephrasing_prompt_template_path)
            self.decompose_prompt_rephrased = read_txt(args.decompose_prompt_rephrased_path)
            self.fewshot_cot_prompt_rephrased = read_txt(args.fewshot_cot_prompt_rephrased_path)
            self.fewshot_ost_prompt_rephrased = read_txt(args.fewshot_ost_prompt_rephrased_path)

    def _extract_from_cache(self, subquestion_list: List[str]):
        high_score_questions = []
        selected_answers = []
        values = []
        low_score_questions = []
        low_score_values = []
        low_score_answers_list = []
        unmatched_questions = []

        for subquestion in subquestion_list:
            best_match = process.extractOne(subquestion, self.reasoning_cache.keys(), scorer=fuzz.ratio)

            if best_match:
                best_question, best_score = best_match[0], best_match[1]
                similarity = best_score / 100
                cache_entry = self.reasoning_cache[best_question]
                score = cache_entry["score"]
                if similarity == 1:
                    if score >= 0.9:
                        high_score_questions.append(best_question)
                        selected_answers.append(cache_entry["selected_answer"])
                        values.append(score)
                    else:
                        low_score_questions.append(best_question)
                        low_score_values.append(score)
                        low_score_answers_list.append(cache_entry["answer_list"])
                else:
                    unmatched_questions.append(subquestion)
            else:
                unmatched_questions.append(subquestion)

        return {
            "high_score_questions": high_score_questions,
            "selected_answers": selected_answers,  # most likely answer corresponding to each subquestion
            "values": values,
            "low_score_questions": low_score_questions,
            "low_score_values": low_score_values,
            "low_score_answers_list": low_score_answers_list,
            "unmatched_questions": unmatched_questions,
        }

    def _get_most_likely_answer(self, io_output_list: List[str]) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        if len(io_output_list) == 1:
            most_confident_answer_full_completion = io_output_list[0]
            confidence = 1
        else:
            _, most_confident_answer_full_completion, _, confidence = self.evaluator.find_most_confident_answer(
                io_output_list
            )
            assert confidence > 0

        return most_confident_answer_full_completion, confidence

    def _fewshot_cot_answer_question(self, question: str, paraphrased: bool, num_return: int, hint: str = None):
        fewshot_cot_prompt = self.fewshot_cot_prompt if not paraphrased else self.fewshot_cot_prompt_rephrased
        question += "\n\n" + hint if hint is not None else ""
        io_input = self.fewshot_cot_config["prompt_template"].format(examples=fewshot_cot_prompt, instruction=question)
"prompt_template": "A chat between a curious user and an AI assistant. The assistant gives step-by-step solutions to the user's questions. You are presented with observations or results related to a phenomenon. Based on the information provided, infer the possible reasons or explanations for the observed outcomes. In the end of assistant's response, a final answer must be given in the format of \"The answer is: <ANSWER>.\", where <ANSWER> should only be \"A\", \"B\", \"C\" or \"D\" without any description.\n\n{examples}\n\n### Instruction:\n{instruction}\n\n### Response:\n\nPlease answer it in a complete sentence",

        io_output_list = self.io.generate(
            io_input,
            num_return=num_return,
            max_tokens=self.max_tokens,
            stop_tokens=self.fewshot_cot_config["stop_tokens"],
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]  #! cleaning
        return io_input, cleaned_io_output_list

    def _fewshot_cot_answer_question_with_external_knowledge(self, question: str, external_knowledge: str, paraphrased: bool, num_return: int, hint: str = None):
        fewshot_cot_prompt = self.fewshot_cot_prompt if not paraphrased else self.fewshot_cot_prompt_rephrased
        question += "\n\n" + hint if hint is not None else ""
        io_input = self.fewshot_cot_config["prompt_template"].format(examples=fewshot_cot_prompt, instruction=question)
        io_input += (
               f"Context: {external_knowledge}"
            )
        io_output_list = self.io.generate(
            io_input,
            num_return=num_return,
            max_tokens=self.max_tokens,
            stop_tokens=self.fewshot_cot_config["stop_tokens"],
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]  #! cleaning
        return io_input, cleaned_io_output_list

    def generate_direct_answers(self, user_question: str, paraphrased: bool, hint: str):
        direct_answer_list, value_list = [], []

        #! few shot cot
        num_return = self.mcts_num_last_votes
        io_input, cleaned_io_output_list = self._fewshot_cot_answer_question(
            question=user_question, paraphrased=paraphrased, num_return=num_return, hint=hint
        )

        try:
            most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_list)
        except Exception as e:
            raise GeneratorError(
                source="generate direct answer from: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )

        direct_answer_list.append(most_likely_answer)
        value_list.append(likelihood)

        return direct_answer_list, value_list
    def _get_most_likely_answer(self, io_output_list: List[str]) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        if len(io_output_list) == 1:
            most_confident_answer_full_completion = io_output_list[0]
            confidence = 1
        else:
            _, most_confident_answer_full_completion, _, confidence = self.evaluator.find_most_confident_answer(
                io_output_list
            )
            assert confidence > 0

        return most_confident_answer_full_completion, confidence
    def find_most_confident_answer(self, completions: List[str], prior_weights: List[float] = None):
        """Returns the most confident answer, its completion, its id in the input list, and its confidence."""
        if completions is None or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        answer2ids = defaultdict(list)
        for id, c in enumerate(completions):
            try:
                model_answer = self.extract_answer_from_model_completion(c)
                has_existed = False
                for existing_answer in answer2completions.keys():
                    if self.check_answers_equiv(model_answer, existing_answer):
                        assert not has_existed
                        has_existed = True
                        answer2completions[existing_answer].append(c)
                        answer2ids[existing_answer].append(id)
                if not has_existed:
                    answer2completions[model_answer].append(c)
                    answer2ids[model_answer].append(id)
            except:
                pass

        assert len(answer2completions.keys()) > 0, "There are no valid completions."
        if prior_weights is not None:
            assert len(completions) == len(prior_weights)
            completion2count = {}
            for answer, answer_completions in answer2completions.items():
                count = len(answer_completions)
                for answer_completion in answer_completions:
                    completion2count[answer_completion] = count

            completion2score = {}
            for id, (completion, count) in enumerate(completion2count.items()):
                prior_weight = prior_weights[id]
                score = prior_weight * (count / len(completions))
                completion2score[completion] = score

            most_confident_completion = max(completion2score.keys(), key=lambda x: completion2score[x])

            return (
                self.extract_answer_from_model_completion(most_confident_completion),
                most_confident_completion,
                completions.index(most_confident_completion),
                completion2score[most_confident_completion],
            )
        else:
            most_confident_answer = max(answer2completions.keys(), key=lambda x: len(answer2completions[x]))
            assert (
                len(answer2completions[most_confident_answer]) > 0
            ), "There are no completions for the most confident answer."
            confidence = len(answer2completions[most_confident_answer]) / len(completions)
            assert confidence > 0
            return (
                most_confident_answer,
                answer2completions[most_confident_answer][0],
                answer2ids[most_confident_answer][0],
                confidence,
            )
    def generate_subquestions(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
    ):
        subquestion_list, subanswer_list, value_list = [], [], []
        decompose_prompt = self.decompose_prompt if not paraphrased else self.decompose_prompt_rephrased

        #! generate subquestions
        existing_subquestions_and_subanswers, next_subquestion_id = concat_subqs_and_subas(
            solution_trace, self.question_index
        )

        io_input = (
            decompose_prompt
            + "\n\n"
            + f"Question {self.question_index}: {user_question}"
            + "\n"
            + existing_subquestions_and_subanswers
            + "\n"
            + f"The text you generate must start with the string of subquestion index Question {self.question_index}.{next_subquestion_id}:."
        )

        io_output_list = self.io.generate(
            io_input,
            max_tokens=128,
            num_return=self.num_subquestions,
            stop_tokens=[
                "Answer",
                "\n",
                "The answer",
                f"Answer {self.question_index}.{next_subquestion_id}",
                f"Answer {self.question_index}.{next_subquestion_id}:",
                f"Answer {self.question_index}.{next_subquestion_id}: ",
            ],
        )

        # subquestion_list = [io_output.split("?")[0] + "?" for io_output in io_output_list]  # cleaning, you might wanna modify this
        subquestion_list = list(set([o.strip() for o in io_output_list if o.startswith(f"Question {self.question_index}.{next_subquestion_id}:")]))
        if len(subquestion_list) < 1:
            subquestion_list = list(set([o.strip() for o in io_output_list]))
        print(f"subquestion list: {subquestion_list}")


        #! generate subanswers to the subquestions generated above
        io_input_list = []
        for subquestion in subquestion_list:
            io_input = (
                decompose_prompt
                + "\n\n"
                + f"Question {self.question_index}: {user_question}"
                + "\n"
                + existing_subquestions_and_subanswers
                + f"Question {self.question_index}.{next_subquestion_id}: "
                + subquestion
                + "\n"
                + f"Please use one complete sentence to answer the question: {self.question_index}.{next_subquestion_id}."
            )
            io_input_list.append(io_input)

        if reach_terminal_subquestion(subquestion=subquestion, user_question=user_question):
            num_return = self.mcts_num_last_votes
        else:
            num_return = self.num_votes

        io_output_list = self.io.generate(
            io_input_list,
            max_tokens=512,
            num_return=num_return,
            stop_tokens=['\n\n\n',
                f"Question {self.question_index}.{next_subquestion_id + 1}",
            ],
        )
        cleaned_io_output_list = [
            [io_output.strip() for io_output in io_output_group] for io_output_group in io_output_list
        ]

        for i, cleaned_io_output_group in enumerate(cleaned_io_output_list):
            try:
                most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_group)
            except Exception as e:
                raise GeneratorError(
                    source="generate answer to subquestions",
                    io_input=io_input_list[i],
                    io_output_list=cleaned_io_output_group,
                )
            subanswer_list.append(most_likely_answer)
            value_list.append(likelihood)

        assert len(subquestion_list) == len(subanswer_list) == len(value_list)

        print(f"subquestion answer: {subanswer_list}")

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []
        if self.enable_potential_score:
            for subq, suba in zip(subquestion_list, subanswer_list):
                if reach_terminal_subquestion(subq, user_question):
                    potential_answers_list.append(None)
                else:
                    response_prefix = make_response_prefix(
                        solution_trace, Node_Type.SUBQUESTION, new_subq=subq, new_suba=suba
                    )

                    potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix

                    potential_score_output = self.io.generate(
                        potential_score_input,
                        num_return=self.num_votes,
                        max_tokens=128,
                        stop_tokens=self.fewshot_cot_config["stop_tokens"],
                    )
                    potential_score_input2 = [
                        "Question: "
                        + user_question
                        + "\nAnswer: "
                        + response_prefix
                        + z
                        + "\nTherefore, the answer (arabic numerals) is"
                        for z in potential_score_output
                    ]
                    cleaned_io_output_list = self.io.generate(
                        potential_score_input2,
                        num_return=1,
                        max_tokens=128,
                        stop_tokens=self.fewshot_cot_config["stop_tokens"],
                    )
                    cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                    potential_answers_list.append(
                        [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                    )
        else:
            potential_answers_list = [None] * len(subquestion_list)

        return subquestion_list, subanswer_list, value_list, potential_answers_list

    def generate_re_subanswers(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
    ):
        re_subanswer_list, value_list = [], []

        user_question_context = user_question

        last_subquestion_id = int(sorted(solution_trace.keys())[-1])
        last_subquestion = solution_trace[last_subquestion_id]["subquestion"]

        #! few shot cot
        question = (
            f"{user_question_context} {last_subquestion}"
            if not paraphrased
            else f"{user_question_context} Question: {last_subquestion}"
        )
        io_input, cleaned_io_output_list = self._fewshot_cot_answer_question(
            question=question, paraphrased=paraphrased, num_return=self.num_votes
        )
        try:
            most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_list)
        except Exception as e:
            raise GeneratorError(
                source="generate re-subanswers: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )
        re_subanswer_list.append(most_likely_answer)
        value_list.append(likelihood)

        print(f"re subanswer: {re_subanswer_list}")

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []
        if self.enable_potential_score:
            solution_trace_copy = deepcopy(solution_trace)
            for re_suba in re_subanswer_list:
                solution_trace_copy[last_subquestion_id]["subanswer"] = {"text": re_suba}
                response_prefix = make_response_prefix(solution_trace_copy, Node_Type.SUBQUESTION)
                potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix

                potential_score_output = self.io.generate(
                    potential_score_input,
                    num_return=self.num_votes,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                potential_score_input2 = [
                    "Question: "
                    + user_question
                    + "\nAnswer: "
                    + response_prefix
                    + z
                    + "\nTherefore, the answer is"
                    for z in potential_score_output
                ]
                cleaned_io_output_list = self.io.generate(
                    potential_score_input2,
                    num_return=1,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                potential_answers_list.append(
                    [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                )
        else:
            potential_answers_list = [None] * len(re_subanswer_list)

        return re_subanswer_list, value_list, potential_answers_list

    def generate_rag_and_re_subanswers(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
    ):
        re_subanswer_list, value_list = [], []

        user_question_context = user_question

        last_subquestion_id = int(sorted(solution_trace.keys())[-1])
        last_subquestion = solution_trace[last_subquestion_id]["subquestion"]

        #! few shot cot
        question = (
            f"{user_question_context}\n\n{last_subquestion}"
            if not paraphrased
            else f"{user_question_context} Question: {last_subquestion}"
        )

        print(f"rag subquestion 1: {question}")

        retrieved_context = self.retriever.retrieve(question)

        question = (
            f"{user_question_context} {last_subquestion}\n\n### Relevant Context:\n{retrieved_context}."
            if not paraphrased
            else f"{user_question_context} Question: {last_subquestion}"
        )
        print(f"rag subquestion 2: {question}")

        io_input, cleaned_io_output_list = self._fewshot_cot_answer_question(
            question=question, paraphrased=paraphrased, num_return=self.num_votes
        )
        try:
            most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_list)
            most_likely_answer = [f"{answer.strip().strip('\n')}\n\n### Relevant Context: {retrieved_context}\n" for answer in most_likely_answer]
        except Exception as e:
            raise GeneratorError(
                source="generate re-subanswers: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )
        re_subanswer_list.append(most_likely_answer)
        value_list.append(likelihood)

        print(f"rag subq answer {re_subanswer_list}")

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []
        if self.enable_potential_score:
            solution_trace_copy = deepcopy(solution_trace)
            for re_suba in re_subanswer_list:
                solution_trace_copy[last_subquestion_id]["subanswer"] = {"text": re_suba}
                response_prefix = make_response_prefix(solution_trace_copy, Node_Type.SUBQUESTION)
                potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix

                potential_score_output = self.io.generate(
                    potential_score_input,
                    num_return=self.num_votes,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                potential_score_input2 = [
                    "Question: "
                    + user_question
                    + "\nAnswer: "
                    + response_prefix
                    + z
                    + "\nTherefore, the answer (arabic numerals) is"
                    for z in potential_score_output
                ]
                cleaned_io_output_list = self.io.generate(
                    potential_score_input2,
                    num_return=1,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                potential_answers_list.append(
                    [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                )
        else:
            potential_answers_list = [None] * len(re_subanswer_list)

        return re_subanswer_list, value_list, potential_answers_list

    def generate_rephrased_user_question(self, user_question: str):
        rephrased_user_question_list = []
        io_input = self.rephrasing_prompt_template
        io_input += "\n\n"
        io_input += "Rephrase Original Question: " + user_question + "\n"
        io_input += "Rephrased question you generate should start with Given a list of conditions, please answer the question. Condition 1:, and it should be one line"
        io_output = self.io.generate(model_input=io_input, max_tokens=512, num_return=1, stop_tokens=[])[0]
        io_output = "Given a list of conditions, please answer the question: " + user_question + " Condition 1:" + io_output.split("Condition 1:")[-1] if "Condition 1:" in io_output else "Given a list of conditions, please answer the question. Condition 1: " + io_output
        rephrased_user_question_list.append(io_output)

        print(f"Rephrased user question is: {rephrased_user_question_list}")

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  # essentially direct answer list
        if self.enable_potential_score:
            response_prefix = make_response_prefix(None, None)
            potential_score_input = "Question: " + rephrased_user_question_list[0] + "\nAnswer: " + response_prefix
            potential_score_output = self.io.generate(
                potential_score_input,
                num_return=self.num_votes,
                max_tokens=128,
                stop_tokens=self.fewshot_cot_config["stop_tokens"],
            )
            potential_score_input2 = [
                "Question: "
                + rephrased_user_question_list[0]
                + "\nAnswer: "
                + response_prefix
                + z
                + "\nTherefore, the answer (arabic numerals) is"
                for z in potential_score_output
            ]
            cleaned_io_output_list = self.io.generate(
                potential_score_input2, num_return=1, max_tokens=128, stop_tokens=self.fewshot_cot_config["stop_tokens"]
            )
            cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

            potential_answers_list.append(
                [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
            )
        else:
            potential_answers_list = [None] * len(rephrased_user_question_list)

        return rephrased_user_question_list, potential_answers_list

    def generate_user_question_retrieve(self, user_question: str):
        rephrased_user_question_list = []

        retrieved_context = self.retriever.retrieve(user_question)

        io_output = f"Given additional informations, please answer the question.\n### Relevant Context: {retrieved_context}\nUser Question: {user_question}." 
        rephrased_user_question_list.append(io_output)

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  # essentially direct answer list
        if self.enable_potential_score:
            response_prefix = make_response_prefix(None, None)
            potential_score_input = "Question: " + rephrased_user_question_list[0] + "\nAnswer: " + response_prefix
            potential_score_output = self.io.generate(
                potential_score_input,
                num_return=self.num_votes,
                max_tokens=128,
                stop_tokens=self.fewshot_cot_config["stop_tokens"],
            )
            potential_score_input2 = [
                "Question: "
                + rephrased_user_question_list[0]
                + "\nAnswer: "
                + response_prefix
                + z
                + "\nTherefore, the answer is"
                for z in potential_score_output
            ]
            cleaned_io_output_list = self.io.generate(
                potential_score_input2, num_return=1, max_tokens=128, stop_tokens=self.fewshot_cot_config["stop_tokens"]
            )
            cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

            potential_answers_list.append(
                [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
            )
        else:
            potential_answers_list = [None] * len(rephrased_user_question_list)

        return rephrased_user_question_list, potential_answers_list

    def generate_rag_step(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
        parent_is_subquestion: bool,
    ):
        ost_step_list = []
        if parent_is_subquestion:
            existing_ost_steps, next_ost_step_id = concat_subqs_subas_as_ost_steps(solution_trace)
        else:
            existing_ost_steps, next_ost_step_id = concat_ost_steps(solution_trace)
        
            if next_ost_step_id == 1:
                return self.generate_ost_step(user_question=user_question, solution_trace=solution_trace, paraphrased=paraphrased, parent_is_subquestion=parent_is_subquestion)

        retrieve_question = f"{user_question}\n\n{existing_ost_steps}"
        retrieved_context = self.retriever.retrieve(retrieve_question)

        io_input = (
            self.fewshot_ost_config["prompt_template"].format(
                examples="",
                instruction=user_question,
            )
            + existing_ost_steps
            + "\n"
            + f"### Relevant Context:\n{retrieved_context}\n\n" 
            + f"The text you generate must start with string of current step index Step {next_ost_step_id}:"
        )
        io_output_list = self.io.generate(
            model_input=io_input, max_tokens=256, num_return=self.num_a1_steps, stop_tokens=['\n\n\n', f'Step {next_ost_step_id+1}',str(next_ost_step_id+1)]
        )
        ost_step_list = list(set([f"{io_output.strip().strip('\n')}\n\n### Relevant Context: {retrieved_context}\n" for io_output in io_output_list if io_output.startswith(f"Step {next_ost_step_id}")]))
        if len(ost_step_list) < 1:
            ost_step_list = list(set([f"Step {next_ost_step_id}: {io_output.strip().strip('\n')}" for io_output in io_output_list]))
        print(f"rag step list {ost_step_list}")

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  # essentially direct answer list
        if self.enable_potential_score:
            for ost_step in ost_step_list:
                response_prefix = make_response_prefix(solution_trace, Node_Type.OST_STEP, new_ost_step=ost_step)

                potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix

                potential_score_output = self.io.generate(
                    potential_score_input,
                    num_return=self.num_votes,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                potential_score_input2 = [
                    "Question: "
                    + user_question
                    + "\nAnswer: "
                    + response_prefix
                    + z
                    + "\nTherefore, the answer is"
                    for z in potential_score_output
                ]
                cleaned_io_output_list = self.io.generate(
                    potential_score_input2,
                    num_return=1,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                potential_answers_list.append(
                    [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                )
        else:
            potential_answers_list = [None] * len(ost_step_list)

        return ost_step_list, potential_answers_list
    
    def generate_ost_step(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
        parent_is_subquestion: bool,
    ):
        ost_step_list = []
        if parent_is_subquestion:
            existing_ost_steps, next_ost_step_id = concat_subqs_subas_as_ost_steps(solution_trace)

        else:
            existing_ost_steps, next_ost_step_id = concat_ost_steps(solution_trace)        
        
        io_input = (
            self.fewshot_ost_config["prompt_template"].format(
                examples='',
                instruction=user_question,
            )
            + existing_ost_steps
            + '\n'
            + f"The text you generate must start with the string Step {next_ost_step_id}:\n"
        )

        io_output_list = self.io.generate(
            model_input=io_input, max_tokens=256, num_return=self.num_a1_steps, stop_tokens=[f"Step {next_ost_step_id+1}", "\n\n\n"]
        )

        ost_step_list = list(set([io_output.strip().strip('\n') for io_output in io_output_list if io_output.startswith(f"Step {next_ost_step_id}")]))
        if len(ost_step_list)<1:
            ost_step_list = list(set([f"Step {next_ost_step_id}: {io_output.strip().strip('\n')}" for io_output in io_output_list]))

        assert(len(ost_step_list)>0)

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  # essentially direct answer list
        if self.enable_potential_score:
            for ost_step in ost_step_list:
                response_prefix = make_response_prefix(solution_trace, Node_Type.OST_STEP, new_ost_step=ost_step)

                potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix

                potential_score_output = self.io.generate(
                    potential_score_input,
                    num_return=self.num_votes,
                    max_tokens=128,
                    stop_tokens=[str(next_ost_step_id+1)],
                )
                potential_score_input2 = [
                    "Question: "
                    + user_question
                    + "\nAnswer: "
                    + response_prefix
                    + z
                    + "\nTherefore, the answer (arabic numerals) is"
                    for z in potential_score_output
                ]
                cleaned_io_output_list = self.io.generate(
                    potential_score_input2,
                    num_return=1,
                    max_tokens=128,
                    stop_tokens=[str(next_ost_step_id+1)],
                )
                cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                potential_answers_list.append(
                    [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                )
        else:
            potential_answers_list = [None] * len(ost_step_list)

        return ost_step_list, potential_answers_list

    '''def generate_rag_step(self, user_question: str, paraphrased: bool, hint: str):
        #! generate retrieve query
        rag_prompt = (
            "Given the following question, generate a concise and effective search query to retrieve relevant knowledge:\n"
            f"Question {self.question_index}: {user_question}"
            "Query:"
        )

        io_output_list = self.io.generate(
            rag_prompt,
            max_tokens=128,
            num_return=self.num_subquestions,
            stop_tokens=[
                "\n",
                "\n\n",
                "Answer",
                "Answer ",
            ],
        )

        print("rag query is: " + ", ".join(io_output_list))
        #! do retrieve and get answer
        retrieved_documents = self.retriever.retrieve(io_output_list[0])

        direct_answer_list, value_list = [], []

        return retrieved_documents, value_list'''

    def generate_rag_subquestions(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
    ):
        #! generate retrieve query
        io_input = (
            decompose_prompt
            + "\n\n"
            + f"Question {self.question_index}: {user_question}"
            + "\n"
            + existing_subquestions_and_subanswers
            + f"Question {self.question_index}.{next_subquestion_id}:"
        )

        io_output_list = self.io.generate(
            io_input,
            max_tokens=128,
            num_return=self.num_subquestions,
            stop_tokens=[
                "Answer",
                "Answer ",
                f"Answer {self.question_index}.{next_subquestion_id}",
                f"Answer {self.question_index}.{next_subquestion_id}:",
                f"Answer {self.question_index}.{next_subquestion_id}: ",
            ],
        )

        #! do retrieve and get answer
        retrieved_documents = self.retriever.retrieve(io_output_list[0])

        #! merge into question

        #! use retrieved result to reanswer question



        subquestion_list, subanswer_list, value_list = [], [], []
        decompose_prompt = self.decompose_prompt if not paraphrased else self.decompose_prompt_rephrased

        #! generate subquestions
        existing_subquestions_and_subanswers, next_subquestion_id = concat_subqs_and_subas(
            solution_trace, self.question_index
        )
        io_input = (
            decompose_prompt
            + "\n\n"
            + f"Question {self.question_index}: {user_question}"
            + "\n"
            + existing_subquestions_and_subanswers
            + f"Question {self.question_index}.{next_subquestion_id}:"
        )
        io_output_list = self.io.generate(
            io_input,
            max_tokens=128,
            num_return=self.num_subquestions,
            stop_tokens=[
                "Answer",
                "Answer ",
                f"Answer {self.question_index}.{next_subquestion_id}",
                f"Answer {self.question_index}.{next_subquestion_id}:",
                f"Answer {self.question_index}.{next_subquestion_id}: ",
            ],
        )

        # subquestion_list = [io_output.split("?")[0] + "?" for io_output in io_output_list]  # cleaning, you might wanna modify this
        subquestion_list = [o.strip() for o in io_output_list]

        #! generate subanswers to the subquestions generated above
        io_input_list = []
        for subquestion in subquestion_list:
            io_input = (
                decompose_prompt
                + "\n\n"
                + f"Question {self.question_index}: {user_question}"
                + "\n"
                + existing_subquestions_and_subanswers
                + f"Question {self.question_index}.{next_subquestion_id}: "
                + subquestion
                + "\n"
                + f"Answer {self.question_index}.{next_subquestion_id}:"
            )
            io_input_list.append(io_input)

        if reach_terminal_subquestion(subquestion=subquestion, user_question=user_question):
            num_return = self.mcts_num_last_votes
        else:
            num_return = self.num_votes

        io_output_list = self.io.generate(
            io_input_list,
            max_tokens=512,
            num_return=num_return,
            stop_tokens=[
                f"Question {self.question_index}.{next_subquestion_id + 1}",
            ],
        )
        cleaned_io_output_list = [
            [io_output.strip() for io_output in io_output_group] for io_output_group in io_output_list
        ]

        for i, cleaned_io_output_group in enumerate(cleaned_io_output_list):
            try:
                most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_group)
            except Exception as e:
                raise GeneratorError(
                    source="generate answer to subquestions",
                    io_input=io_input_list[i],
                    io_output_list=cleaned_io_output_group,
                )
            subanswer_list.append(most_likely_answer)
            value_list.append(likelihood)

        assert len(subquestion_list) == len(subanswer_list) == len(value_list)

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []
        if self.enable_potential_score:
            for subq, suba in zip(subquestion_list, subanswer_list):
                if reach_terminal_subquestion(subq, user_question):
                    potential_answers_list.append(None)
                else:
                    response_prefix = make_response_prefix(
                        solution_trace, Node_Type.SUBQUESTION, new_subq=subq, new_suba=suba
                    )
                    potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix

                    potential_score_output = self.io.generate(
                        potential_score_input,
                        num_return=self.num_votes,
                        max_tokens=128,
                        stop_tokens=self.fewshot_cot_config["stop_tokens"],
                    )
                    potential_score_input2 = [
                        "Question: "
                        + user_question
                        + "\nAnswer: "
                        + response_prefix
                        + z
                        + "\nTherefore, the answer (arabic numerals) is"
                        for z in potential_score_output
                    ]
                    cleaned_io_output_list = self.io.generate(
                        potential_score_input2,
                        num_return=1,
                        max_tokens=128,
                        stop_tokens=self.fewshot_cot_config["stop_tokens"],
                    )
                    cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                    potential_answers_list.append(
                        [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                    )
        else:
            potential_answers_list = [None] * len(subquestion_list)

        return subquestion_list, subanswer_list, value_list, potential_answers_list                    