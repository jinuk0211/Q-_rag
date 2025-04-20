if not self.disable_a1:
    do_action_generate_ost_step(True)
if not self.disable_rag:
    do_action_generate_rag_step(True)
do_action_generate_direct_answers()
do_action_generate_subquestions()

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
(subquestion_list, subanswer_list, value_list, potential_answers_list) = (
    self.generator.generate_subquestions(
                    user_question=self.user_question, solution_trace=self.solution_trace, paraphrased=self.paraphrased
                ))
    def generate_subquestions(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
    ):
        subquestion_list, subanswer_list, value_list = [], [], []
        decompose_prompt = self.decompose_prompt if not paraphrased else self.decompose_prompt_rephrased

        #! generate subquestions
        io_input = (
            decompose_prompt
            + "\n\n"
            + f"Question {self.question_index}: {user_question}"
            + "\n"
            + existing_subquestions
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

def subanswer():
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
#--------------------------------------
def reach_terminal_subquestion(subquestion: str, user_question: str):
    assert subquestion is not None
    if "Now we can answer" in subquestion:
        #! remember that: when the original question is answerable, please start the subquestion with "Now we can answer the question: "
        return True
    user_question_2nd_part = split_user_question(user_question)[1]
    if user_question_2nd_part.lower() in subquestion.lower():
        return True
    return False
def split_user_question(user_question: str):
    user_question = user_question.strip().rstrip(".")
    last_period_id = user_question.rfind(".")
    assert last_period_id < len(user_question) - 1
    user_question_context = user_question[: last_period_id + 1].strip()
    user_question_problem = user_question[last_period_id + 1 :].strip()
    return user_question_context, user_question_problem


#--------------------------------------
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
#-------------------------------------
def make_response_prefix(
    solution_trace: Dict[int, Dict[str, str]], node_type: Node_Type, new_subq=None, new_suba=None, new_ost_step=None
) -> str:
    if node_type in [Node_Type.SUBQUESTION, Node_Type.RE_SUBANSWER]:
        response_prefix = ""
        answer_marker = "The answer is"  # todo: hard code "The answer is"
        for subquestion_id, solution_step in solution_trace.items():
            if subquestion_id == 0:
                continue

            assert subquestion_id > 0
            assert "subquestion" in solution_step.keys() and "subanswer" in solution_step.keys()

            response_prefix += solution_step["subanswer"]["text"].split(answer_marker)[0]
            response_prefix += " "

        if new_subq is not None and new_suba is not None:
            response_prefix += new_suba.split(answer_marker)[0]

        response_prefix = response_prefix.strip(" ")
    elif node_type is Node_Type.OST_STEP:
        response_prefix = ""

        last_tuple = list(solution_trace.items())[-1]
        last_tuple_recording = last_tuple[1]
        if "ost_step" in last_tuple_recording.keys():
            for step_id, step_text in last_tuple_recording["ost_step"].items():
                response_prefix += step_text + " "

        if new_ost_step is not None:
            response_prefix += new_ost_step

        response_prefix = response_prefix.strip(" ")
    elif node_type is None and solution_trace is None:
        response_prefix = ""
    else:
        raise ValueError(f"Invalid node type: {node_type}.")
    think = "Let's think step by step. "
    return think + response_prefix if think not in response_prefix else response_prefix
#---------------------------------------
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
    


    def rephrased_question(self, user_question: str):
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

  