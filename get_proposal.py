def llm_proposal(model=None,tokenizer=None,prompt=None,model_name='qwen'):
    if model_name =='qwen':
        messages = [ {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True)
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs, max_new_tokens=512)
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response  
    if model_name == 'gpt':
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )

        # 🎯 출력 결과
        reply = response['choices'][0]['message']['content'].strip()
        return reply 