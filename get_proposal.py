def get_proposal(model, processor, prompt, img_path, model_name ='llama'):
    #temperature=0.7, max_tokens=2048, seed=170, max_length=2048, truncation=True,do_sample=True, max_new_tokens=1024
    if model_name == 'qwen':
        response = []
        cnt = 2
        while not response and cnt:
            cnt -= 1
            messages = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                        { "role": "user", 
                        "content": [{"type": "image", "image": f"file://{img_path}"},
                        {"type": "text", "text": f"{prompt}"},],
                        }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if not response:
            print(f'obtain<qwen>response fail!\n')
            return []
        return response[0]

    elif model_name == 'llama':

        image = Image.open(img_path)

        messages = [
            {"role": "user", "content": [
                {"type": "image"},{"type": "text","text": f"{prompt}"}]}
                ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=500)
        print(processor.decode(output[0]))
        return output[0]    
