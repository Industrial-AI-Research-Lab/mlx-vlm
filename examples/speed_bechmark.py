from mlx_vlm import load, apply_chat_template, generate
from mlx_vlm.utils import load_image
from mlx_vlm.utils import process_image

import time

img_filepath = ["/path/to/test_img.png"]
with open('path/to/prompt.txt', 'r') as f:
    long_sys_prompt = f.read()

model_name = [
    'mlx-community/Qwen2-VL-2B-Instruct-bf16', 
    'mlx-community/Qwen2-VL-2B-Instruct-4bit',
    'mlx-community/Qwen2-VL-7B-Instruct-bf16',
    'mlx-community/Qwen2-VL-7B-Instruct-4bit',
              ]

img_resolution = {
    '1024x768': (1024, 768),
    'high_res': (2402, 1342)
}

prompt_message = {
    'no_prompt' : [{"role": "user", "content": ""}],
    'long_prompt': [{"role": "system", "content": long_sys_prompt}, {"role": "user", "content": ""}]
}
max_tokens=1000
temperature=0.3
retries = 1
verbose = True

for mn in model_name:
    model, processor = load(mn)
    config = model.config   
    for ir in img_resolution:
        resized_images = [process_image(load_image(image), img_resolution[ir], None) for image in img_filepath]
        for pm in prompt_message:
            print()
            print(f'========== {mn, pm, ir} ==========')
            for i in range(retries):

                prompt = apply_chat_template(processor, config, prompt_message[pm], num_images=len(img_filepath))

                qwen_vl_output, service_info = generate(
                    model,
                    processor,
                    prompt,
                    resized_images,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    verbose=verbose,
                    return_service_info=True
                )

                print(service_info)

