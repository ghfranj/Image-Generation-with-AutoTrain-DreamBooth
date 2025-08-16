This repo aims to show the use of diffusion models in AutoTrain provided by DreamBooth

initializing trained model:
~~~
model = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    model,
    torch_dtype= float16,
)
pipe.to("cuda")
pipe.load_lora_weights(prj_path, weight_name="pytorch_lora_weights.safetensors")
~~~

## Testing:
~~~
all_images = []
prompts = ["photo of Ghfran, cute smile, full body, with wings like an angel, high quality",
           "photo of Ghfran in a forest, sitting beside the lake",
           "photo of Ghfran, in street, cyberpunk, like a spy, standing on a car, purple sky",
           "photo of Ghfran on a skyscraper roof, firework, smiling to the camera",
           "photo of Ghfran in samurai costum in china, realism",
           "photo of Ghfran in a long wedding dress",
           "photo of Ghfran like a CEO, in a fancy office, dressed in black"]

for i, prompt in enumerate(prompts):
    image = pipe(prompt=prompt, generator=generator).images[0]
    # image = image.to('cpu')
    print(type(image))
    all_images.append(image)
    print(image)
~~~

## Samples of results:

![image](https://github.com/ghfranj/Image-Generation-with-AutoTrain-DreamBooth/assets/98123238/4853dbe5-a7e1-4690-ac3c-b4f573feae5c)


