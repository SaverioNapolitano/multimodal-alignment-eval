Downstream task awareness 
- Captioning: "Describe the image. Consider that the target audience is " + `chatGPT`/`Gemini` + " and its task will be to generate the same image based on the caption" 
- Generation 
  - "generate an image based on the following caption. Consider that the target audience is " + `humans`/`non-human tools` + " and their purpose is to assess how close the generated image is to the original reference from which the caption was extracted"
  - "generate an image based on the following caption. Consider that the target audience is " + `humans`/`non-human tools` + " and their purpose is to assess how close the generated image is to the original reference from which the caption was extracted. Consider that the caption was created by " + `chatGPT`/`Gemini` 