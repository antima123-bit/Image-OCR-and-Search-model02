import re
import io
import torch
import gradio as gr
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

def OCRmodel():

    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True, torch_dtype=torch.float32)
    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    
    # Move the model to the correct device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, processor

model, processor = OCRmodel()

# Function to read the image and process it for OCR
def ocr(image_data):
    """
    Process the uploaded image and extract text using the OCR model.

    Args:
    image_data: Image data in bytes.

    Returns:
    Extracted text as a string.
    """
    text_query = "Extract all the text in Sanskrit and English from the image."
    # Prepare messages for the model with the image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_data}, 
                {"type": "text", "text": text_query}],
        }
    ]

    # Prepare text and image input for the model
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    
    # Process inputs
    inputs = processor(
        text=[text], 
        images=image_inputs, 
        padding=True, 
        return_tensors="pt"
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")  # Move to GPU if available

    # Generate the output from the model
    with torch.no_grad():
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=2000, no_repeat_ngram_size=3, temperature=0.7)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    return " ".join(output_text).strip()

# Function to highlight search terms in the text
def highlight_keywords(text, keywords):
    pattern = "|".join(re.escape(keyword) for keyword in keywords)
    highlighted_text = re.sub(f"({pattern})", rf'<mark style="background-color:{"red"};">\1</mark>', text, flags=re.IGNORECASE)
    return highlighted_text

# Gradio interface function
def process_image(image, search_query):
    
    extracted_text = ocr(image)
    
    if search_query:
        # Highlight matching keywords
        keywords = search_query.split()  # Split input into individual keywords
        highlighted_text = highlight_keywords(extracted_text, keywords)
    else:
        highlighted_text = extracted_text
    
    return highlighted_text

# Gradio Interface
application = gr.Interface(
    fn=process_image,  # Function to process the image and search query
    inputs=[
        gr.Image(type="pil", label="Upload Image"),  # Image input
        gr.Textbox(label="Enter search keywords")  # Textbox for search query
    ],
    outputs=gr.HTML(label="Extracted and Highlighted Text")  # Output area
)

# Launch the Gradio app
application.launch(share=True)