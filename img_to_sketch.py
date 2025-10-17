import gradio as gr
import numpy as np
import scipy.ndimage

def image_to_sketch(image):
    # Convert RGB to grayscale
    def rgbtogrey(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
    
    # Dodge function
    def dodge(front, back):
        final_sketch = front*255/(255-back)
        final_sketch[final_sketch > 255] = 255
        final_sketch[back == 255] = 255
        return final_sketch.astype('uint8')
    
    gray = rgbtogrey(image)
    i = 255 - gray
    blur = scipy.ndimage.gaussian_filter(i, sigma=13)
    r = dodge(blur, gray)
    return r

demo = gr.Interface(
    fn=image_to_sketch,
    inputs=gr.Image(type="numpy"),   # User uploads image here
    outputs=gr.Image(type="numpy"),
    title="Image to Sketch Converter 🎨",
    description="Upload an image and get its sketch instantly!"
)

demo.launch()
