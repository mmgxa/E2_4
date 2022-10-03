import streamlit as st
import torch

from PIL import Image
from torchvision.transforms import ToTensor
import os
from argparse import ArgumentParser


from typing import Dict

MODEL: str = "resnet18"

hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
            
# @st.cache
def load_model(model):
    jit_path = os.path.join(os.getcwd(), model)
    model = torch.jit.load(jit_path)
    model.eval()
    # get the classnames
    with open("cifar10_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return model, categories

# @st.cache
def predict(args, inp_img: Image, top_k: int) -> Dict[str, float]:
    inp_img = ToTensor()(inp_img) # the input image is scaled to [0.0, 1.0]
    inp_img = inp_img.unsqueeze(0)
    inp_img = inp_img.to(torch.float32, copy=True)
    model, categories = load_model(args.model)
    topk_ids = []
    # inference
    out = model.forward_jit(inp_img)
    topk = out.topk(top_k)[1][0]
    topk_ids.append(topk.cpu().numpy())
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    topk_prob, topk_label = torch.topk(probabilities, top_k)
    confidences = {}
    confidences['Labels'] = [categories[topk_label[i]] for i in range(topk_prob.size(0))]
    confidences['Confidence (%)'] = [f"{round(float(topk_prob[i]),2) *100:.2f}" for i in range(topk_prob.size(0))]
    return confidences

def main(args):
    st.set_page_config(
        page_title="EMLOv2 - S4 - MMG",
        layout="centered",
        page_icon="🐍",
        initial_sidebar_state="expanded",
    )

    st.title("CIFAR10 Classifier")
    st.subheader("Upload an image to classify it with a Model of Your Choice")

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "png", "jpeg"]
    )
    top_k = st.number_input('Return top k samples', 1,10,10)

    if st.button("Predict"):
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            st.image(image, caption="Uploaded Image", use_column_width=False)
            st.write("")

                    
            try:
                with st.spinner("Predicting..."):
                    predictions = predict(args, image, top_k)
                    # get key with highest value
                    st.success(f"Predictions are...")
                    st.markdown(hide_table_row_index, unsafe_allow_html=True)
                    st.table(predictions)
            except:
                st.error("Something went wrong. Please try again.")
        else:
            st.warning("Please upload an image.")

    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='model.trace.pt')
    args = parser.parse_args()
    main(args)
