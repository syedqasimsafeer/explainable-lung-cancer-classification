
import os
import json
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess


st.set_page_config(
    page_title="Explainable Lung Cancer Classification",
    page_icon="🫁",
    layout="wide"
)


@st.cache_resource
def load_model_and_config():
    custom_objects = {
        "preprocess_input": resnet_preprocess
    }

    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(
            "deployment/best_lung_cancer_model.keras",
            custom_objects=custom_objects,
            safe_mode=False,
            compile=False
        )

    with open("deployment/deployment_config.json", "r") as f:
        config = json.load(f)

    return model, config


@st.cache_data
def load_results():
    return pd.read_csv("artifacts/final_model_comparison_results.csv")


@st.cache_data
def load_xai_results():
    try:
        return pd.read_csv("artifacts/gradcam_xai_reliability_results.csv")
    except Exception:
        return None


def find_base_model(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and len(layer.layers) > 20:
            return layer
    raise ValueError("Base model not found.")


@st.cache_resource
def build_gradcam_model(_model):
    base_model = find_base_model(_model)

    base_index = None
    for i, layer in enumerate(_model.layers):
        if layer == base_model:
            base_index = i
            break

    inputs = _model.input
    x = inputs

    for layer in _model.layers[1:base_index]:
        x = layer(x)

    conv_outputs = base_model(x, training=False)
    x = conv_outputs

    for layer in _model.layers[base_index + 1:]:
        try:
            x = layer(x, training=False)
        except TypeError:
            x = layer(x)

    predictions = x

    grad_model = tf.keras.Model(
        inputs=inputs,
        outputs=[conv_outputs, predictions]
    )

    return grad_model


def read_uploaded_image(uploaded_file, image_size):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return None, None, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size))
    img_array = img_resized.astype(np.float32)
    img_array_batch = np.expand_dims(img_array, axis=0)

    return img_rgb, img_resized, img_array_batch


def predict_image(model, img_array, class_names):
    probabilities = model.predict(img_array, verbose=0)[0]
    predicted_id = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_id]
    confidence = float(probabilities[predicted_id])

    return predicted_id, predicted_class, confidence, probabilities


def generate_gradcam_heatmap(grad_model, img_array, class_index):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)

    if max_value == 0:
        heatmap = heatmap.numpy()
    else:
        heatmap = heatmap / max_value
        heatmap = heatmap.numpy()

    return heatmap


def overlay_gradcam(resized_img, heatmap, alpha=0.45):
    heatmap_resized = cv2.resize(
        heatmap,
        (resized_img.shape[1], resized_img.shape[0])
    )

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        resized_img.astype(np.uint8),
        1 - alpha,
        heatmap_color,
        alpha,
        0
    )

    return overlay, heatmap_resized


def create_foreground_mask(resized_img):
    gray = cv2.cvtColor(resized_img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    mask = gray > 20
    mask = mask.astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def compute_reliability_score(resized_img, heatmap_resized, confidence):
    foreground_mask = create_foreground_mask(resized_img)

    total_energy = np.sum(heatmap_resized) + 1e-8
    foreground_energy = np.sum(heatmap_resized * foreground_mask)

    focus_ratio = foreground_energy / total_energy
    reliability_score = confidence * focus_ratio

    return float(focus_ratio), float(reliability_score)


def get_prediction_message(predicted_class):
    if predicted_class == "normal":
        return "The model predicts this CT image as Normal."
    elif predicted_class == "benign":
        return "The model predicts this CT image as Benign."
    else:
        return "The model predicts this CT image as Malignant."


def main():
    st.title("🫁 Explainable Lung Cancer Classification from CT Images")

    st.write(
        "This app uses transfer learning and Grad-CAM explainability to classify lung CT images "
        "as Normal, Benign, or Malignant."
    )

    model, config = load_model_and_config()
    grad_model = build_gradcam_model(model)

    image_size = int(config["image_size"])
    class_names = config["class_names"]
    best_model_name = config["best_model_name"]

    results_df = load_results()
    xai_results_df = load_xai_results()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset", "IQ-OTH/NCCD")
    col2.metric("Classes", "3")
    col3.metric("Selected Model", best_model_name)
    col4.metric("Explainability", "Grad-CAM")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 CT Image Prediction",
        "🔥 Grad-CAM Explanation",
        "📊 Model Results",
        "ℹ️ About"
    ])

    with tab1:
        st.subheader("Upload a lung CT image")

        uploaded_file = st.file_uploader(
            "Upload JPG, JPEG, or PNG CT image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            original_img, resized_img, img_array = read_uploaded_image(
                uploaded_file,
                image_size
            )

            if original_img is None:
                st.error("Could not read the uploaded image. Please upload a valid CT image.")
            else:
                pred_id, pred_class, confidence, probabilities = predict_image(
                    model,
                    img_array,
                    class_names
                )

                st.session_state["resized_img"] = resized_img
                st.session_state["img_array"] = img_array
                st.session_state["pred_id"] = pred_id
                st.session_state["pred_class"] = pred_class
                st.session_state["confidence"] = confidence
                st.session_state["probabilities"] = probabilities

                left_col, right_col = st.columns(2)

                with left_col:
                    st.image(
                        original_img,
                        caption="Uploaded CT Image",
                        use_container_width=True
                    )

                with right_col:
                    st.subheader("Prediction Result")

                    if pred_class == "malignant":
                        st.error(f"Predicted Class: {pred_class.upper()}")
                    elif pred_class == "benign":
                        st.warning(f"Predicted Class: {pred_class.upper()}")
                    else:
                        st.success(f"Predicted Class: {pred_class.upper()}")

                    st.metric("Prediction Confidence", f"{confidence * 100:.2f}%")
                    st.info(get_prediction_message(pred_class))

                    prob_df = pd.DataFrame({
                        "Class": class_names,
                        "Probability": probabilities
                    })

                    fig = px.bar(
                        prob_df,
                        x="Class",
                        y="Probability",
                        title="Prediction Probabilities",
                        text=prob_df["Probability"].round(3)
                    )
                    fig.update_layout(yaxis=dict(range=[0, 1]))
                    st.plotly_chart(fig, use_container_width=True)

                st.warning(
                    "Medical disclaimer: This app is for educational and research demonstration only. "
                    "It is not a clinical diagnostic tool. Always consult qualified medical professionals."
                )

    with tab2:
        st.subheader("Grad-CAM Explanation")

        if "img_array" not in st.session_state:
            st.info("Please upload an image in the CT Image Prediction tab first.")
        else:
            resized_img = st.session_state["resized_img"]
            img_array = st.session_state["img_array"]
            pred_id = st.session_state["pred_id"]
            pred_class = st.session_state["pred_class"]
            confidence = st.session_state["confidence"]

            heatmap = generate_gradcam_heatmap(
                grad_model,
                img_array,
                pred_id
            )

            overlay, heatmap_resized = overlay_gradcam(
                resized_img,
                heatmap
            )

            focus_ratio, reliability_score = compute_reliability_score(
                resized_img,
                heatmap_resized,
                confidence
            )

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.image(
                    resized_img.astype(np.uint8),
                    caption="Original Resized CT",
                    use_container_width=True
                )

            with col_b:
                st.image(
                    np.uint8(255 * heatmap_resized),
                    caption="Grad-CAM Heatmap",
                    use_container_width=True,
                    clamp=True
                )

            with col_c:
                st.image(
                    overlay,
                    caption="Grad-CAM Overlay",
                    use_container_width=True
                )

            st.subheader("Explainability Scores")

            score_col1, score_col2, score_col3 = st.columns(3)
            score_col1.metric("Predicted Class", pred_class)
            score_col2.metric("Grad-CAM Focus Ratio", f"{focus_ratio:.3f}")
            score_col3.metric("Reliability Score", f"{reliability_score:.3f}")

            st.info(
                "The reliability score is calculated as prediction confidence multiplied by Grad-CAM focus ratio. "
                "It is an experimental interpretability indicator, not a clinical validation metric."
            )

    with tab3:
        st.subheader("Model Comparison Results")

        st.dataframe(results_df, use_container_width=True)

        metrics = [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Macro ROC-AUC"
        ]

        plot_df = results_df.melt(
            id_vars="Model",
            value_vars=metrics,
            var_name="Metric",
            value_name="Score"
        )

        fig = px.bar(
            plot_df,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            title="Model Performance Comparison"
        )
        fig.update_layout(yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            ### Key Result

            ResNet50 was selected for deployment because it achieved the highest Macro F1 score.
            Macro F1 was used as the selection metric because the dataset is imbalanced and the benign class has fewer samples.
            """
        )

        if xai_results_df is not None:
            st.subheader("Grad-CAM Reliability Results")

            xai_summary = xai_results_df.groupby("correct")[
                [
                    "gradcam_focus_ratio",
                    "confidence_explanation_reliability_score"
                ]
            ].mean().reset_index()

            st.dataframe(xai_summary, use_container_width=True)

            fig_xai = px.box(
                xai_results_df,
                x="correct",
                y="confidence_explanation_reliability_score",
                title="Reliability Score for Correct vs Wrong Predictions"
            )
            st.plotly_chart(fig_xai, use_container_width=True)

    with tab4:
        st.subheader("About This Project")

        st.markdown(
            """
            This project performs lung CT image classification using explainable deep learning.

            ### Pipeline

            - Dataset acquisition from IQ-OTH/NCCD lung cancer CT dataset
            - Image preprocessing and augmentation
            - Transfer learning with ResNet50, DenseNet121, and EfficientNetB0
            - Class imbalance handling using class weights
            - Performance comparison using Accuracy, Precision, Recall, F1-score, and ROC-AUC
            - Grad-CAM explainability for model interpretation
            - Streamlit deployment

            ### Novelty Direction

            The project does not only report accuracy. It compares models using Macro F1 and Grad-CAM-based reliability analysis.
            This is important because medical datasets are often imbalanced, and accuracy alone can hide weak minority-class performance.
            """
        )

        st.warning(
            "This project is for research and educational demonstration only. "
            "It should not be used for real medical diagnosis."
        )


if __name__ == "__main__":
    main()
