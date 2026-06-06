import os
import json
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import tensorflow as tf

from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess


st.set_page_config(
    page_title="Explainable Lung Cancer Classification",
    page_icon="🫁",
    layout="wide"
)


IMG_SIZE = 224
NUM_CLASSES = 3


def build_resnet50_deployment_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

    base_model = ResNet50(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.35, name="dropout_1")(x)

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.001),
        name="dense_128"
    )(x)

    x = layers.Dropout(0.25, name="dropout_2")(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="classification_output"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="ResNet50_Deployment"
    )

    return model


@st.cache_resource
def load_model_and_config():
    with open("deployment/deployment_config.json", "r") as f:
        config = json.load(f)

    model = build_resnet50_deployment_model()

    model.load_weights(
        "deployment/best_lung_cancer_model.weights.h5"
    )

    return model, config


@st.cache_data
def load_results():
    path = "artifacts/final_model_comparison_results.csv"

    if os.path.exists(path):
        return pd.read_csv(path)

    return pd.DataFrame()


@st.cache_data
def load_xai_results():
    path = "artifacts/gradcam_xai_reliability_results.csv"

    if os.path.exists(path):
        return pd.read_csv(path)

    return pd.DataFrame()


def find_base_model(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "resnet" in layer.name.lower():
            return layer

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
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
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return None, None, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size))
    img_array = img_resized.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    return img_rgb, img_resized, img_array


def predict_image(model, img_array, class_names):
    model_input = resnet_preprocess(img_array.copy())

    probabilities = model.predict(model_input, verbose=0)[0]
    predicted_id = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_id]
    confidence = float(probabilities[predicted_id])

    return predicted_id, predicted_class, confidence, probabilities


def generate_gradcam_heatmap(grad_model, img_array, class_index):
    model_input = resnet_preprocess(img_array.copy())

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(model_input, training=False)
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
        return heatmap.numpy()

    heatmap = heatmap / max_value

    return heatmap.numpy()


def overlay_gradcam(resized_img, heatmap, alpha=0.45):
    heatmap_resized = cv2.resize(
        heatmap,
        (resized_img.shape[1], resized_img.shape[0])
    )

    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        resized_img.astype(np.uint8),
        1 - alpha,
        heatmap_color,
        alpha,
        0
    )

    return overlay, heatmap_resized


def create_foreground_mask(resized_img):
    gray = cv2.cvtColor(
        resized_img.astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    )

    mask = gray > 20
    mask = mask.astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return mask


def compute_reliability_score(resized_img, heatmap_resized, confidence):
    foreground_mask = create_foreground_mask(resized_img)

    total_energy = np.sum(heatmap_resized) + 1e-8
    foreground_energy = np.sum(heatmap_resized * foreground_mask)

    focus_ratio = foreground_energy / total_energy
    reliability_score = confidence * focus_ratio

    return float(focus_ratio), float(reliability_score)


def main():
    st.title("🫁 Explainable Lung Cancer Classification from CT Images")

    st.write(
        "This app uses transfer learning and Grad-CAM explainability to classify "
        "lung CT images as Normal, Benign, or Malignant."
    )

    model, config = load_model_and_config()
    grad_model = build_gradcam_model(model)

    image_size = int(config.get("image_size", 224))
    class_names = config.get(
        "class_names",
        ["normal", "benign", "malignant"]
    )

    best_model_name = config.get("best_model_name", "ResNet50")

    results_df = load_results()
    xai_results_df = load_xai_results()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Dataset", "IQ-OTH/NCCD")
    col2.metric("Classes", "3")
    col3.metric("Selected Model", best_model_name)
    col4.metric("Explainability", "Grad-CAM")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔍 CT Image Prediction",
            "🔥 Grad-CAM Explanation",
            "📊 Model Results",
            "ℹ️ About"
        ]
    )

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
                st.error(
                    "Could not read the uploaded image. Please upload a valid CT image."
                )
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

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                    prob_df = pd.DataFrame(
                        {
                            "Class": class_names,
                            "Probability": probabilities
                        }
                    )

                    fig = px.bar(
                        prob_df,
                        x="Class",
                        y="Probability",
                        title="Prediction Probabilities",
                        text=prob_df["Probability"].round(3)
                    )

                    fig.update_layout(
                        yaxis=dict(range=[0, 1])
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                st.warning(
                    "Medical disclaimer: This app is for educational and research "
                    "demonstration only. It is not a clinical diagnostic tool."
                )

    with tab2:
        st.subheader("Grad-CAM Explanation")

        if "img_array" not in st.session_state:
            st.info(
                "Please upload an image in the CT Image Prediction tab first."
            )
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

            c1, c2, c3 = st.columns(3)

            with c1:
                st.image(
                    resized_img.astype(np.uint8),
                    caption="Original Resized CT",
                    use_container_width=True
                )

            with c2:
                st.image(
                    np.uint8(255 * heatmap_resized),
                    caption="Grad-CAM Heatmap",
                    use_container_width=True,
                    clamp=True
                )

            with c3:
                st.image(
                    overlay,
                    caption="Grad-CAM Overlay",
                    use_container_width=True
                )

            st.subheader("Explainability Scores")

            s1, s2, s3 = st.columns(3)

            s1.metric("Predicted Class", pred_class)
            s2.metric("Grad-CAM Focus Ratio", f"{focus_ratio:.3f}")
            s3.metric("Reliability Score", f"{reliability_score:.3f}")

            st.info(
                "The reliability score is experimental. It is calculated as "
                "prediction confidence multiplied by Grad-CAM focus ratio."
            )

    with tab3:
        st.subheader("Model Comparison Results")

        if results_df.empty:
            st.warning("Model comparison file not found.")
        else:
            st.dataframe(
                results_df,
                use_container_width=True
            )

            metrics = [
                "Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro F1",
                "Macro ROC-AUC"
            ]

            available_metrics = [
                m for m in metrics if m in results_df.columns
            ]

            plot_df = results_df.melt(
                id_vars="Model",
                value_vars=available_metrics,
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

            fig.update_layout(
                yaxis=dict(range=[0, 1])
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        if not xai_results_df.empty:
            st.subheader("Grad-CAM Reliability Results")

            if (
                "correct" in xai_results_df.columns
                and "confidence_explanation_reliability_score" in xai_results_df.columns
            ):
                fig_xai = px.box(
                    xai_results_df,
                    x="correct",
                    y="confidence_explanation_reliability_score",
                    title="Reliability Score for Correct vs Wrong Predictions"
                )

                st.plotly_chart(
                    fig_xai,
                    use_container_width=True
                )

    with tab4:
        st.subheader("About This Project")

        st.markdown(
            """
            This project performs lung CT image classification using explainable deep learning.

            **Pipeline**

            - IQ-OTH/NCCD lung CT dataset
            - Image preprocessing and augmentation
            - Transfer learning using ResNet50, DenseNet121, and EfficientNetB0
            - Class imbalance handling using class weights
            - Model comparison using Accuracy, Precision, Recall, F1-score, and ROC-AUC
            - Grad-CAM explainability
            - Streamlit deployment

            **Important Note**

            This app is for educational and research demonstration only.
            It is not a medical diagnostic system.
            """
        )


if __name__ == "__main__":
    main()