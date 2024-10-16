# CNN-Interpretability

This project is focused on exploring the interpretability of deep learning models for computer vision tasks. It includes techniques such as saliency maps, Grad-CAM (with and without Captum), and class model visualization to better understand how models make predictions. Additionally, experiments were performed involving fooling images and style transfer to examine model robustness and creativity. This repository aims to provide insights into the inner workings of convolutional neural networks (CNNs) and improve their transparency.

# Features and Techniques

1. **Saliency Maps**:
   ![saliency_map](https://github.com/user-attachments/assets/b5345a29-f854-4d9b-a752-e126c504dd84)

   This technique is used to visualize which pixels of an image most strongly influence a model’s prediction. By computing the gradients of the output with respect to the input image, the saliency map highlights important areas that contribute to the classification. We also examine the reliability of saliency maps using sanity checks.

2. **Grad-CAM**:
   ![gradcam](https://github.com/user-attachments/assets/9c5a4748-cee2-4145-84b9-18163d3f1edb)
   Grad-CAM is used to provide a deeper understanding of where the model “looks” when making predictions. By utilizing class-specific gradient information flowing into the last convolutional layer, Grad-CAM generates coarse localization maps. This helps identify which parts of an image are most important for a particular class prediction.

3. **Class Model Visualization**:
   <img width="1146" alt="截圖 2024-10-16 上午8 52 23" src="https://github.com/user-attachments/assets/b36b4d19-de86-4711-8193-ef376fc3cc23">
   Perform gradient ascent on an input image to find a visual representation that maximizes the score for a specific class. This method offers insights into what features or patterns the model associates with a particular class label.

6. **Fooling Images**:
   <img width="612" alt="image" src="https://github.com/user-attachments/assets/97486146-e5e8-48aa-bed8-d37cc20eac73">
   This part of the project generates images that are perceived by the model to strongly belong to a particular class, despite being meaningless to the human eye. The goal is to demonstrate potential vulnerabilities and limitations in how models learn visual features.

8. **Style Transfer**:
   <img width="1129" alt="截圖 2024-10-16 上午8 55 13" src="https://github.com/user-attachments/assets/77607492-6bd7-4142-ae6e-866f99676882">
    <img width="1138" alt="截圖 2024-10-16 上午8 57 05" src="https://github.com/user-attachments/assets/07ca510d-81dc-49d2-b8db-1492f8a6a9dc">

   Use neural style transfer to generate new images by combining the content of one image with the artistic style of another. Different parameter settings are experimented with to assess how content and style interact and which parameters yield the best visual outputs.

10. **Using Captum for Interpretation**:
    <img width="977" alt="截圖 2024-10-16 上午8 57 33" src="https://github.com/user-attachments/assets/7d2f57e9-376c-4dde-bbae-d6e46db9919d">

   Some of the visualizations were generated using Captum, an open-source tool from Facebook that helps with model interpretability. Captum was particularly useful in generating Grad-CAM and saliency maps in a structured and modular way, making it easier to visualize and compare results.

This project aims to enhance model interpretability and understandability by applying and comparing these visualization techniques, helping to bridge the gap between model outputs and human intuition.
