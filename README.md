# A Novel Approach for Cuffless and Continuous Blood Pressure Measurement Using Dual Sensor Fusion Technology on the Wrist

**Authors**:  
Haydar Ankışhan<sup>1</sup>, Blaise B. Frederick<sup>2</sup>, Lia M. Hocke<sup>3</sup>  
1. Stem Cell Institute, Ankara University, Ankara, Turkey  
2. Mclean Hospital, MIC, Harvard University, MA, USA  
3. Harvard Medical School, Department of Psychiatry, Harvard University, MA, USA  

**Corresponding Author**:  
Haydar Ankışhan, [ankishan@ankara.edu.tr](mailto:ankishan@ankara.edu.tr)  

## Abstract
This repository contains the implementation of a novel system for continuous and cuffless blood pressure (BP) measurement, leveraging dual photo plethysmography (PPG) signals. Our model, termed Graph Attention Network-Transformer (GAN-T), integrates Graph Convolutional Network (GCN) and Transformer layers to analyze PPG signals from the wrist. This innovative approach allows for the continuous and non-invasive estimation of both systolic (SBP) and diastolic (DBP) blood pressure.

GAN-T uses advanced pulse wave analysis (PWA) to extract hybrid features (1x11 D with Principal Component Analysis (PCA) components (1x2 D)) from dual PPG signals. Our data-driven model demonstrates high precision, achieving a mean absolute error (MAE) of 2.47 mmHg for SBP and 1.09 mmHg for DBP. The Pearson correlation coefficients (PCC) are 0.9733 for SBP and 0.9595 for DBP, indicating strong predictive accuracy.

## Keywords
Cuffless blood pressure measurement, Dual PPG sensor fusion, Transformer, Graph Attention Network, Hypertension, PCA

## Introduction
Cuffless BP measurement has emerged as a promising alternative to traditional methods that require a cuff to occlude arteries. This technique significantly reduces patient discomfort and enables continuous monitoring. Our approach utilizes dual PPG sensors and advanced machine learning models to improve the accuracy and reliability of BP estimation.

The integration of dual PPG sensors helps to mitigate issues caused by motion artifacts and sensor placement variability. By capturing a comprehensive set of physiological signals, our system enhances the robustness and precision of BP measurements. Our model combines GCN architecture with Transformer layers to detect complex patterns in the PPG signals, offering a significant advancement in non-invasive BP monitoring technology.

## Methodology
- **Feature Extraction**: We extract 1x13 D hybrid features from dual PPG signals, incorporating both time-domain and PCA components.
- **Model Architecture**: The GAN-T model integrates GCN and Transformer layers to analyze the extracted features and predict SBP and DBP values.
- **Evaluation**: Our model is rigorously evaluated using a dataset from McLean Hospital, adhering to Harvard Medical School criteria for data utilization.

## Results
- **SBP Estimation**: MAE of 2.47 mmHg with a PCC of 0.9733 (R²=0.94)
- **DBP Estimation**: MAE of 1.09 mmHg with a PCC of 0.9595 (R²=0.91)

## Contributions
- Introduction of the GAN-T model for continuous, cuffless BP measurement using dual PPG signals.
- Development of novel 1x13 D hybrid features, enhancing BP estimation accuracy.
- Comprehensive evaluation demonstrating the model's effectiveness in real-world settings.

## Conclusion
Our GAN-T model represents a significant advancement in BP monitoring, offering precise, non-invasive, and real-time predictions. By combining dual PPG sensor fusion with cutting-edge deep learning techniques, this research contributes significantly to the field of health informatics and opens new avenues for impactful healthcare applications.

## Contact
For more information, please contact Haydar Ankışhan at [ankishan@ankara.edu.tr](mailto:ankishan@ankara.edu.tr).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
We would like to thank the Stem Cell Institute at Ankara University, Mclean Hospital, and Harvard Medical School for their support and collaboration in this research.
