<div align="center">

# FedXR-GAN

### Federated Learning for Pneumonia Classification using GAN-Augmented Chest X-Rays

<p>
  <strong>AI in Healthcare</strong><br><br>
  <img title="UPC Logo" alt="UPC Logo" src="doc/media/logo.png" width="400">
</p>

<p>
  <strong>Universitat Politècnica de Catalunya · BarcelonaTech (UPC)</strong><br>
  EPSEM — Escola Politècnica Superior d'Enginyeria de Manresa
</p>

<p>
  <em>May 2026</em>
</p>

</div>

---

## Project Overview

FedXR-GAN is a research-oriented project focused on combining:

- **Federated Learning (FL)**
- **GAN-based medical image synthesis**
- **Chest X-ray pneumonia classification**

The objective is to investigate whether synthetic GAN-generated chest X-rays can improve distributed medical diagnosis systems while preserving patient privacy through decentralized training.

---

## Project Structure

```bash
FedXR-GAN/
├── data/                # Dataset and preprocessing
├── doc/                 # Latex files and PDF render
├── federated/           # Federated Learning pipeline
├── gan/                 # GAN architectures and training
├── models/              # CNN classification models
└── results/             # Metrics, plots and outputs
```

---

## Technologies

<div align="center">

| Area | Technologies |
|---|---|
| Deep Learning | TensorFlow |
| Federated Learning | Flower |
| Image Generation | DCGAN |
| Dataset | [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| Evaluation | FID, Accuracy, F1 |

</div>

---

## Setup
After cloning, run once to configure the local simulation:
```bash
bash federated/script.sh
```

---

## Academic Context

This repository was developed as part of the:

> **AI in Healthcare**  
> Master’s Degree in Machine Learning and Cybersecurity for Internet-Connected Systems  
> Universitat Politècnica de Catalunya (UPC)

---

## License

This project is intended for academic and research purposes only.
