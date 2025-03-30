# Shakespearean Text Generator  
A Deep Learning-based text generation model using LSTMs  

## Overview  
This project implements a character-level text generator trained on Shakespeare's works. It uses an LSTM model with dropout regularization to generate text in Shakespearean style. The model learns character sequences and generates text with adjustable creativity using temperature scaling and Top-K sampling.  

## Features  
✅ Trains an LSTM-based neural network to generate text  
✅ Uses temperature scaling to control text randomness  
✅ Implements Top-K sampling for better output diversity  
✅ Supports custom text length generation  
✅ Pretrained model available for direct inference  

## Technologies Used  
- **Python**  
- **TensorFlow / Keras**  
- **NumPy**  
- **LSTM (Long Short-Term Memory)**  
- **Natural Language Processing (NLP)**  

## Dataset  
The model is trained on the Shakespeare dataset provided by TensorFlow:  
[Shakespeare Text Dataset](https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt)  

---

## Installation & Setup  

### **1️. Clone the Repository**  
bash
git clone https://github.com/your-username/shakespeare-text-generator.git
cd shakespeare-text-generator


### **2️. Install Dependencies**
bash
pip install tensorflow numpy


### **3️. Train the Model**
(Uncomment the training code in text_generator.py before running this command.)
bash
python text_generator.py


### **4️. Generate Text**
Run the script and input desired parameters:
```bash
python text_generator.py
```

**Sample input:**

```bash
Enter length of text to generate: 350
Enter temperature (0.2 to 1.0): 0.5
Enter top-k value for sampling (default 5): 5
```

## Model Architecture

2 LSTM layers (256 units each)

Dropout (0.2) for regularization

Dense Softmax layer for character prediction

Adam optimizer with learning rate scheduling

**Sample Output**
```txt
HAMLET:  
O moon, thou dost wander yonder skies,  
And light doth fall upon my weary heart.  
I speak, yet words do vanish into air,  
Like echoes lost upon the trembling sea.  

JULIET:  
O love, the night is long and full of sighs,  
Yet in thine eyes, a dawn is near to rise.  
Come forth, my heart, and let me hear thy tune,  
For time is fleeting ‘neath the silent moon.
```

### **Future Improvements**

🔹 Use Transformer models (GPT-2, GPT-3) for better text generation

🔹 Implement Beam Search for improved sequence coherence

🔹 Add Web UI for interactive text generation

