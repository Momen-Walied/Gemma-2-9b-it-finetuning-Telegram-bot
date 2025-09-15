# 🇪🇬 Gemma-2-9B Egyptian Arabic Translator Bot

<div align="center">

![Gemma Logo](https://img.shields.io/badge/Gemma-2--9B--IT-blue?style=for-the-badge&logo=google&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Telegram](https://img.shields.io/badge/Telegram-Bot-blue?style=for-the-badge&logo=telegram&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-red?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow?style=for-the-badge)

*Fine-tuned Google Gemma-2-9B model for English to Egyptian Arabic dialect translation with Telegram bot integration*

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-project-structure) • [🎥 Video Demo](#-video-demo) • [🤖 Bot Demo](#-telegram-bot-demo) • [🔧 Installation](#-installation)

</div>

---

## 🌟 Overview

This project demonstrates the complete pipeline of fine-tuning Google's **Gemma-2-9B-IT** model to translate English text into authentic **Egyptian Arabic dialect** (اللهجة المصرية). The project includes data preparation, model fine-tuning using LoRA (Low-Rank Adaptation), and deployment as an interactive Telegram bot.

### ✨ Key Features

- 🎯 **Specialized Translation**: Fine-tuned specifically for Egyptian Arabic dialect
- 🚀 **Efficient Training**: Uses LoRA and 4-bit quantization for memory-efficient training
- 🤖 **Telegram Integration**: Ready-to-deploy bot for real-time translation
- 📊 **Complete Pipeline**: From data preparation to deployment
- 💾 **Chat History**: Saves user conversations for analysis
- ⚡ **Fast Inference**: Optimized for real-time responses

## 🎯 What Makes This Special?

Unlike generic Arabic translators, this model is specifically trained on **Egyptian dialect** (المصري), capturing:
- Colloquial expressions and slang
- Cultural context and idioms
- Natural conversational flow
- Regional linguistic nuances

## 🎥 Video Tutorial

<div align="center">

[![YouTube Video Tutorial](https://img.shields.io/badge/🎥-Watch%20Demo%20on%20YouTube-red?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/3hjUMs2JJak)

**Watch the complete walkthrough of the project from training to deployment!**

</div>

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Telegram Bot Token

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Gemma-2-9b-it-finetuning-Telegram-bot.git
cd Gemma-2-9b-it-finetuning-Telegram-bot

# Install dependencies
pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
pip install xformers trl peft accelerate bitsandbytes triton
pip install pyTelegramBotAPI transformers datasets pandas
```

### Quick Demo

```python
# Load the fine-tuned model
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="English2Egyptian",  # Your fine-tuned model
    max_seq_length=2048,
    dtype=None,
)

# Translate English to Egyptian Arabic
query = 'Translate to Egyptian dialect: "I love Egypt and its people"'
# Output: "أنا بحب مصر وشعبها" 🇪🇬
```

## 📖 Project Structure

The project is organized into 4 comprehensive parts:

### 📁 Part 1: Model Loading
```
part 1 - load models/
├── part-1.ipynb          # Load and test base Gemma-2-9B model
```
- Loads Google's Gemma-2-9B-IT model
- Sets up 4-bit quantization for memory efficiency
- Tests base model capabilities

### 📁 Part 2: Dataset Creation
```
part 2 - create dataset/
├── part-2.ipynb          # Data preprocessing and formatting
├── data.csv              # English-Egyptian Arabic parallel corpus
```
- Processes parallel English-Egyptian Arabic dataset
- Creates properly formatted training prompts
- Splits data into train/test sets (80/20)

### 📁 Part 3: Fine-tuning Process
```
part 3 - fine tuning process/
├── part-3.ipynb          # LoRA fine-tuning implementation
```
- Implements LoRA (Low-Rank Adaptation) for efficient training
- Configures training parameters and optimization
- Saves the fine-tuned model as "English2Egyptian"

### 📁 Part 4: Telegram Bot
```
part 4 - Telegram/
├── Part4.ipynb           # Telegram bot implementation
```
- Creates interactive Telegram bot
- Implements chat history saving
- Handles real-time translation requests

## 🔧 Technical Details

### Model Architecture
- **Base Model**: Google Gemma-2-9B-IT
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit for memory efficiency
- **Context Length**: 2048 tokens

### Training Configuration
```python
# LoRA Configuration
r = 16                    # Rank
lora_alpha = 16          # Alpha parameter
lora_dropout = 0         # Dropout (optimized)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]

# Training Parameters
batch_size = 2
gradient_accumulation_steps = 4
learning_rate = 2e-5
epochs = 10
```

### Dataset Format
The training data follows this structure:
```json
{
  "instruction": "ترجم الي اللهجه المصريه",
  "input": "I love Egypt and its people",
  "output": "أنا بحب مصر وشعبها"
}
```

## 🤖 Telegram Bot Demo

### Bot Features
- **Welcome Message**: Greets users in Arabic
- **Real-time Translation**: Instant English to Egyptian Arabic translation
- **Chat History**: Saves all conversations in JSON format
- **Error Handling**: Robust error management and user feedback

### Sample Conversation
```
User: "Hello, how are you today?"
Bot: "أهلاً، إزيك النهاردة؟"

User: "I want to visit the pyramids"
Bot: "عايز أزور الأهرامات"
```

### Setting Up the Bot

1. **Create a Telegram Bot**:
   - Message @BotFather on Telegram
   - Use `/newbot` command
   - Get your bot token

2. **Configure the Bot**:
   ```python
   bot = telebot.TeleBot("YOUR_TELEGRAM_BOT_TOKEN")
   ```

3. **Run the Bot**:
   ```bash
   python telegram_bot.py
   ```

## 📊 Dataset Information

The training dataset contains **893 parallel sentences** covering:
- Daily conversations
- Cultural expressions
- Common phrases and idioms
- Egyptian-specific terminology

### Sample Data Points
| English | Egyptian Arabic |
|---------|-----------------|
| "I want to tell you something funny" | "عايز احكيلكم حاجه غريبه جداً" |
| "I have never visited the Pyramids" | "عمرى ما زرت الاهرامات" |
| "People from all over the world" | "الناس من كل أنحاء العالم" |

## 🚀 Performance & Results

### Training Metrics
- **Training Loss**: Converged after 10 epochs
- **Memory Usage**: ~12GB VRAM with 4-bit quantization
- **Training Time**: ~2-3 hours on modern GPU

### Translation Quality
The model demonstrates excellent performance in:
- ✅ Maintaining cultural context
- ✅ Using appropriate Egyptian dialect forms
- ✅ Handling colloquial expressions
- ✅ Preserving meaning and tone

## 🛠️ Advanced Usage

### Custom Training
To train on your own dataset:

1. **Prepare Data**: Format as CSV with "English Sentence" and "Arabic Sentence" columns
2. **Update Dataset Path**: Modify `data.csv` path in Part 2
3. **Adjust Parameters**: Tune LoRA and training parameters in Part 3
4. **Run Pipeline**: Execute notebooks sequentially

### Model Deployment
```python
# Save model for deployment
model.save_pretrained_merged("English2Egyptian", 
                           tokenizer, 
                           save_method="merged_16bit")

# Load for inference
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="English2Egyptian",
    max_seq_length=2048,
    dtype=None,
)
```

## 📋 Requirements

### Core Dependencies
```
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
torch>=2.0.0
transformers>=4.30.0
datasets>=2.0.0
peft>=0.4.0
trl>=0.4.0
bitsandbytes>=0.39.0
accelerate>=0.20.0
pyTelegramBotAPI>=4.0.0
pandas>=1.3.0
```

### Hardware Requirements
- **Minimum**: 12GB VRAM GPU
- **Recommended**: 16GB+ VRAM GPU (RTX 3090, A100, etc.)
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ free space

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **🐛 Bug Reports**: Open an issue with detailed description
2. **✨ Feature Requests**: Suggest new features or improvements
3. **📝 Documentation**: Help improve documentation
4. **🔧 Code**: Submit pull requests with improvements

### Development Setup
```bash
git clone https://github.com/your-username/Gemma-2-9b-it-finetuning-Telegram-bot.git
cd Gemma-2-9b-it-finetuning-Telegram-bot
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google**: For the amazing Gemma-2-9B model
- **Unsloth**: For the efficient fine-tuning framework
- **Hugging Face**: For the transformers library
- **Egyptian Arabic Community**: For linguistic insights and validation

## 📞 Contact & Support

- **🌐 Website**: [momenwalied.camitai.com](https://momenwalied.camitai.com/)
- **GitHub Issues**: [Report bugs or request features](https://github.com/your-username/Gemma-2-9b-it-finetuning-Telegram-bot/issues)
- **Discussions**: [Join the community discussion](https://github.com/your-username/Gemma-2-9b-it-finetuning-Telegram-bot/discussions)

---

<div align="center">

**Made with ❤️ for the Egyptian Arabic NLP community**

*If this project helped you, please consider giving it a ⭐!*

</div>
