Q-Claw (powered by Qwen)

[![IMG-2978.webp](https://i.postimg.cc/90CktdGg/IMG-2978.webp)](https://postimg.cc/1n7MyN7D)


Q-Claw is a feature-rich terminal AI assistant that brings the power of Qwen directly to your command line. Designed as a comprehensive local AI interface, it combines intelligent conversational AI with practical system utilities.

[![claw.png](https://i.postimg.cc/FHF9rX6T/claw.png)](https://postimg.cc/RN8rdy4t)


Key Features:

AI-Powered Chat - Interactive conversations using Qwen models via Ollama 

Text-to-Speech - Built-in voice synthesis using Kokoro TTS 

Voice Input - Speech recognition with Vosk for hands-free interaction 

Search Integration - Wikipedia and DuckDuckGo search directly from terminal 

System Monitoring - Fetch OS info, kernel version, CPU, RAM, and uptime 

Shell Integration - Execute shell commands with persistent directory state

Command History - Full readline support with persistent history







 Smart Features:
Offline responses for greetings and common queries
Stateful shell execution with directory persistence
Streaming text output for natural feel
Automatic voice mode with customizable speeds
Continuous voice listening mode for hands-free operation
Configurable settings stored in ~/.q-claw/settings.json






Perfect for: Developers, system administrators, and power users seeking an intelligent, always-available terminal assistant with voice capabilities and web search integration.


```bash
git clone https://github.com/orion110/Q-Claw
cd Q-Claw
chmod +x Q-Claw.py
Q-Claw python3 

Download and run Ollama
curl -fsSL https://ollama.ai/install.sh | sh

ollama pull qwen2:0.5b

Start the Ollama service

ollama serve
 
pip install kokoro-onnx sounddevice numpy --break-system-packages

Download model files to ~/.q-claw/
Visit: https://github.com/remsky/Kokoro-ONNX

wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip

unzip vosk-model-small-en-us-0.15.zip -d /home//Q-Claw/vosk-model

Download:
- kokoro-v1.0.onnx
- voices-v1.0.bin
Place in: ~/Q-Claw/

Download and extract Vosk model
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip

unzip vosk-model-small-en-us-0.15.zip -d ~/.q-claw/vosk-model

pip install vosk sounddevice --break-system-packages

```



[![1153917653694562395.webp](https://i.postimg.cc/wMpDnvrR/1153917653694562395.webp)](https://postimg.cc/PpSLwtTd)

