# radioSound
Python script to generate sound effects of TV, radio, or Talkie-walkie (.mp3 or .wav)

# Installation
Dependencies (100% pip, no system tools required) :
```bash
	pip install numpy scipy soundfile imageio-ffmpeg
```

# Usage
```bash
	python radioEffect.py input.mp3 output.mp3
```

# Settings
    --noise     : 0.001 (light) à 0.01 (loud)
    --threshold : 0.1 (high compression) à 0.4 (low compression)
    --ratio     : 4 (low compression) à 10 (very high compression)
	
# Recommended values
For a "radio" effect: --noise 0.006 --threshold 0.15 --ratio 10  
For a "talkie-walkie" effect: --noise 0.01 --threshold 0.1 --ratio 15  
For a "old TV" effect: --noise 0.002 --ratio 6  

Coded with LLM (Claude)