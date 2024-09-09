# Measuring Human and AI Values based on Generative Psychometrics with Large Language Models

### Requirements
- Python 3.10
- numPy
- torch
- transformers
- accelerate
- openai
- semchunk
- tiktoken

You may install the required packages by:
```bash
pip install -r requirements.txt
```

### Example Usage

#### Perception-level value measurements
```python
from gpv import GPV
perceptions = [
    "I love helping others", # Each perception is one sentence
    "Mary wants to get high scores in her exams",
    "Having fun all the time is important.",
]
values = ["hedonism", "achievement", "power", "benevolence", "universalism"]
gpv = GPV()
results = gpv.measure_perceptions(perceptions, values)
```

#### Text-level value measurements (for the text author)
```python
from gpv import GPV
texts = [
    "Today is a good day. I woke up early and went for a run in the park. The weather was perfect, and I felt energized. After my run, I had a healthy breakfast and spent some time reading a book. In the afternoon, I met up with some friends for lunch, and we had a great time catching up. I feel grateful for the wonderful day I had and look forward to more days like this...", # An example of a blog post
    "...",
]
values = ["hedonism", "achievement", "power", "benevolence", "universalism"]
gpv = GPV()
results = gpv.measure_texts(texts, values)
```

#### Text-level value measurements (for the given subjects)
```python
from gpv import GPV
texts = [
    "Mary is a PhD student in computer science. She is working on a project that aims to develop a new algorithm for image recognition. She is very passionate about her work and spends most of her time in the lab. She is determined to make a breakthrough in her field and become a successful researcher. Henry, on the other hand, is a high school student who is struggling with his grades. He is not interested in studying and spends most of his time playing video games. He is not motivated to do well in school and often skips classes. He dreams of becoming a professional gamer and making a living by playing video games.",
    "...",
]
values = ["hedonism", "achievement", "power", "benevolence", "universalism"]
gpv = GPV(measure_author=False)
results = gpv.measure_entities(text, values, entities)
```
