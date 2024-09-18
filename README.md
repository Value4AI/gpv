# Measuring Human and AI Values based on Generative Psychometrics with Large Language Models

## 🚀 Introduction

This codebase accompanies the paper [*Measuring Human and AI Values based on Generative Psychometrics with Large Language Models*](https://github.com/Value4AI/gpv). We introduce **G**enerative **P**sychometrics for **V**alues (GPV), an LLM-based, data-driven value measurement paradigm, theoretically grounded in text-revealed selective perceptions.

Compared with traditional tools for measuring human values, GPV (1) effectively mitigates response bias and resource demands by dispensing with self-reports; (2) captures authentic behaviors instead of relying on forced ratings; (3) can handle historical or subjective data; (4) measures values in open-ended value spaces and easily adapts to new or evolving values without expert effort; and (5) enables more scalable and flexible value measurement.

Compare with recent works on measuring LLM values, GPV (1) mitigates response bias and yields more theoretically valid results; (2) is more practically relevant for measuring LLM values based on their scalable and free-form responses; and (3) enables context-specific measurements.

## 📦 Requirements
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

## 🔑 Example Usage

Note that there are two LLMs involved in GPV: the **parsing LLM** and the **measuring LLM**. 

You may set the **parsing LLM** by feeding the `parsing_model_name` parameter when initializing the GPV object. For example, `gpv = GPV(parsing_model_name="gpt-4o-mini")`. Accordingly, you need to set your API key as an environment variable `OPENAI_API_KEY` or [here](./gpv/models/models.py). Alternative LLMs can be used; please see `./gpv/models/` for more details.

The **measuring LLM** is set to our [`ValueLlama`](https://huggingface.co/Value4AI/ValueLlama-3-8B) by default.

### Perception-level value measurements
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

### Parsing long texts into perceptions
```python
from gpv import GPV

texts = [
    "Today is a good day. I woke up early and went for a run in the park. The weather was perfect, and I felt energized. After my run, I had a healthy breakfast and spent some time reading a book. In the afternoon, I met up with some friends for lunch, and we had a great time catching up. I feel grateful for the wonderful day I had and look forward to more days like this...", # e.g., a blog post
    "...",
]

gpv = GPV()
results = gpv.parse_texts(texts)
```

### Text-level value measurements (for the text author)
```python
from gpv import GPV

texts = [
    "Today is a good day. I woke up early and went for a run in the park. The weather was perfect, and I felt energized. After my run, I had a healthy breakfast and spent some time reading a book. In the afternoon, I met up with some friends for lunch, and we had a great time catching up. I feel grateful for the wonderful day I had and look forward to more days like this...", # e.g., a blog post
    "...",
]
values = ["hedonism", "achievement", "power", "benevolence", "universalism"]

gpv = GPV()
results = gpv.measure_texts(texts, values)
```

### Text-level value measurements (for the given subjects)
```python
from gpv import GPV

text = "Mary is a PhD student in computer science. She is working on a project that aims to develop a new algorithm for image recognition. She is very passionate about her work and spends most of her time in the lab. She is determined to make a breakthrough in her field and become a successful researcher. Henry, on the other hand, is a high school student who is struggling with his grades. He is not interested in studying and spends most of his time playing video games. He is not motivated to do well in school and often skips classes. He dreams of becoming a professional gamer and making a living by playing video games."  # e.g., an essay
values = ["hedonism", "achievement", "power", "benevolence", "universalism"]
measurement_subjects = ["Mary", "Henry"]

gpv = GPV(measure_author=False)
results = gpv.measure_entities(text, values, measurement_subjects)
```

### Text-level value measurements based on RAG (for the given subjects)
```python
from gpv import GPV

path = "data/西游记-zh.txt"
with open(path, "r") as file:
    book = file.read() # e.g., a novel
measurement_subjects = ["唐僧", "悟空", "八戒", "沙僧"]
coref_resolve = {
    "唐僧": ["唐三藏", "师父"],
    "悟空": ["猴王", "行者"],
    "八戒": ["猪八戒", "猪悟能"],
    "沙僧": ["沙和尚", "沙悟净"],
}
values = ["Universalism", "Hedonism", "Achievement", "Power", "Security", "Self-Direction", "Stimulation", "Tradition", "Benevolence", "Conformity"]

gpv = GPV(measure_author=False)
results = gpv.measure_entities_rag(
    text=book,
    values=values,
    measurement_subjects=measurement_subjects,
    coref_resolve=coref_resolve
    )
```

## 📄 Citation

If you find this codebase helpful, we would appreciate it if you give us a star and cite our paper:

```bibtex
TODO
```
