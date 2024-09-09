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

#### Text-level value measurements
```python
from gpv import GPV
texts = [
    "So anyhow, one day I was in her apartment, we were smoking, and such, and we got on the subjects of ghosts.  Yvonne and I told her about what happened at my apartment and her apartment.  Beth then said, 'I have a ghost too.'  I was like yeah right.  She said I do it's like to play with my daughter's toys, so I said prove it!  She got up and went into her daughter's room and took out her toy box.  I was like what are you doing, all she said was,'wait.'  So we waited for like 15 minutes, then i did the I have to go to the bathroom trick.  I looked in her daughter's room and two toys were taken out.  I went screaming back to them saying it happened.  When we all came back to look all of the toys were taken out of the box, it was so strange to me.  I never doubted her again.   A few months later I get a call from Beth and she asks me to watch her child, I was like ok, then after I hung up I remembered the ghost.  So I started getting a little worried, but then again I promised her.  When I arrived I saw the little girl, and Beth told me what she liked to do and so on.  After, I played with her daughter, and changed her and gave her a snack, then I wagon told to put her down, I did, and she didn't cry or anything.  As I was sitting there, I was a little freaked, I felt a presence around me, but I was like I couldn't move.  I started to watch TV, and the arms on my hair started raising up.", # An example of a blog post
    
    "But see, this is nothing new, this has been my whole life.    The man has never said, ' good job sagan, I knew you could do it!'    EVER. ...",
]
values = ["hedonism", "achievement", "power", "benevolence", "universalism"]
gpv = GPV()
results = gpv.measure_texts(texts, values)
```

