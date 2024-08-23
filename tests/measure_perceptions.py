import warnings
import logging
import pandas as pd
import datetime
import json
from tqdm import tqdm
from pprint import pprint
import time

from gpv import GPV



perceptions = [
    "I love helping others",
    "I am always happy",
    "I hate my job",
    "I am a very successful person",
]

values = ["self-direction", "stimulation", "hedonism", "achievement", "power", "security", "conformity", "tradition", "benevolence", "universalism"]


gpv = GPV(parsing_model_name='gemma-7b')

start = time.time()

results = gpv.measure_perceptions(perceptions, values)

pprint(results)

print("Time taken:", time.time() - start)
