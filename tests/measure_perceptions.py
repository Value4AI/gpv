from pprint import pprint

from gpv import GPV



perceptions = [
    "I love helping others",
    "I am always happy",
    "I hate my job",
    "I am a very successful person",
]

values = ["self-direction", "stimulation", "hedonism", "achievement", "power", "security", "conformity", "tradition", "benevolence", "universalism"]


gpv = GPV()
results = gpv.measure_perceptions(perceptions, values)
pprint(results)