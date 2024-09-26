import requests
import time
from pprint import pprint
import random
from multiprocessing import Pool

DATA = [
    "John Smith visited Paris last summer. He works for Google in their New York office.",
    "The United Nations held a conference in Geneva. Switzerland is known for its beautiful Alps.",
    "Apple Inc. released a new iPhone model. Tim Cook made the announcement at their headquarters in Cupertino.",
    "Shakespeare wrote Romeo and Juliet. The play is set in Verona, Italy.",
    "NASA launched a mission to Mars. The Perseverance rover landed in the Jezero crater.",
    "The Beatles performed their last concert on the rooftop of Apple Corps in London. John Lennon was wearing his famous round glasses.",
    "Mount Everest is the highest peak in the world. It's located in the Himalayas, bordering Nepal and Tibet.",
    "The Louvre Museum in Paris houses the Mona Lisa. Leonardo da Vinci painted this masterpiece in the early 16th century.",
    "Microsoft CEO Satya Nadella spoke at the World Economic Forum in Davos. He discussed the future of artificial intelligence.",
    "The Great Wall of China stretches over 13,000 miles. It was built during the Ming Dynasty to protect against Mongol invasions.",
    "Elon Musk founded SpaceX in 2002. The company's Falcon 9 rocket regularly launches from Cape Canaveral in Florida.",
    "The Amazon rainforest spans across several South American countries. It's home to countless species, including the jaguar and the giant otter.",
    "Queen Elizabeth II resided in Buckingham Palace. The royal residence is located in the City of Westminster, London.",
    "The World Health Organization declared COVID-19 a pandemic in March 2020. Dr. Tedros Adhanom Ghebreyesus made the announcement from Geneva, Switzerland.",
    "The Golden State Warriors won the NBA championship. Stephen Curry was named the Finals MVP at the Chase Center in San Francisco.",
    "Leonardo DiCaprio starred in the movie Titanic. The film, directed by James Cameron, depicted the sinking of the RMS Titanic in the North Atlantic Ocean.",
    "The Eiffel Tower was built for the 1889 World's Fair in Paris. Gustave Eiffel's company designed and constructed the iconic landmark.",
    "The International Space Station orbits the Earth every 92 minutes. Astronauts from NASA, Roscosmos, and other space agencies conduct experiments in microgravity.",
    "The Taj Mahal is located in Agra, India. Emperor Shah Jahan commissioned the mausoleum in memory of his wife Mumtaz Mahal.",
    "Mark Zuckerberg founded Facebook at Harvard University. The social media giant is now part of Meta Platforms, headquartered in Menlo Park, California.",
    "The United States of America declared independence on July 4, 1776. The Declaration of Independence was primarily written by Thomas Jefferson, with contributions from John Adams and Benjamin Franklin. The document was adopted by the Continental Congress in Philadelphia, Pennsylvania. This historic event marked the beginning of the American Revolutionary War against Great Britain, led by King George III. General George Washington commanded the Continental Army, eventually leading the colonies to victory and the formation of a new nation.",
    "The human genome was first sequenced in 2003 by the Human Genome Project, a collaborative effort involving researchers from the United States, United Kingdom, Japan, France, and Germany. The project was led by Francis Collins at the National Institutes of Health and James Watson at Cold Spring Harbor Laboratory. This groundbreaking achievement in molecular biology and genetics has had far-reaching implications for medicine, biotechnology, and our understanding of human evolution. Since then, companies like 23andMe and Ancestry.com have made personal genomics accessible to the general public, while institutions such as the Broad Institute continue to push the boundaries of genetic research."
]

def make_request(url, token):
    data = {
        "inputs": DATA[random.randint(0, len(DATA) - 1)]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    start_time = time.time()
    response = requests.post(url, json=data, headers=headers)
    # print(response.json())
    end_time = time.time()
    return 

if __name__ == "__main__":
    token = "pb_XlsoE0-CVLqm60TZtW7hGg"
    url = "https://serving.staging.predibase.com/7c7efa/deployments/v2/llms/gemma-crusoe/classify"
    num_trials = 10000

    start = time.time()
    with Pool(500) as pool:
        pool.starmap(make_request, [(url, token)] * num_trials)
    end = time.time()

    average_time = (end-start) / num_trials
    # pprint(results[:2])
    print(f"Number of requests: {len(times)}")
    print(f"Average time taken: {average_time} seconds")
    print(f"Requests per second: {1 / average_time}")
