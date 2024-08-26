from transformers import pipeline
from typing import List

class Summarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initializes the summarization pipeline with the specified model.
        
        Args:
            model_name (str): The model to use for the summarization pipeline.
        """
        self.summarizer = pipeline("summarization", model=model_name, device=1)
    
    def summarize(self, articles: List[str], batch_size: int = 8, max_length: int = 130, min_length: int = 30, do_sample: bool = False) -> List[str]:
        """
        Summarizes a batch of articles.
        
        Args:
            articles (List[str]): A list of articles to summarize.
            batch_size (int): The number of articles to process in each batch.
            max_length (int): The maximum length of the summary.
            min_length (int): The minimum length of the summary.
            do_sample (bool): Whether or not to sample during the summarization process.
            
        Returns:
            List[str]: A list of summarized texts.
        """
        summaries = []
        # Process articles in batches
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            # Generate summaries for the current batch
            batch_summaries = self.summarizer(batch, max_length=max_length, min_length=min_length, do_sample=do_sample)
            # Extract summary text from each result
            summaries.extend(batch_summaries)
        
        return summaries

# Example usage:
if __name__ == "__main__":
    ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
""" 

    articles = [ARTICLE] * 10  # Example list of 10 articles

    summarizer = Summarizer()
    summaries = summarizer.summarize(articles, batch_size=3)
    
    for summary in summaries:
        print(summary)
