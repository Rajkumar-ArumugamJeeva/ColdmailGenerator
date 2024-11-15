import os

from langchain_groq import ChatGroq


from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(    model="llama-3.1-70b-versatile",temperature=0,groq_api_key=os.getenv("GROQ_API_KEY"))

    def extract_jobs(self, cleaned_text):
        prompt_extract=PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        The scraped  text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing
        following keys: 'role','experience','skills' and 'description'.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):
        """
        )
        chain_extract = prompt_extract | self.llm
        res=chain_extract.invoke(input={'page_data':cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res=json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("context too big. Unable to parse jobs.")
        return res if isinstance(res,list) else [res]
    
    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
        '''
        ### JOB DESCRIPTION:
        {job_description}
        ### INSTRUCTION:
        you are Rajkumar Arumugam Jeeva.
        I'm a passionate computer vision engineer pursuing my Master's in Applied Science in Biomedical Engineering at the
        University of Ottawa. I have two years of hands-on experience in computer vision, working on tasks like object detection,
        segmentation, and building machine learning models for classification. I’ve also worked with machine vision cameras and
        edge devices like Raspberry Pi, Pico, Arduino, and ESP modules. My research focuses on automatic video quality
        assessment for gait videos using pose estimation. I have a publication in IEEE on applying optical flow to laryngeal videos
        to detect abnormalities.
        your job is to write a cold email to the Hiring manager regarding the job mentioned describing the I'm the 
        suitable candidate for the  job.
        Also add the most relevant projects from the following links to showcase Rajkumar's  portfolio :{link_list}
        Remember, you are Rajkumar Arumugam Jeeva.
        Do not provise a preamble.
        ### EMAIL (NO PREAMBLE):   '''
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content
    





        

    

if __name__ == "__main__":

    print(os.getenv("GROQ_API_KEY"))
