from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()  # Load environment variables from .env file


def main():
    information = """
    Anura Kumara Dissanayake[b] (born 24 November 1968), commonly referred to by his initials AKD, is a Sri Lankan politician who has served as the president of Sri Lanka since 2024.[2][3][4][5] Dissanayake is the first Sri Lankan president to be elected in a second round of vote counting,[6][7] and the first not to be a member of the traditional political parties of Sri Lanka.[8] He is also the ninth executive president of Sri Lanka, a constitutional distinction that separates the executive presidency established in 1978 from the earlier non-executive presidency.[9][10][11]

    Born on 24 November 1968 in the village of Galewela in the Central Province of Sri Lanka, Dissanayake moved with his family to the village of Thambuththegama in the North Central Province in 1972.[12] While he was a student at the University of Peradeniya, he joined the Socialist Students Union, the student wing of the Janatha Vimukthi Peramuna (JVP) as a student when the Indo-Sri Lanka Accord was signed in 1987[13] and engaged in clandestine revolutionary activity until he was forced to go underground due to threats posed by the counterinsurgency operations undertaken by the government. A year later, he transferred to the University of Kelaniya. After he was awarded his bachelor's degree in science at the University of Kelaniya in 1995, he was elected the national organiser of the Socialist Students Union in 1997.[14]

    Dissanayake was selected to the Central Committee of the JVP in 1997.[15] He was then appointed to the JVP Political Bureau in 1998[16] and entered parliament from the national list in 2000. He was elected to parliament in 2004 from the Kurunegala District with the highest number of preferential votes and served as Minister of Agriculture, Livestock, Land and Irrigation from 2004 to 2005. In 2008, he was appointed the leader of the JVP in parliament. He entered again Parliament through the national list in 2010. In 2014, he became the leader of the JVP[17] and was elected to Parliament from the Colombo District in 2015. He served as Chief Opposition Whip from 2015 to 2018 and was selected as the most active Member of Parliament continuously for five years in a row. In 2019, he became the founding leader of the National People's Power (NPP).[18]

    Dissanayake ran for president in the 2019 presidential election and came in third place with 3.16% of the vote. He ran again in 2024 and was elected as President of Sri Lanka on 22 September 2024, becoming the first president to be elected from a third party.[19]
    """
    summary_template = """
    given the information {information} about a  person I want you to create:
    1. A short summary of the person in 2-3 sentences.
    2. Two interesting facts about them.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
