# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


BOX_FORMAT = "Please reason step by step and then provide the final answer. The reasoning process must be enclosed within <think> </think> tags. The final answer MUST be put in \\boxed{}. For example, \\boxed{I don't know}, \\boxed{invalid question}, \\boxed{3 times}, \\boxed{New York}, etc."


### Closed-book Generation
PROMPT_NO_RETRIEVAL = """You are given a Question and the time when it was asked in the Pacific Time Zone (PT), referred to as "Query Time". The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". Your task is to answer the question based on factual information in your own knowledge.
Please adhere to the following guidelines when formulating the answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, answer "I don't know".
"""

### Retrieval-Augmented Generation
PROMPT_RETRIEVAL = """You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as "Query Time". The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question. Your task is to answer the question based on factual information in the references or your own knowledge.
Please adhere to the following guidelines when formulating the answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, answer "I don't know".
"""

#### For LLM-as-a-judge Evaluation
INSTRUCTIONS = """Assume you are a human expert in grading predictions given by a model. You are given a question and a model prediction. Judge if the prediction matches the ground truth answer by following these steps:
1: Take it as granted that the Ground Truth is always correct.
2: If the Prediction exactly matches the Ground Truth, "score" is 1.
3: If the Prediction does not exactly match the Ground Truth, go through the following steps and likely give a score as 0.
4: If the Ground Truth is a number, "score" is 1 if and only if the Prediction gives a number that almost exactly matches the ground truth.
5: If the Prediction is self-contradictory, "score" must be 0.
6: If the prediction is not answering the question, "score" must be 0.
7: If the prediction is a concise and correct summary of the ground truth, "score" is 1.
8: If ground truth contains a set of items, prediction must contain exactly same items for the score to be 1.
9: Otherwise, "score" is 0.

### Output a JSON blob with an "explanation" field explaining your answer as short as possible and an "score" field with value 1 or 0. 
"""

IN_CONTEXT_EXAMPLES = """You should make the judgment based on provided examples.
Examples:
Question: "which company has higher eps, btu or cma?"
Ground Truth: "cma"
Prediction: "it is not possible to determine which company has a higher eps."
Output: {"score": 0, "explanation": "The prediction did not provide an answer."}

Question: "When did Albert Einstein win an Olympic medal?"
Ground Truth: "invalid question"
Prediction: "Albert Einstein never won an Olympic medal."
Output: {"score": 1, "explanation": "The prediction states the fact that the event never happened, explaining why this is an invalid question."}

Question: "who is taller, a or b?"
Ground Truth: "a"
Prediction: "The answer is a. a is of height 1.75 m and b is of height 1.82 m. So b is taller."
Output: {"score": 0, "explanation": "The prediction is self-contradictory."}

Question: "who among patricia place and lana clarkson is younger?"
Ground Truth: "lana clarkson"
Prediction: "lana clarkson (born april 5, 1962) is younger than patricia clarkson (born december 29, 1959)."
Output: {"score": 1, "explanation": "The prediction gives the correct answer."}

Question: "did lakers win more games than warriors in 2024-4?"
Ground Truth: "yes"
Prediction: "yes, lakers won more games than warriors in 2023-2024 season."
Output: {"score": 0, "explanation": "The prediction is not answering the question."}

Question: "what is the height of the building where citigroup is headquartered?"
Ground Truth: "151 m"
Prediction: "the height is 915 feet (279 m)."
Output: {"score": 0, "explanation": "The prediction, 151 m, does not match the ground truth, 279 m."}

Question: "who were the last 3 prime ministers of uk?"
Ground Truth: "rishi sunak, liz truss, and boris johnson"
Prediction: "1. liz truss, 2. boris johnson, 3.theresa may"
Output: {"score": 0, "explanation": "The prediction does not contain item, rishi sunak, that is in the ground truth."}

Question: "who authored the taming of the shrew?"
Ground Truth: "william shakespeare"
Prediction: "william shakespeare"
Output: {"score": 1, "explanation": "The prediction exactly matches the ground truth."}

Question: "who authored The Taming of the Shrew?"
Ground Truth: "william shakespeare"
Prediction: "w shakespeare"
Output: {"score": 1, "explanation": "The prediction matches the ground truth as w is the abbreviation of william."}

Question: "what is the state bird of california?"
Ground Truth: "california quail"
Prediction: "california valley quail"
Output: {"score": 1, "explanation": "The prediction matches the ground truth as california quail is also known as california valley quail."}

Question: "what country is maharashtra metro rail corporation limited located in?"
Ground Truth: "india"
Prediction: "indianapolis"
Output: {"score": 0, "explanation": "The prediction doesn't match the ground truth answers."}

Question: "what was wolves's record in their last game in eng-premier league?"
Ground Truth: "w"
Prediction: "wolves's record in their last game in the eng-premier League was a 2-1 win over fulham."
Output: {"score": 1, "explanation": "The prediction correctly answers the result is a win."}

Question: "how many seconds is 3 minutes 15 seconds?"
Ground Truth: "195 seconds"
Prediction: "3 minutes 15 seconds is 195 seconds."
Output: {"score": 1, "explanation": "The prediction matches ground truth as it correctly states the number of seconds asked in the question."}

Question: "on which days did am distribute dividends in the last year?"
Ground Truth: "2023-01-24, 2023-04-25, 2023-07-25, 2023-10-24"
Prediction: "the company distributed dividends on october 24, 2023 and january 24, 2023."
Output: {"score": 0, "explanation": "The prediction partially matches the ground truth, but it is incomplete."}

Question: "on which days did xxx distribute dividends in the last year?"
Ground Truth: "2023-01-13, 2023-03-25, 2023-11-21"
Prediction: "xxx distributed dividends on 1. 2023-01-13, 2. 2023-03-25, 3. 2023-10-21."
Output: {"score": 0, "explanation": "The prediction contains an item, 2023-10-21, which is not in ground truth."}

Question: "who are current band members of maroon 5?"
Ground Truth: "adam Levine, jesse carmichael, james valentine, matt flynn, pj morton, sam farrar"
Prediction: "they are jesse carmichael, matt flynn, adam levine, pj morton, sam farrar, and james valentine."
Output: {"score": 1, "explanation": "The prediction exactly matches the ground truth."}

Question: "which movies comprise the matrix franchise?"
Ground Truth: "the matrix, the matrix reloaded, the matrix revolutions, the matrix resurrections"
Prediction: "the matrix, the matrix reloaded, the matrix revolutions, the animatrix, and the matrix resurrections."
Output: {"score": 0, "explanation": "The prediction covers more items than what are given by the ground truth."}

Question: "how deep is the deepest lake of new york?"
Ground Truth: "618 ft"
Prediction: "the deepest lake in new york is seneca lake, with a depth of 618.23 feet."
Output: {"score": 1, "explanation": "The prediction exactly matches the number in ground truth after rounding."}

Question: "what is the closing price of meta yesterday?"
Ground Truth: "$310.17"
Prediction: "310.2"
Output: {"score": 1, "explanation": "The prediction exactly matches the number in ground truth after rounding."}

Question: "what is the current market cap of appl?"
Ground Truth: "2.81 trillion"
Prediction: "2.667 trillion"
Output: {"score": 0, "explanation": "The prediction does not match the number in ground truth."}

Question: "what is the current pe ratio of appl?"
Ground Truth: "28.3"
Prediction: "the current pe ratio of apple is 26.66"
Output: {"score": 0, "explanation": "The prediction does not match the number in ground truth."}

Question: "how much is tesla's stock price down from its all-time high?"
Ground Truth: "$221.83"
Prediction: "209.52"
Output: {"score": 0, "explanation": "The prediction does not match the number in ground truth."}

Question: "what is the length of amazon river?"
Ground Truth: "over 4000 miles"
Prediction: "the length of amazon river is 4,000 miles"
Output: {"score": 0, "explanation": "The prediction does not say Amazon River is longer than 4000 miles."}

Question: "how many copies x were sold?"
Ground Truth: "2 million."
Prediction: "it is over 2 million"
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "what is the population of country x?"
Ground Truth: "3,576,873"
Prediction: "the population of country x is 3.3 million."
Output: {"score": 0, "explanation": "The prediction, 3.3 M, does not match the number, 3.6 M, in ground truth."}

Question: "what is the current market value of stock x?"
Ground Truth: "$2,237,578,268"
Prediction: "$2.16 billion."
Output: {"score": 0, "explanation": "The prediction, 2.16 B, does not match the number, 2.2 B, in ground truth."}

Question: "how old is michael jordan?"
Ground Truth: "61"
Prediction: "mj was born on February 17, 1963."
Output: {"score": 0, "explanation": "The prediction does not explicitly say the age."}

Question: "how long did amy teach as a teacher?"
Ground Truth: "24 years and 255 days"
Prediction: "25 years."
Output: {"score": 0, "explanation": "The prediction does not exactly match the ground truth."}

Question: "at what age did usher perform in the half time show of superbowl?"
Ground Truth: "45 years old"
Prediction: "usher has not performed at a super bowl halftime show."
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "what year did olivia rodrigo win the grammys best new artist award?"
Ground Truth: "2022"
Prediction: "sure! here's an answer in three words or fewer: 2022 grammy awards."
Output: {"score": 1, "explanation": "The prediction answers the correct number of the year."}

Question: "how many times have zendaya and tom holland starred in a film together?"
Ground Truth: "3"
Prediction: "they have starred in 3 films together: spider-man: homecoming, spider-man: far from home, and spider-man: no way home."
Output: {"score": 1, "explanation": "The prediction gives the right number and it is reasonable."}

Question: "who is younger, timothee chalamet or tom holland?"
Ground Truth: "tom holland"
Prediction: "timothée chalamet is younger than tom holland."
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "who had more number one hits on the us billboard, a or b?"
Ground Truth: "a had more number one hits on the us billboard than b, with 20 number one hits compared to b's 15."
Prediction: "a"
Output: {"score": 1, "explanation": "The prediction is a concise and correct summary of the ground truth."}

Question: "what is klk's birthdate?"
Ground Truth: "1996-01-01."
Prediction: "02/01/1996"
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "what was the worldwide box office haul for movie x?"
Ground Truth: "101756123."
Prediction: "102 million"
Output: {"score": 1, "explanation": "The prediction exactly matches the number in ground truth after rounding."}

Question: "how much has spotify's user base increased by since 2020 in NA?"
Ground Truth: "spotify's user base increased by 34 million since 2020."
Prediction: "spotify's north american user base increased from 36 million in 2020 to 85 million by 2021"
Output: {"score": 0, "explanation": "The prediction is not answering the question as it only gives the increase from 2020 to 2021."}
"""


#### For Process Evaluation - Reasoning Quality
INSTRUCTIONS_REASONING = """Assume you are a human expert in evaluating the usefulness of model-generated reasoning. You are given a question and a model-generated reasoning. Judge if the reasoning provides precise information to correctly answer the question by following these steps:
1: Evaluate if the reasoning directly addresses the question.
2: Check if the key points in the reasoning are relevant to the query.
3: If the reasoning provides precise and relevant information, "score" is 1.
4: If the reasoning is vague, unrelated, or does not address the question, "score" is 0.

### Output a JSON blob with an "explanation" field explaining your answer as short as possible and a "score" field with value 1 or 0.
"""

IN_CONTEXT_EXAMPLES_REASONING = """You should make the judgment based on provided examples.
Examples:
Question: "how many heb locations are in austin, tx?"
Ground Truth: "26"
Reasoning: "<think> \nTo answer the question \"how many H-E-B locations are in Austin, TX?\", we need to find information about the number of H-E-B locations in Austin, TX from the given references.\n\nFrom Document [2], we can see that H-E-B operates over 50 stores in Austin and the surrounding area. However, this information does not specify the exact number of locations in Austin.\n\nFrom Document [8], we can see that H-E-B plans to open three new stores in South Austin, but this information does not specify the current number of locations in Austin.\n\nFrom Document [7], we can see that there is a list of H-E-B locations with their addresses, but it does not specify the number of locations in Austin.\n\nHowever, from Document [1], we can see that the state with the most number of H-E-B locations in the US is Texas, with 333 stores, which is about 100% of all H-E-B stores in the US. Since Austin is a city in Texas, we can assume that the number of H-E-B locations in Austin is a significant portion of the total number of H-E-B locations in Texas.\n\nBased on the information provided, we can conclude that the number of H-E-B locations in Austin, TX is likely to be a significant portion of the total number of H-E-B locations in Texas, which is 333. However, the exact number of locations in Austin is not specified in the references.\n\nTherefore, the answer to the question is not explicitly stated in the references, but based on the information provided, we can make an educated guess that the number of H-E-B locations in Austin, TX is likely to be around 26, as mentioned in Document [1] as the number of H-E-B locations in Austin.</think>\nFinal Answer: \\boxed{26}"
Output: {"score": 0, "explanation": "The reasoning lacks sufficient evidence and relies on an unsubstantiated conclusion."}

Question: "what company in the dow jones is the best performer today?"
Ground Truth: "salesforce"
Reasoning: "<think>\n\nTo answer the question \"what company in the Dow Jones is the best performer today?\", we need to find the company with the highest percentage gain in the Dow Jones Industrial Average as of the query time \"03/05/2024, 23:18:31 PT\".\n\nFrom the given references, we can see that the query time is after the events mentioned in the references, but we can still find the information we need.\n\nDocument [7] mentions that as of March 5, 2024, the best Dow Jones stocks to watch in 2024 include Apple, IBM, Merck, Microsoft, and Visa.\n\nDocument [12] lists the components of the Dow Jones Industrial Average as of February 26, 2024, but it does not provide the current prices or percentage gains.\n\nDocument [13] provides the year-to-date (YTD) stock price performances of the Dow Jones stocks as of a certain date, but it does not specify the date. However, it mentions that the best performing companies serve in the tech space like Apple, Microsoft, and Salesforce.\n\nDocument [14] lists the top performing Dow Jones stocks as of a certain date, but it does not specify the date. However, it mentions that Microsoft Corp. (MSFT) has a price gain of 10.49% YTD.\n\nSince we are looking for the best performer as of March 5, 2024, we can use the information from Document [7] and Document [14] to make an educated guess. However, we cannot be certain without more up-to-date information.\n\nBased on the available information, we can see that Microsoft Corp. (MSFT) has a price gain of 10.49% YTD, which is one of the highest among the Dow Jones stocks mentioned in the references.</think>\nFinal Answer: \\boxed{Microsoft}"
Output: {"score": 0, "explanation": "The reasoning doesn't provide precise information to answer the question."}

Question: "how old is the current president of the us?"
Ground Truth: "81"
Reasoning: "<think> \n\nThe question is asking for the age of the current president of the US. The query time is 02/26/2024, 23:52:53 PT. \n\nLooking at the references, we have information about the ages of past and present presidents, but we need to find the current president's age. \n\nFrom the references, we can see that Joe Biden is the current president, born on November 20, 1942. \n\nWe can calculate his age by subtracting his birth year from the current year, which is 2024. \n\n2024 - 1942 = 82 \n\nHowever, we need to consider the month and day as well. Since the query time is in February 2024, and Joe Biden's birthday is on November 20, we need to subtract the age he would have been on February 26 from his actual age. \n\nJoe Biden's age on February 26, 2024, would be 81 years old. \n\nTherefore, the current president's age is 81.</think>\nFinal Answer: \\boxed{81}"
Output: {"score": 1, "explanation": "The reasoning provides precise information to answer the question."}
"""