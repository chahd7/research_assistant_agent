The goal of this project is to build, using the all-MiniLM-L6-v2 embedding model, the DuckDuckGo API, and Retrieved-Augmented system, an agent that has the role of a research assistant. It will: 
* Turn your query into a vector.
* Search the web, scrape the text, and turn all the content into more vectors.
* Find the text vectors whose coordinates are closest to the query’s coordinates.

Given a query, the duckduckgo api will fetch the internet with the goal of providing the top 6 urls that match the closest the given query. DuckDuckGo was chosen as search engine as it provides a free and easily accessible API for this role. 
The urls are then unwrapped to get rid of the duckduckgo wrapper and obtain the direct urls constituing the uddg using the urllib library. 

Using the request library, we request if then sent to each url with the goal of obtaining a response. It is checked if the response code is 200 (showcasing that it has been successful) and that 
the type of response obtained is in an html format, to avoid obtaining other types of formats such as pdf and zip. Using the BeautifulSoup library, the paragraphs are then extracted and combined together into one text which is then stripped away from all whitespaces. 
In the case where no paragraph tags were available, the meta tag is explored to obtain any type of information through the description. In the case where that is not available as well, the title is then returned. To deal with long articles, the text obtained from each url is split into chunks.

The search agent will first search for the urls using the query, before fetching for the text and splitting them into different chunks to create the content store. The different passages and the query are then embedded using vector embedding and their similarity is calculated using cosine similarity. 
Using the score obtained for each passage, they are then ranked based on how important they are and while providing the urls from which they were obtained. The same process is then done to each sentence to regroup into a summary the sentences that are the closest to the query. 
