#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import urllib.parse
from ddgs import DDGS  #search engine to get the information from. Provides a free and api-less alternative  
import requests
from bs4 import BeautifulSoup #for web scraping
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import re


# In[2]:


SEARCH_RESULTS = 6        #number of urls to check
PASSAGES_PER_PAGE = 4     #number of passages to pull from each url
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" #model that will be used
TOP_PASSAGES = 5          #number of passages to use for the summary
SUMMARY_SENTENCES = 5     #number of sentences to include in each summary
TIMEOUT = 8               #time to wait for a page to load


# In[3]:


def unwrap_ddg(url):
    """Takes a duckduckgo wrapper and returns instead the direct link"""
    try:
        parsed = urllib.parse.urlparse(url) #part the url 
        #get the first half of the url 
        if parsed.netloc() == "duckduckgo.com":
            qs = urllib.parse.parse_qs(parsed.query) #turn the second half into a dictionary with key uddg
            uddg = qs.get("uddg") #get the value linked to the uddg key 
            if uddg: #check that there is a url linked with it
                return urllib.parse.unquote(uddg[0]) #decode the value which contains the url
    except Exception:
        pass

    return url


# In[4]:


def search_web(query, max_results=SEARCH_RESULTS):
    urls = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=max_results): #for through each dictionary in the library
            #print(result)
            url = result.get("href") or result.get("url")
            if not url:
                continue 
            url = unwrap_ddg(url)
            urls.append(url)

        return urls



query = "the history of ballet"
result = search_web(query)
print(result)


# In[ ]:


def fetch_text(url, timeout=TIMEOUT):
    headers = {"User-Agent": "Mozilla/5.0 (research-agent)"}
    try:
        res = requests.get(url=url, headers=headers, timeout=timeout)
        #check what status received. If anything else than 200 then return 
        if res.status_code != 200:
            return ""

        ct = res.headers.get("content-type", "")
        if "html" not in ct.lower(): #check if anything aside from an html page was returned 
            return ""

        soup = BeautifulSoup(res.text, "html.parser")

        #remove all the non needed tags 
        for tags in soup(["script", "style", "noscript", "header", "footer", "svg", "iframe", "nav", "aside"]):
            tags.extract()

        #get the paragraphs in the test 
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join([p for p in paragraphs if p])

        if text.strip(): #get rid of white spaces
            return re.sub(r"/s+", " ", text).strip()

        #in the case where we dont have clear p tags 
        meta = soup.find("meta", attrs={"name":"description"}) or soup.find("meta", attrs={"proprety": "og:description"})
        if meta and meta.get("content"):
            return meta["content"].strip()
        #else just return the title 
        if soup.title and soup.title.string:
            return soup.title.string.strip()

    except Exception:
        return ""

    return ""


# In[22]:


def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def chunks_passage(sentence, max_words=120):
    words = sentence.split()
    chunks = []
    i = 0

    while i<len(words):
        chunk = words[i : i+max_words]
        chunks.append(" ".join(chunk))
        i += max_words

    return chunks


# In[26]:


class ResearchAgent: 
    def __init__(self, embed_model=EMBEDDING_MODEL):
        print(f"Loading model: {embed_model}")
        self.embedder = SentenceTransformer(embed_model)

    def run(self, query):
        start = time.time()

        #search 
        urls = search_web(query)
        print(f"{len(urls)} were found.")

        #fetch and chunk 
        docs = []
        for url in urls: 
            text = fetch_text(url)
            if not text:
                continue
            chunks = chunks_passage(text)
            for c in chunks[:PASSAGES_PER_PAGE]:
                docs.append({"url":url, "passage":c})


        if not docs:
            print("No documents fetched")
            return {"query": query, "passages": [], "summary": ""}

        #embedding the text
        texts = [d["passage"] for d in docs]
        emb_texts = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        emb_query = self.embedder.encode([query], convert_to_numpy=True)[0]

        #get the cosine similarity and rank them 
        def cosine(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

        similarities = [cosine(emb_t, emb_query) for emb_t in emb_texts]
        ranked_idx = np.argsort(similarities)[::-1][:TOP_PASSAGES]
        top_passages = [{"url": docs[i]["url"], "passage": docs[i]["passage"], "score":float(similarities[i])} for i in ranked_idx]


        #summarizing (given a passage which sentence best describes the query)
        sentences = []
        for t in top_passages:
            for s in split_sentences(t["passage"]):
                sentences.append({"sent":s, "url":t["url"]})

        if not sentences:
            summary = "No summary could be generated"
        else:
            sent_texts = [s["sent"] for s in sentences]
            sent_embs = self.embedder.encode(sent_texts, convert_to_numpy=True)
            sent_similarity = [cosine(s, emb_query) for s in sent_embs]

            top_sent_idx = np.argsort(sent_similarity)[::-1][:SUMMARY_SENTENCES]
            chosen = [sentences[idx] for idx in top_sent_idx]

            #remove duplicates and format
            seen = set()
            lines = []
            for s in chosen:
                key = s["sent"].lower()[:80] #only first 80 characters checked for duplicates
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"{s["sent"]} (Source: {s["url"]})")
            summary = " ".join(lines)

        elapsed = time.time() - start
        return {"query": query, "passages": top_passages, "summary": summary, "time": elapsed}






# In[28]:


#run the agent and get the results 
agent = ResearchAgent()
query = "What are the origins of christmas?"
print(f"Running query: {query}\n")
out = agent.run(query)

print("\nTop passages:")
for p in out["passages"]:
    print(f"- score {p['score']:.3f} src {p['url']}\n  {p['passage'][:200]}...\n")


print("--- Extractive summary ---")
print(out["summary"])
print("--------------------------")
print(f"\nDone in {out['time']:.1f}s")

