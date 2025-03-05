# TAPLA: Travel planning AI Agent   
This project is a simple demonstration on how we can easily build an LLM-powered AI Agent with Hugging Face and smolagents.
This simple Agent can use customized tools, making use of the [Amadeus API](https://developers.amadeus.com/) to achieve tasks such as:
- Find available flights between two destination cities and figure out the best rates for a given date
- Find the best available hotel deals on the destination city

## Install requirements
```
pip install -r requirements.txt
```

## Amadeus API
**NOTE:** To test this application you need some API keys:
- **Hugging Face**: Your API key from Hugging Face to get access to LLMs
- **Amadeus API key**: An API key from Amadeus(testing key is fine).

## Use the agent to plan your trip!
You can use the agent to plan your trip by executing the code cells in the [notebook guide](./agent.ipynb).
Note that for this simple demo application we make use of the testing API from [Amadeus](https://developers.amadeus.com/), therefore
flights and hotel availability are **fake data**!
