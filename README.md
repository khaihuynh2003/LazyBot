install required python packages

Run in vscode terminal

prepare data: python ingest.py

run chatbot:

chainlit run model.py -w (retrieve data without history)

chainlit run model1.py -w (retrieve data with history)

chainlit run mod.py -w (plus conversation persistence and resume)
