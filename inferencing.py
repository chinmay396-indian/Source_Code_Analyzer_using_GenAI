from langchain.llms.ctransformers import CTransformers
from langchain.memory import ConversationSummaryMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain


#inferencing
def inferencing(vector_db, user_input):
    llm = CTransformers(model="C:\Chinmay Backup\Chinmay\Data Science\GEN AI\LLMs on CPU\model\llama-2-7b-chat.ggmlv3.q4_0.bin",
                 model_type = "llama",
                 config={'max_new_tokens':250,
                      'temperature':0.01})
    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history",return_messages=True)
    vector_db = vector_db
    
    qa = ConversationalRetrievalChain.from_llm(llm=llm,retriever =vector_db.as_retriever(search_type="mmr", search_kwargs={"k":3}),memory=memory)
    result = qa(user_input) 

    return str(result["answer"])