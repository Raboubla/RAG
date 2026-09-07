import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Tout les configuration dans la page
st.set_page_config(page_title="RAG Local - Clone NotebookLM", layout="wide")

def extract_documents(uploaded_files):
    """Extrait le texte des fichiers uploadés par l'utilisateur."""
    docs = []
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        
        # Les Loaders de LangChain nécessitent un chemin de fichier, 
        # donc utilisation de fichier temporels
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
            
        try:
            if file_extension.lower() == ".pdf":
                loader = PyMuPDFLoader(temp_file_path)
            elif file_extension.lower() in [".txt", ".md"]:
                # TextLoader mais pas le Markdown simple car ca gere mieux le TXT
                loader = TextLoader(temp_file_path, encoding="utf-8")
            else:
                continue
                
            loaded_docs = loader.load()
            
            # remplacement par le vrai nom du fichier pour les métadonnees
            for doc in loaded_docs:
                doc.metadata['source'] = uploaded_file.name
                
            docs.extend(loaded_docs)
        finally:
            os.remove(temp_file_path)
            
    return docs

def process_and_vectorize(documents):
    """Découpe les documents et les stocke dans une base vectorielle Chroma."""
    # Découpage
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Vectorisation
        # HuggingFaceEmbeddings 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Stocage dans ChromaDB (base local)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vectorstore, chunks

# Initialisation de l'historique de chat et du vectorstore
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# BARRE LATÉRALE
with st.sidebar:
    st.title("Paramètres & Documents")
    
    # Zone de téléchargement de fichiers
    uploaded_files = st.file_uploader(
        "Chargez vos documents", 
        type=["pdf", "txt", "md"], 
        accept_multiple_files=True
    )
    
    # Bouton d'indexation
    if st.button("Indexer les documents", use_container_width=True):
        if uploaded_files:
            with st.spinner("Traitement des documents en cours..."):
                # Extraction
                documents = extract_documents(uploaded_files)
                
                # Chunking et Vectorisation
                vectorstore, chunks = process_and_vectorize(documents)
                
                # Sauvegarde du vectorstore dans la session Streamlit
                st.session_state.vectorstore = vectorstore
                
                st.success(f"Indexation terminée : {len(chunks)} fragments (chunks) générés à partir de {len(documents)} page(s)/document(s).")
                st.info("Prochaine étape : Mode Recherche Sémantique (Étape 3).")
        else:
            st.warning("Veuillez charger au moins un document avant d'indexer.")
            
    st.divider()
    
    # Toggle d'activation/désactivation du LLM
    llm_enabled = st.toggle("Activer l'Assistant RAG (LLM)", value=False)
    
    if llm_enabled:
        st.success("Mode : Assistant RAG Complet")
    else:
        st.info("Mode : Recherche Sémantique Pure")

# ZONE PRINCIPALE
st.title("Assistant RAG Local ETU002744")
st.markdown("Projet Mr Tsinjo : Introduction et Pratique de l'IA")

# Affichage de l'historique conversationnel
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Si la réponse contient des sources, on affiche l'expander dans l'historique
        if "sources" in message:
            with st.expander("Voir les extraits ayant servi de contexte"):
                for i, src in enumerate(message["sources"]):
                    st.markdown(f"**Extrait {i+1} - Fichier : `{src['source']}`**\n> {src['content']}\n")

# Saisie utilisateur
if prompt := st.chat_input("Posez une question sur vos documents..."):
    # Affichage et sauvegarde du message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generation de la réponse de l'assistant
    with st.chat_message("assistant"):
        if st.session_state.vectorstore is None:
            response = "Veuillez d'abord charger et indexer des documents dans la barre latérale."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            if llm_enabled:
                # Generation LLM (RAG Complet)
                from langchain_core.prompts import PromptTemplate
                from langchain_community.llms import Ollama
                
                # Récupération des fragments pertinents
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # Prompt pro max via chat :
                template = """Tu es un assistant utile, précis et fiable. 
Tu dois répondre à la question de l'utilisateur en te basant EXCLUSIVEMENT sur le contexte fourni ci-dessous. 
Si le contexte ne contient pas la réponse ou ne permet pas de répondre, dis simplement "Je suis désolé, mais l'information ne se trouve pas dans les documents fournis." N'invente JAMAIS d'informations.

Contexte :
{context}

Question :
{question}

Réponse :"""
                prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])
                final_prompt = prompt_template.format(context=context, question=prompt)
                
                # Envoi au modèle local
                llm = Ollama(model="mistral")
                
                with st.spinner("Réflexion en cours ..."):
                    llm_response = llm.invoke(final_prompt)
                    
                st.markdown(llm_response)
                
                # Transparence : Affichage visuel des extraits
                sources_data = [{"source": d.metadata.get("source", "Inconnu"), "content": d.page_content} for d in relevant_docs]
                
                with st.expander("Voir les extraits ayant servi de contexte"):
                    for i, src in enumerate(sources_data):
                        st.markdown(f"**Extrait {i+1} - Fichier : `{src['source']}`**\n> {src['content']}\n")
                        
                # Sauvegarde dans l'historique
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": llm_response,
                    "sources": sources_data
                })
            else:
                # Recherche Sémantique Pure (sans LLM)
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.invoke(prompt)
                
                response = f"Résultats bruts de la recherche pour : *{prompt}*\n\n"
                for i, doc in enumerate(relevant_docs):
                    source_file = doc.metadata.get('source', 'Fichier inconnu')
                    chunk_content = doc.page_content
                    
                    response += f"**Extrait {i+1} - Fichier : `{source_file}`**\n> {chunk_content}\n\n---\n"
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
