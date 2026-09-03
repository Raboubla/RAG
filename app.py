import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

# Configuration de la page
st.set_page_config(page_title="RAG Local - Clone NotebookLM", page_icon="📚", layout="wide")

def extract_documents(uploaded_files):
    """Extrait le texte des fichiers uploadés par l'utilisateur."""
    docs = []
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        
        # Les Loaders de LangChain nécessitent un chemin de fichier, 
        # on utilise donc un fichier temporaire.
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
            
        try:
            if file_extension.lower() == ".pdf":
                loader = PyMuPDFLoader(temp_file_path)
            elif file_extension.lower() in [".txt", ".md"]:
                # TextLoader gère aussi bien le TXT que le Markdown simple
                loader = TextLoader(temp_file_path, encoding="utf-8")
            else:
                continue
                
            loaded_docs = loader.load()
            
            # On remplace le chemin temporaire par le vrai nom du fichier pour les métadonnées
            for doc in loaded_docs:
                doc.metadata['source'] = uploaded_file.name
                
            docs.extend(loaded_docs)
        finally:
            # Nettoyage du fichier temporaire
            os.remove(temp_file_path)
            
    return docs

# Initialisation de l'historique de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.title("⚙️ Paramètres & Documents")
    
    # 1. Zone de téléchargement de fichiers
    uploaded_files = st.file_uploader(
        "Chargez vos documents", 
        type=["pdf", "txt", "md"], 
        accept_multiple_files=True
    )
    
    # 2. Bouton d'indexation
    if st.button("Indexer les documents", use_container_width=True):
        if uploaded_files:
            with st.spinner("Extraction des documents en cours..."):
                # Étape 2.1 : Extraction
                documents = extract_documents(uploaded_files)
                st.success(f"✅ Extraction terminée : {len(documents)} page(s) / document(s) extrait(s).")
                st.info("🔜 Prochaine étape : Chunking (Découpage des textes).")
        else:
            st.warning("Veuillez charger au moins un document avant d'indexer.")
            
    st.divider()
    
    # 3. Toggle d'activation/désactivation du LLM
    llm_enabled = st.toggle("Activer l'Assistant RAG (LLM)", value=False)
    
    if llm_enabled:
        st.success("Mode : Assistant RAG Complet")
    else:
        st.info("Mode : Recherche Sémantique Pure")

# --- ZONE PRINCIPALE ---
st.title("📚 Assistant RAG Local")
st.markdown("Discutez avec vos documents (PDF, TXT, Markdown) en toute confidentialité.")

# Affichage de l'historique conversationnel
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Saisie utilisateur
if prompt := st.chat_input("Posez une question sur vos documents..."):
    # Affichage et sauvegarde du message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Simulation de la réponse de l'assistant (sera remplacée aux étapes 3 et 4)
    with st.chat_message("assistant"):
        if llm_enabled:
            response = f"*(Squelette)* Réponse générée par le LLM pour la requête : **{prompt}**"
        else:
            response = f"*(Squelette)* Résultats de la recherche sémantique pour : **{prompt}**"
            
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
