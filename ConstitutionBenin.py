# =========================================================
# 🇧🇯 Assistant Constitutionnel du Bénin
# Chatbot IA pour comprendre la Constitution
# =========================================================

# 📦 Importation des bibliothèques nécessaires
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# =========================================================
# ⚙️ Configuration de la page
# =========================================================
st.set_page_config(page_title="Assistant Constitutionnel du Bénin", page_icon="🇧🇯", layout="wide")

# =========================================================
# 🇧🇯 En-tête avec drapeau et titre
# =========================================================
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 15px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/0a/Flag_of_Benin.svg"
             alt="Drapeau du Bénin" width="150" style="border-radius: 10px;">
        <h1 style="margin-top: 10px;">⚖️ Assistant Constitutionnel du Bénin 🇧🇯</h1>
        <p style="font-size:17px;">Discutez avec l'assistant pour mieux comprendre la Constitution du Bénin.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 🔐 Clé API OpenAI
# =========================================================
openai_api_key = st.sidebar.text_input("🔑 Entrez votre clé OpenAI :", type="password")
if not openai_api_key:
    st.warning("⚠️ Entrez votre clé OpenAI pour utiliser l’assistant.")
    st.stop()

# =========================================================
# 📘 Téléversement du document PDF
# =========================================================
file = st.sidebar.file_uploader("📄 Téléversez la Constitution du Bénin (PDF)", type="pdf")

if not file:
    st.info("Veuillez téléverser la Constitution du Bénin pour commencer.")
    st.stop()

# =========================================================
# 🧩 Lecture et découpage du texte (avec cache)
# =========================================================
@st.cache_resource
def process_pdf(file):
    """Extrait le texte du PDF et le découpe en segments pour l'indexation"""
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Découpage du texte en petits morceaux (chunks)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Lecture et indexation
with st.spinner("📖 Lecture et indexation de la Constitution..."):
    chunks = process_pdf(file)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(chunks, embeddings)

# =========================================================
# 🧠 Mémoire conversationnelle (persiste sur plusieurs tours)
# =========================================================
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================================================
# 🤖 Modèle et chaîne de QA conversationnelle
# =========================================================
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0.1,
    model_name="gpt-4-turbo"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    memory=st.session_state.memory,
    verbose=False
)

# =========================================================
# 💡 Suggestions de questions (au début du chat)
# =========================================================
st.markdown("### 💬 Suggestions de questions pour commencer :")
col1, col2 = st.columns(2)

with col1:
    if st.button("Quels sont les droits fondamentaux garantis par la Constitution ?"):
        st.session_state["preset_question"] = "Quels sont les droits fondamentaux garantis par la Constitution du Bénin ?"

    if st.button("Comment le Président de la République est-il élu ?"):
        st.session_state["preset_question"] = "Comment le Président de la République du Bénin est-il élu ?"

    if st.button("Quelles sont les missions de la Cour Constitutionnelle ?"):
        st.session_state["preset_question"] = "Quelles sont les missions de la Cour Constitutionnelle du Bénin ?"

with col2:
    if st.button("Quels sont les symboles de l’État béninois ?"):
        st.session_state["preset_question"] = "Quels sont les symboles de l’État béninois selon la Constitution ?"

    if st.button("Que dit la Constitution sur la séparation des pouvoirs ?"):
        st.session_state["preset_question"] = "Que dit la Constitution du Bénin sur la séparation des pouvoirs ?"

    if st.button("Comment la Constitution peut-elle être révisée ?"):
        st.session_state["preset_question"] = "Comment la Constitution du Bénin peut-elle être révisée ?"

# =========================================================
# 💬 Zone de chat interactive (itérative)
# =========================================================
st.subheader("💭 Discussion interactive avec l’assistant")

# Afficher l’historique précédent
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Vérifier si une question pré-sélectionnée a été cliquée
preset_question = st.session_state.pop("preset_question", None)

# Champ d’entrée de message utilisateur
prompt = preset_question or st.chat_input("Posez votre question sur la Constitution du Bénin...")

if prompt:
    # Affichage du message utilisateur
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Traitement et réponse du modèle
    with st.spinner("🧠 L'assistant consulte la Constitution..."):
        response = qa_chain({"question": prompt})
        answer = response["answer"]

    # Affichage de la réponse
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Enregistrer la réponse dans l’historique
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# =========================================================
# 🔁 Bouton de réinitialisation
# =========================================================
st.sidebar.markdown("---")
if st.sidebar.button("♻️ Réinitialiser la conversation"):
    st.session_state.memory.clear()
    st.session_state.chat_history = []
    st.success("✅ Conversation réinitialisée avec succès.")
