# views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings

from .models import Message, ChatSession, Document
from .forms import SignupForm  # keep
from .utils import load_document

import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangDocument
import chromadb

def index(request):
    return render(request, 'chat_app/index.html')

def signup_view(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Account created! You can log in now.")
            return redirect('login')
        else:
            messages.error(request, "Please correct the errors.")
    else:
        form = SignupForm()
    return render(request, 'chat/signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('chat')
        else:
            messages.error(request, "Invalid credentials")
    return render(request, 'chat/login.html')

def logout_view(request):
    logout(request)
    return redirect('login')

# -----------------------------
# Gemini + Embeddings + Chroma
# -----------------------------
# 1) Gemini SDK (from env var)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# 2) HF embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    """Generate embeddings using Hugging Face (list[str] -> list[list[float]])"""
    return embedding_model.encode(texts).tolist()

# 3) Persistent Chroma and per-user collections
CHROMA_DIR = getattr(settings, "CHROMA_DIR", os.path.join(getattr(settings, "BASE_DIR", os.getcwd()), "chroma_db"))
os.makedirs(CHROMA_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

def get_user_collection(user):
    name = f"docs_user_{user.id}"
    try:
        return chroma_client.get_collection(name)
    except:
        return chroma_client.create_collection(name)

# -----------------------------
# Chat view
# -----------------------------
@login_required
def chat_view(request, session_id=None):
    # 1) Sidebar sessions
    if session_id:
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)

    else:
        session = None

    sessions = ChatSession.objects.filter(user=request.user).order_by("-created_at")



    # if session_id:
    #     session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    # else:
    #     session = sessions.first()
    #     if not session:
    #         session = ChatSession.objects.create(user=request.user, title="New Chat")
    #         return redirect("chat_view",session_id=session_id)



    # 2) POST: chat/doc
    if request.method == "POST":
        user_text = request.POST.get("message", "").strip()
        uploaded_file = request.FILES.get("document")

        # 2b) Handle document upload
        if uploaded_file:
            try:
                # Save file in DB (and to MEDIA_ROOT/documents/)
                doc_model = Document.objects.create(
                    user=request.user,
                    session=session,
                    file=uploaded_file,
                    title=uploaded_file.name,
                )

                # Load & split document
                file_path = doc_model.file.path
                docs = load_document(file_path)  # returns list[LangChain Document]  :contentReference[oaicite:3]{index=3}
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents(docs)

                # Prepare texts + embeddings
                texts = [c.page_content for c in chunks]
                embeddings = embed_texts(texts)

                # Per-user collection; add with IDs + metadata
                collection = get_user_collection(request.user)
                ids = [f"{doc_model.id}-{i}" for i in range(len(texts))]
                metadatas = [{"title": doc_model.title, "doc_id": doc_model.id, "user_id": request.user.id, "session_id": session.id,}] * len(texts)

                collection.add(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )

                Message.objects.create(
                    session=session,
                    user=request.user,
                    content=f"📄 Uploaded document: {doc_model.title}",
                    is_bot=False,
                )

                Message.objects.create(
                    session=session,
                    user=request.user,
                    content=f"I’ve processed your document **{doc_model.title}**. You can now ask me questions about it.",
                    is_bot=True,
                )
                messages.success(request, "Document uploaded and processed successfully.")
            except Exception as e:
                messages.error(request, f"Document upload failed: {e}")

        # 2c) Generate AI response (Normal chat + RAG)
        if user_text:
            try:

                if not session:
                    # Create new chat session on first message
                    session = ChatSession.objects.create(
                        user=request.user,
                        title=user_text[:30]  # first message as title (or default)
                    )

                Message.objects.create(
                    session=session,
                    user=request.user,
                    content=user_text,
                    is_bot=False,
                )



                # Conversation context (last 20)
                last_msgs = list(
                    Message.objects.filter(session=session).order_by("-timestamp")[:20]
                )[::-1]
                convo = "\n".join(
                    [("User: " if not m.is_bot else "Assistant: ") + m.content for m in last_msgs]
                )

                # Retrieve top 3 user-specific docs from Chroma
                collection = get_user_collection(request.user)
                query_embedding = embed_texts([user_text])[0]
                results = collection.query(query_embeddings=[query_embedding], n_results=3, where={"session_id": session.id})
                retrieved = results.get("documents", [[]])[0]
                context_docs = "\n\n".join(retrieved) if retrieved else ""

                # Build prompt (no trailing "AI:")
                prompt = f"""
You are a helpful assistant in a chat app. If the retrieved context is relevant, use it; otherwise answer normally.

Retrieved context (may be empty):
{context_docs if context_docs else "(none)"}

Conversation so far:
{convo}

User question: {user_text}

Answer directly. Do NOT prefix with "AI:".
""".strip()

                response = model.generate_content(prompt)
                ai_text = (getattr(response, "text", None) or "").strip()

                if not ai_text and getattr(response, "candidates", None):
                    ai_text = response.candidates[0].content.parts[0].text.strip()

                if not ai_text:
                    ai_text = "Sorry, I couldn't generate a response."

                # Save assistant message
                Message.objects.create(
                    session=session,
                    user=request.user,
                    content=ai_text,
                    is_bot=True,
                )
            except Exception as e:
                Message.objects.create(
                    session=session,
                    user=request.user,
                    content=f"Error generating AI response: {e}",
                    is_bot=True,
                )


        return redirect("chat_view", session.id)



    # 3) Render
    messages_list = Message.objects.filter(session=session).order_by("timestamp")
    user_documents = Document.objects.filter(session=session).order_by("-uploaded_at")
    return render(
        request,
        "chat/chat.html",
        {
            "sessions": sessions,
            "session": session,
            "chat_messages": messages_list,
            "documents": user_documents,
        },
    )
