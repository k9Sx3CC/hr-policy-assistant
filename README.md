📘 HR Policy Assistant — RAG with Streamlit, FAISS, Sentence Transformers & Groq

A practical Retrieval-Augmented Generation (RAG) application that lets a user upload an HR Policy PDF and ask questions about it.

The app:

Extracts PDF text with PyMuPDF

Splits the text into overlapping chunks

Converts chunks into vector embeddings with Sentence Transformers

Stores/searches embeddings using FAISS

Retrieves the most relevant policy sections

Sends only the retrieved policy context to Groq

Generates an answer with OpenAI GPT-OSS 20B (openai/gpt-oss-20b)

Shows the retrieved source pages so the answer can be checked against the policy

Architecture

                 ┌─────────────────────┐
                 │   HR Policy PDF     │
                 └──────────┬──────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   PyMuPDF     │
                    │ Text Extract  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Chunking      │
                    │ 220 words     │
                    │ 40 overlap    │
                    └───────┬───────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Sentence Transformer │
                 │ all-MiniLM-L6-v2     │
                 └──────────┬───────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │     FAISS     │
                    │ Vector Search │
                    └───────┬───────┘
                            │
                     Top 5 chunks
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Groq GPT-OSS 20B     │
                 │ Grounded Generation  │
                 └──────────┬───────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Streamlit UI  │
                    │ Answer + Pages│
                    └───────────────┘

Project files

hr-policy-assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

Features

PDF upload through the browser

Page-aware PDF extraction

Overlapping text chunks

Semantic vector retrieval

FAISS similarity search

Groq openai/gpt-oss-20b

Answers restricted to retrieved policy context

PDF page references

Retrieved chunks visible for verification

No database required

No local model download for the LLM

Suitable for deployment on Streamlit Community Cloud

Important limitation

This version expects a text-based PDF.

If the HR policy is a scanned/image-only PDF, PyMuPDF may return little or no text. In that situation, OCR should be added as a separate preprocessing step.

API key

The app requires a Groq API key.

The Python code reads:

GROQ_API_KEY

For Streamlit Community Cloud, add it through the app's Secrets settings.

Example secret:

GROQ_API_KEY = "your_groq_api_key_here"

Do not put the real API key in app.py, README.md, or GitHub.

How RAG works in this project

1. Ingestion

The uploaded PDF is read page-by-page.

Each extracted chunk keeps its original PDF page number.

2. Embedding

The application uses:

sentence-transformers/all-MiniLM-L6-v2

to transform policy chunks into numerical vectors.

3. Vector search

FAISS stores the vectors in an inner-product index.

Embeddings are normalized, so the inner product acts as cosine similarity.

The application retrieves the top 5 most relevant chunks for each question.

4. Generation

Only the retrieved chunks are passed to GPT-OSS 20B.

The system prompt explicitly tells the model:

do not invent policy details

do not use outside knowledge to fill missing information

say when the policy does not contain the answer

mention PDF page numbers when possible

This makes the application a grounded RAG assistant rather than a generic chatbot.

Deploy without VS Code, Colab, or Terminal

You can build and deploy this project entirely through the GitHub website and Streamlit Community Cloud.

Step 1 — Create a GitHub repository

Go to GitHub and create a new repository.

Recommended name:

hr-policy-assistant

You can make it public if you want the project to be visible in your portfolio.

Step 2 — Add the four files

Inside the repository, create these files:

app.py
requirements.txt
README.md
.gitignore

Copy the contents from this project into the corresponding files.

Do not rename:

app.py
requirements.txt

because Streamlit Cloud will use app.py as the entry point.

Step 3 — Get a Groq API key

Create/sign in to your Groq account and create an API key.

Keep the key private.

Do not paste it into GitHub.

Step 4 — Open Streamlit Community Cloud

Open:

https://share.streamlit.io/

Sign in and connect your GitHub account.

Step 5 — Create the Streamlit app

Choose:

Create app

Then select:

Repository: your-username/hr-policy-assistant

Branch: main

Main file path: app.py

Choose a custom app URL if you want one.

Click Deploy.

Step 6 — Add the Groq secret

After the app is created:

Open the app's settings/menu.

Find Secrets.

Add:

GROQ_API_KEY = "your_groq_api_key_here"

Save the secret.

Reboot/redeploy the app if necessary.

The secret should never be committed to GitHub.

Step 7 — Test the app

Upload an HR policy PDF.

The app should display:

PDF page count

number of chunks

extracted character count

knowledge base status

Then ask questions such as:

How many annual leave days are employees entitled to?

or:

What is the maternity leave policy?

or:

What is the probation period?

The answer should be based on retrieved sections from the uploaded document.

Updating the deployed application

You do not need Git, VS Code, or a terminal.

Edit app.py directly on GitHub:

Open the repository.

Open app.py.

Click the edit/pencil button.

Make your change.

Click Commit changes.

Streamlit Community Cloud detects repository changes and updates the deployed app.

If you change requirements.txt, Streamlit Cloud will reinstall the dependencies and redeploy the app.

Security notes

Never commit a Groq API key.

Use Streamlit Secrets for deployment.

Do not upload confidential HR documents to a public demo unless you have permission.

This project is a technical demonstration and should not replace official HR or legal review.

Uploaded documents are processed by the running Streamlit application and relevant text is sent to Groq for answer generation.

Troubleshooting

GROQ_API_KEY is not configured

Open Streamlit Cloud → App settings → Secrets and add:

GROQ_API_KEY = "your_key"

No extractable text found

The PDF is probably scanned/image-only.

Use a text-based PDF or add an OCR pipeline.

FAISS installation error

Make sure faiss-cpu is present in requirements.txt.

App takes time to start

The Sentence Transformer model has to be downloaded the first time it is loaded. Streamlit's resource cache prevents repeated model loading during normal app use.

The answer is not correct

Open Retrieved policy sections below the answer.

Check whether the relevant policy section was retrieved.

If the correct section is missing, improve retrieval by changing:

chunk size

chunk overlap

top-k

embedding model

Suggested portfolio description

HR Policy Assistant — RAG Application
Built a Streamlit-based Retrieval-Augmented Generation application that extracts HR policy documents with PyMuPDF, creates Sentence Transformer embeddings, performs semantic retrieval with FAISS, and generates grounded policy answers using Groq's OpenAI GPT-OSS 20B model. The application displays retrieved policy sections and PDF page references to improve answer traceability.

Technologies

Python

Streamlit

PyMuPDF

Sentence Transformers

FAISS

Groq API

OpenAI GPT-OSS 20B

Retrieval-Augmented Generation (RAG)

GitHub

Streamlit Community Cloud
