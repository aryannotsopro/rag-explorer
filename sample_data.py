"""
sample_data.py - Populate Pinecone with sample documents for testing.

Usage:
    python sample_data.py           # Ingest sample data
    python sample_data.py --clear   # Clear the index first, then ingest
    python sample_data.py --check   # Only show index stats
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import List

from config import get_settings
from data_ingestion import DocumentProcessor
from models import DocumentCategory, DocumentChunk, DocumentMetadata, FileType
from rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Sample documents  — (title, category, author, date_offset_days, content)
# --------------------------------------------------------------------------- #
SAMPLE_DOCUMENTS = [
    # ── Technology ────────────────────────────────────────────────────────── #
    {
        "title": "Introduction to Large Language Models",
        "category": DocumentCategory.TECHNOLOGY,
        "author": "Dr. Sarah Chen",
        "days_ago": 10,
        "tags": ["llm", "ai", "nlp"],
        "content": """
Large language models (LLMs) are a class of neural network models trained on massive text corpora
to predict the next token in a sequence. They have revolutionized natural language processing by
demonstrating emergent capabilities such as in-context learning, chain-of-thought reasoning,
and few-shot generalisation.

Modern LLMs such as GPT-4, Claude, and Gemini are built on the Transformer architecture,
introduced by Vaswani et al. in 2017. The core mechanism is self-attention, which allows the
model to weigh the importance of different tokens when predicting the next one.

Training involves two phases: (1) pre-training on web-scale text using a causal language modelling
objective, and (2) fine-tuning with human feedback (RLHF) or direct preference optimisation (DPO)
to align the model with human values and instructions.

Key scaling laws show that model performance improves predictably with more parameters, more data,
and more compute. However, the efficiency frontier has shifted dramatically with techniques like
mixture-of-experts (MoE), speculative decoding, and quantization (INT4/INT8).

Challenges include hallucinations (generating plausible but incorrect facts), context length
limitations, biases from training data, and the environmental cost of training at scale.

Retrieval Augmented Generation (RAG) is a popular technique to ground LLM responses in verified
external knowledge bases, reducing hallucinations and enabling dynamic knowledge updates without
expensive retraining.
        """,
    },
    {
        "title": "Transformer Architecture Deep Dive",
        "category": DocumentCategory.TECHNOLOGY,
        "author": "Prof. Michael Torres",
        "days_ago": 25,
        "tags": ["transformer", "attention", "deep learning"],
        "content": """
The Transformer architecture fundamentally changed sequence modelling by replacing recurrence
with attention mechanisms, enabling massive parallelisation during training.

Multi-head self-attention allows the model to jointly attend to information from different
representation subspaces. Each attention head computes:

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

where Q (queries), K (keys), and V (values) are projections of the input embeddings.

Positional encodings inject sequence order information since self-attention is permutation-invariant.
Modern models use Rotary Position Embeddings (RoPE) or ALiBi for improved long-context handling.

The feed-forward network (FFN) in each layer is a two-layer MLP applied independently to each
position. In GPT-style models, GELU activation is preferred over ReLU for smoother gradients.

Layer normalisation (pre-norm vs post-norm) significantly affects training stability. Most modern
models use pre-norm (normalise before each sub-layer), which allows stable training at large scale.

Mixture-of-Experts (MoE) transformer variants use sparse routing: each token activates only a
small subset of "experts" (FFN layers), dramatically increasing model capacity without proportional
compute increase.
        """,
    },
    {
        "title": "Vector Databases and Semantic Search",
        "category": DocumentCategory.TECHNOLOGY,
        "author": "Aisha Robinson",
        "days_ago": 5,
        "tags": ["vector-db", "pinecone", "embeddings", "search"],
        "content": """
Vector databases store high-dimensional embedding vectors and enable approximate nearest-neighbour
(ANN) search at scale. They are the backbone of modern semantic search and RAG systems.

Unlike traditional relational databases that match on exact values, vector databases measure
semantic similarity using distance metrics:
  - Cosine similarity: measures the angle between two vectors (preferred for text embeddings)
  - Euclidean (L2): measures geometric distance
  - Dot product: useful when embeddings are not normalised

Leading vector databases include Pinecone (managed, serverless), Weaviate (open-source, rich
schema), Qdrant (Rust-based, high performance), Milvus (distributed), and pgvector (PostgreSQL
extension for hybrid SQL + vector queries).

Pinecone's serverless architecture auto-scales storage and query capacity independently of compute.
It supports metadata filtering via a MongoDB-style filter language, allowing hybrid attribute +
vector searches without post-filtering performance penalties.

ANN algorithms used under the hood include HNSW (Hierarchical Navigable Small World graphs),
which builds a multi-layer proximity graph for sub-linear search time, and IVF (Inverted File
Index) with Product Quantisation for memory-efficient approximate search.

Metadata filtering is crucial for production RAG systems: it narrows the candidate space before
ANN search, dramatically improving both precision and latency.
        """,
    },
    {
        "title": "Cloud Computing and Microservices",
        "category": DocumentCategory.TECHNOLOGY,
        "author": "James Park",
        "days_ago": 40,
        "tags": ["cloud", "aws", "kubernetes", "microservices"],
        "content": """
Cloud computing delivers computing resources—servers, storage, networking, and software—over the
internet on a pay-per-use basis. The three primary service models are:

1. Infrastructure as a Service (IaaS): Raw compute (EC2, GCE) and storage (S3, GCS).
2. Platform as a Service (PaaS): Managed runtimes, databases, and middleware (Heroku, Google App Engine).
3. Software as a Service (SaaS): End-user applications delivered via browser (Gmail, Salesforce).

Microservices architecture decomposes applications into small, independently deployable services
that communicate via lightweight APIs (REST or gRPC). Benefits include independent scaling,
technology heterogeneity, and fault isolation.

Kubernetes (K8s) orchestrates containerised microservices across a cluster, managing scheduling,
auto-scaling, self-healing, and rolling deployments. Key primitives are Pods, Deployments, Services,
ConfigMaps, and Persistent Volume Claims.

Service meshes (Istio, Linkerd) add observability, traffic management, and mTLS encryption
transparently, without application code changes.

Twelve-factor app methodology (12factor.net) defines best practices: store config in env vars,
back services as attached resources, log to stdout, and keep dev/prod parity.
        """,
    },
    {
        "title": "Cybersecurity Fundamentals",
        "category": DocumentCategory.TECHNOLOGY,
        "author": "Dr. Elena Vasquez",
        "days_ago": 18,
        "tags": ["security", "cryptography", "owasp"],
        "content": """
Cybersecurity protects computer systems, networks, and data from digital attacks, theft, and damage.
The CIA triad—Confidentiality, Integrity, Availability—is the foundational security model.

Common attack vectors:
  - SQL Injection: Inserting malicious SQL to manipulate database queries.
  - Cross-Site Scripting (XSS): Injecting scripts into web pages viewed by other users.
  - Phishing: Social engineering to trick users into revealing credentials.
  - Man-in-the-Middle (MitM): Intercepting communication between two parties.
  - Denial of Service (DoS/DDoS): Overwhelming a service with traffic.

Cryptography underpins secure communications. Symmetric encryption (AES-256) is fast but requires
secure key exchange. Asymmetric encryption (RSA, ECC) solves key exchange via public/private key
pairs. TLS (Transport Layer Security) combines both: asymmetric for handshake, symmetric for bulk data.

Zero Trust security model assumes no implicit trust, even inside the corporate network. Every request
is authenticated and authorised based on identity, device posture, and context.

The OWASP Top 10 is an industry-standard document listing the most critical web application security
risks, updated regularly to reflect evolving threat landscapes.
        """,
    },

    # ── Science ───────────────────────────────────────────────────────────── #
    {
        "title": "Quantum Computing Principles",
        "category": DocumentCategory.SCIENCE,
        "author": "Prof. David Kimura",
        "days_ago": 35,
        "tags": ["quantum", "qubits", "physics"],
        "content": """
Quantum computing harnesses the principles of quantum mechanics—superposition, entanglement, and
interference—to perform computations that are intractable for classical computers.

A qubit can exist in a superposition of |0⟩ and |1⟩ simultaneously, unlike a classical bit.
The state of a qubit is a unit vector in a 2D complex Hilbert space: |ψ⟩ = α|0⟩ + β|1⟩,
where |α|² + |β|² = 1.

Quantum entanglement links two qubits such that the measurement of one instantly determines the
state of the other, regardless of physical distance—a phenomenon Einstein called "spooky action
at a distance."

Quantum gates (Hadamard, CNOT, Toffoli) manipulate qubits; a quantum circuit is a sequence of
such gates. Unlike classical gates, quantum gates are reversible (unitary).

Key quantum algorithms:
  - Shor's algorithm: Factors large integers in polynomial time, threatening RSA encryption.
  - Grover's algorithm: Searches an unsorted database in O(√N) time vs O(N) classically.
  - Quantum Fourier Transform: Core subroutine of many quantum algorithms.

Current quantum hardware uses superconducting qubits (IBM, Google), trapped ions (Quantinuum, IonQ),
and photonic approaches. Decoherence and gate error rates remain the primary challenges.
        """,
    },
    {
        "title": "CRISPR Gene Editing Technology",
        "category": DocumentCategory.SCIENCE,
        "author": "Dr. Maria Santos",
        "days_ago": 60,
        "tags": ["crispr", "genetics", "biology"],
        "content": """
CRISPR-Cas9 is a revolutionary gene-editing technology derived from a natural bacterial immune
system. It allows precise modification of DNA sequences in living organisms.

CRISPR (Clustered Regularly Interspaced Short Palindromic Repeats) are DNA sequences found in
bacteria that store fragments of viral DNA as "memories." The Cas9 protein acts as molecular
scissors, cutting double-stranded DNA at a specific location guided by a single guide RNA (sgRNA).

The editing process: (1) Design an sgRNA complementary to the target DNA sequence. (2) Deliver
Cas9+sgRNA complex into the target cell. (3) Cas9 cuts both DNA strands at the target site.
(4) The cell's natural repair mechanisms (NHEJ or HDR) fix the break, editing the gene.

Applications span medicine (treating sickle cell disease, blindness, cancer), agriculture
(disease-resistant crops), and basic research (creating animal models of human diseases).

Nobel Prize in Chemistry 2020 was awarded to Jennifer Doudna and Emmanuelle Charpentier for
developing the CRISPR-Cas9 gene editing method.

Ethical concerns include germline editing (heritable changes), off-target effects, and equitable
access. The 2018 controversy around He Jiankui's CRISPR babies catalysed international governance
discussions.
        """,
    },
    {
        "title": "Climate Science and Global Warming",
        "category": DocumentCategory.SCIENCE,
        "author": "Dr. Amara Osei",
        "days_ago": 15,
        "tags": ["climate", "environment", "carbon"],
        "content": """
Climate science studies the long-term patterns of Earth's atmosphere, oceans, and land surface.
The Intergovernmental Panel on Climate Change (IPCC) synthesises thousands of peer-reviewed studies
to inform global climate policy.

The greenhouse effect: Solar radiation warms the Earth's surface, which emits infrared radiation.
Greenhouse gases (CO₂, CH₄, N₂O, water vapour) absorb this radiation and re-emit it in all
directions, trapping heat in the atmosphere.

Since pre-industrial times, atmospheric CO₂ has risen from ~280 ppm to over 420 ppm—a 50% increase
driven by burning fossil fuels, deforestation, and industrial processes. Global average surface
temperature has risen approximately 1.1°C.

Consequences include: more extreme weather events, sea-level rise (thermal expansion + ice melt),
shifts in precipitation patterns, ocean acidification, and biodiversity loss.

IPCC's 1.5°C pathway requires cutting global emissions by 45% from 2010 levels by 2030 and
reaching net zero by 2050. Key solutions: renewable energy transition, carbon capture, electrified
transport, sustainable land use, and energy efficiency improvements.
        """,
    },
    {
        "title": "The Human Microbiome",
        "category": DocumentCategory.SCIENCE,
        "author": "Prof. Lei Zhang",
        "days_ago": 45,
        "tags": ["microbiome", "biology", "health"],
        "content": """
The human microbiome is the collection of trillions of microorganisms—bacteria, archaea, viruses,
and fungi—that colonise our bodies, primarily the gut. The gut microbiome contains approximately
38 trillion bacterial cells, comparable in number to human cells.

The gut microbiome influences digestion, immune function, mood (gut-brain axis), and susceptibility
to disease. Key bacterial phyla include Firmicutes, Bacteroidetes, Actinobacteria, and Proteobacteria.

Dysbiosis (microbial imbalance) is associated with conditions including inflammatory bowel disease,
obesity, type 2 diabetes, autism spectrum disorder, and depression.

The gut-brain axis is a bidirectional communication network linking enteric nervous system,
vagus nerve, immune system, and microbial metabolites (short-chain fatty acids, neurotransmitters).
Approximately 90% of serotonin is produced by gut enterochromaffin cells, influenced by microbiome.

Faecal microbiota transplantation (FMT) has shown high efficacy (>90%) for recurrent Clostridioides
difficile infections and is being investigated for conditions from IBD to metabolic syndrome.

Diet is the most modifiable factor influencing microbiome composition. High-fibre diets promote
diversity and beneficial species like Bifidobacterium and Lactobacillus.
        """,
    },

    # ── History ───────────────────────────────────────────────────────────── #
    {
        "title": "The Roman Empire: Rise and Fall",
        "category": DocumentCategory.HISTORY,
        "author": "Prof. Giovanni Russo",
        "days_ago": 90,
        "tags": ["rome", "ancient", "empire"],
        "content": """
The Roman Empire, at its peak under Emperor Trajan (117 AD), controlled approximately 5 million
km² and a population of 70 million people. Its legacy—law, language, architecture, governance—
shaped Western civilisation for two millennia.

Rome transitioned from Republic to Empire in 27 BC when Augustus Caesar became the first Roman
Emperor after the civil wars following Julius Caesar's assassination in 44 BC. The Pax Romana
(27 BC – 180 AD) brought two centuries of relative peace and prosperity.

The empire's success rested on a professional standing army (legions), extensive road network
(85,000 km), sophisticated administration, and Roman law (later forming the basis of legal systems
across Europe and Latin America).

Decline involved multiple factors: military overextension, economic strain, political instability
(the "Crisis of the Third Century" saw 26 emperors in 50 years), pandemic disease (Antonine and
Plague of Cyprian), and sustained barbarian pressure.

The Western Roman Empire fell in 476 AD when Germanic chieftain Odoacer deposed Romulus Augustulus.
The Eastern Roman (Byzantine) Empire survived until 1453 AD when Constantinople fell to the Ottoman
Turks under Sultan Mehmed II.
        """,
    },
    {
        "title": "The Industrial Revolution",
        "category": DocumentCategory.HISTORY,
        "author": "Dr. Charlotte Williams",
        "days_ago": 55,
        "tags": ["industrial", "history", "britain", "economics"],
        "content": """
The Industrial Revolution (c. 1760–1840) transformed Britain—and subsequently the world—from
agrarian economies into industrial powerhouses. It marked the most profound shift in human living
standards since the Neolithic agricultural revolution.

Key innovations: James Watt's improved steam engine (1769) powered factories and locomotives.
Richard Arkwright's water frame (spinning machines) mechanised textile production. The Bessemer
process (1856) enabled mass production of steel.

The factory system replaced cottage industries, concentrating workers in urban mills. Manchester,
the "Cottonopolis," exemplified this transformation—its population grew from 25,000 in 1772 to
over 300,000 by 1850.

Social consequences were profound: rapid urbanisation created overcrowded slums, child labour
was widespread, working hours were brutal (14–16 hours/day), but living standards gradually rose.
The middle class expanded; new working class consciousness emerged.

The revolution spread to Belgium, France, Germany, and the United States through the 19th century,
each nation adapting the model to its resources and institutions.

The railways revolutionised transport—Stephenson's Rocket (1829) achieved 48 km/h—shrinking time
and space, creating national markets, and enabling mass movement of goods and people.
        """,
    },
    {
        "title": "World War II: Global Conflict and Its Aftermath",
        "category": DocumentCategory.HISTORY,
        "author": "Prof. Hans Mueller",
        "days_ago": 30,
        "tags": ["ww2", "history", "conflict", "cold war"],
        "content": """
World War II (1939–1945) was the deadliest conflict in human history, involving over 30 countries
and resulting in 70–85 million fatalities—approximately 3% of the 1940 world population.

The war began when Nazi Germany invaded Poland on 1 September 1939, prompting Britain and France
to declare war. Germany's early Blitzkrieg tactics—coordinated air and armoured assaults—overwhelmed
conventional defences, capturing most of Western Europe by 1940.

The turning points came in 1942–1943: the Soviet victory at Stalingrad (the largest single battle
in history, with ~2 million casualties), the Allied victory at El Alamein in North Africa, and
the American naval victory at Midway, halting Japanese Pacific expansion.

The Holocaust was the systematic genocide of 6 million European Jews and millions more (Roma,
disabled people, political prisoners), organised by the Nazi regime under Adolf Hitler.

The atomic bombings of Hiroshima (6 August 1945) and Nagasaki (9 August 1945) killed 100,000–
200,000 people and led to Japan's surrender, ending the war in the Pacific.

The war's aftermath reshaped the world order: the United Nations was founded, the Cold War between
the US and Soviet Union began, European colonial empires started collapsing, and the Marshall Plan
rebuilt Western Europe. The Nuremberg Trials established principles of international criminal law.
        """,
    },
    {
        "title": "Ancient Egypt: Civilisation on the Nile",
        "category": DocumentCategory.HISTORY,
        "author": "Dr. Fatima Al-Hassan",
        "days_ago": 75,
        "tags": ["egypt", "ancient", "archaeology", "pharaoh"],
        "content": """
Ancient Egyptian civilisation flourished along the Nile for over 3,000 years (c. 3100–30 BC),
making it one of humanity's longest-enduring cultures. The Nile's predictable annual floods
deposited rich silt, making the valley extraordinarily fertile in an otherwise arid landscape.

The civilisation is divided into three major kingdoms: Old Kingdom (pyramid age, c. 2686–2181 BC),
Middle Kingdom (c. 2055–1650 BC), and New Kingdom (empire, c. 1550–1069 BC), separated by periods
of fragmentation called Intermediate Periods.

The pyramids of Giza—including the Great Pyramid of Khufu (c. 2560 BC), one of the Seven Wonders
of the Ancient World—demonstrate extraordinary engineering mastery. The Great Pyramid contains
approximately 2.3 million stone blocks, each averaging 2.5–15 tonnes.

Egyptian religion centred on a pantheon of gods (Ra, Osiris, Isis, Horus, Anubis) and belief in
the afterlife. Mummification preserved the body for the soul's return. The Book of the Dead provided
spells to navigate the underworld to a paradise called the Field of Reeds (Aaru).

Hieroglyphics were the formal writing system, used for monuments and religious texts. The Rosetta
Stone (196 BC), discovered in 1799, allowed Champollion to decipher hieroglyphics in 1822.

Egypt was conquered by Alexander the Great in 332 BC. The subsequent Ptolemaic dynasty ended with
Cleopatra VII's defeat by Rome in 30 BC, making Egypt a Roman province.
        """,
    },
    {
        "title": "The Silk Road: Ancient Trade Networks",
        "category": DocumentCategory.HISTORY,
        "author": "Dr. Mei Ling",
        "days_ago": 100,
        "tags": ["trade", "silk road", "china", "history"],
        "content": """
The Silk Road was not a single road but a vast network of trade routes stretching ~6,400 km
from Chang'an (modern Xi'an, China) to Antioch (modern Turkey), connecting East Asia, South Asia,
Central Asia, the Middle East, East Africa, and Europe.

Active from the 2nd century BC to the 15th century AD, it facilitated the exchange not only of
goods—silk, spices, glassware, precious metals, cotton—but also ideas, religions, art, and disease.

China's silk was extraordinarily valuable in Rome; Roman senators were known to pay its weight in
gold. The Chinese jealously guarded silk-production secrets until Byzantine monks allegedly
smuggled silkworm eggs out of China in hollow canes around 550 AD.

Buddhism spread from India to Central and East Asia along these routes. Islam later diffused
westward. The exchange of paper, printing, and gunpowder westward from China transformed European
civilisation.

The Black Death (bubonic plague) is believed to have spread from Central Asia along Silk Road trade
networks in the 14th century, killing an estimated one-third of Europe's population.

The route declined after the fall of the Mongol Empire (which had provided safe passage and
reduced tariffs across Eurasia) and the rise of direct sea routes following Vasco da Gama's
voyage to India (1498).
        """,
    },
    {
        "title": "The Space Race: Cold War in Space",
        "category": DocumentCategory.HISTORY,
        "author": "Dr. Nikolai Petrov",
        "days_ago": 20,
        "tags": ["space", "nasa", "soviet", "cold war", "technology"],
        "content": """
The Space Race was a competition between the United States and Soviet Union for supremacy in
space exploration, driven by Cold War rivalry and technological prestige (1957–1969).

The Soviet Union struck first: Sputnik 1 (4 October 1957), Earth's first artificial satellite,
shocked the West. One month later, Sputnik 2 carried Laika the dog into orbit. In April 1961,
Yuri Gagarin became the first human in space aboard Vostok 1—a 108-minute flight.

The US responded by establishing NASA (1958) and accelerating its program. John Glenn became
the first American to orbit Earth in February 1962. President Kennedy's 1961 challenge—"We choose
to go to the Moon...in this decade"—galvanised the Apollo programme.

Apollo 11 (July 1969) achieved Kennedy's goal: Neil Armstrong became the first human to walk
on the Moon, followed by Buzz Aldrin, while Michael Collins orbited above. Armstrong's words—
"One small step for [a] man, one giant leap for mankind"—were broadcast to 600 million viewers.

The Saturn V rocket that propelled Apollo missions remains the most powerful rocket ever flown:
363 feet tall, generating 7.6 million pounds of thrust at liftoff.

Spinoff technologies from the Space Race include: memory foam, scratch-resistant lenses, water
filters, GPS (evolved from military satellite programmes), and MRI scanners.
        """,
    },
]


# --------------------------------------------------------------------------- #
# Main ingestion logic
# --------------------------------------------------------------------------- #

def create_chunks_from_samples() -> List[DocumentChunk]:
    """Convert the SAMPLE_DOCUMENTS list into DocumentChunk objects."""
    processor = DocumentProcessor()
    all_chunks: List[DocumentChunk] = []

    for i, doc in enumerate(SAMPLE_DOCUMENTS):
        logger.info(
            "[%d/%d] Processing '%s' …", i + 1, len(SAMPLE_DOCUMENTS), doc["title"]
        )
        content = doc["content"].strip()
        doc_date = (datetime.utcnow() - timedelta(days=doc["days_ago"])).strftime("%Y-%m-%d")
        source_name = doc["title"].replace(" ", "_").lower()[:40] + ".txt"

        # Use the splitter directly since we have raw text (no file load needed)
        raw_chunks = processor._splitter.split_text(content)
        total = len(raw_chunks)

        for j, chunk_text in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue
            metadata = DocumentMetadata(
                source=source_name,
                category=doc["category"],
                chunk_index=j,
                total_chunks=total,
                document_date=doc_date,
                author=doc.get("author"),
                file_type=FileType.TXT,
                word_count=len(chunk_text.split()),
                tags=doc.get("tags", []),
            )
            all_chunks.append(DocumentChunk(text=chunk_text.strip(), metadata=metadata))

    logger.info("Created %d chunks from %d documents.", len(all_chunks), len(SAMPLE_DOCUMENTS))
    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate Pinecone with sample RAG data.")
    parser.add_argument("--clear", action="store_true", help="Delete all existing vectors before ingesting.")
    parser.add_argument("--check", action="store_true", help="Only print index stats, do not ingest.")
    args = parser.parse_args()

    engine = RAGEngine()
    engine.ensure_index()
    stats = engine.index_stats()

    if args.check:
        print(f"\n📊 Index stats: {stats}\n")
        return

    if args.clear:
        logger.warning("--clear specified. Deleting all vectors in the index …")
        # Pinecone delete_all requires namespace; use empty string for default namespace
        engine.index.delete(delete_all=True)
        logger.info("Index cleared.")

    chunks = create_chunks_from_samples()

    logger.info("Upserting %d chunks to Pinecone …", len(chunks))
    upserted = engine.sync_upsert(chunks)
    logger.info("✅ Done! Upserted %d vectors.", upserted)

    stats_after = engine.index_stats()
    print(f"\n✅ Sample data ingested successfully!")
    print(f"   Documents: {len(SAMPLE_DOCUMENTS)}")
    print(f"   Chunks upserted: {upserted}")
    print(f"   Total vectors in index: {stats_after.get('total_vector_count', 'N/A')}")
    print(f"\nCategories covered:")
    cats = {}
    for doc in SAMPLE_DOCUMENTS:
        cats[doc["category"].value] = cats.get(doc["category"].value, 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"   {cat.title()}: {count} document(s)")
    print("\nRun 'streamlit run app.py' to start the UI.")


if __name__ == "__main__":
    main()
