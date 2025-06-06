from rag_pipeline import RAGPipeline


def main():
    """Simple example of using the RAG pipeline"""

    # Initialize the RAG pipeline
    print("Initializing RAG Pipeline...")
    rag = RAGPipeline()

    # Build the knowledge base from PDFs
    rag.build_knowledge_base()

    # Interactive Q&A loop
    while True:
        print("\n" + "="*40)
        question = input("Your Question (or 'quit' to exit): ").strip()

        if question.lower() in ['quit']:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        print(f"\n🔍 Searching for: '{question}'")
        print("⏳ Processing...")

        # Get answer from RAG pipeline
        result = rag.query(question)

        print("\n" + "="*60)
        print("📝 ANSWER:")
        print("="*60)
        print(result['answer'])

        # Display the chunks that were used
        if 'sources' in result and result['sources']:
            print("\n" + "="*60)
            print("📚 RETRIEVED CHUNKS:")
            print("="*60)
            for i, source in enumerate(result['sources'], 1):
                print(f"\n🔸 Chunk {i}:")
                print(f"   📄 File: {source['filename']}")
                print(f"   🔢 Chunk ID: {source['chunk_id']}")
                print(f"   📊 Similarity Score: {source['score']:.4f}")
                print(f"   📝 Content Preview:")
                print(f"      {source['content_preview']}")
                print("-" * 50)


if __name__ == "__main__":
    main()
