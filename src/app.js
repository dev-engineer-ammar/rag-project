import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';

import * as dotenv from 'dotenv';
dotenv.config();

async function indexDocument() {
  const loader = new PDFLoader('src/assets/mern.pdf');
  const docs = await loader.load();
  console.log(docs.length);

  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 , chunkOverlap: 200 });
  const splits = await textSplitter.splitDocuments(docs);
  console.log(splits.length);

console.log("API KEY:", process.env.GEMINI_API_KEY);

  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model:  "text-embedding-004"
  });

const testVector = await embeddings.embedQuery();
console.log("Vector Length:", testVector.length);
  console.log(embeddings);


// const pc = new Pinecone();
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});
const index = pc.Index(process.env.PINECONE_INDEX_NAME);
console.log("Index:", process.env.PINECONE_INDEX_NAME);
await PineconeStore.fromDocuments(splits, embeddings, {
  pineconeIndex: index,
  maxConcurrency: 5
});


// console.log(vectorStore)


}

indexDocument();

