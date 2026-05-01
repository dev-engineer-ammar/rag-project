import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";


import * as dotenv from 'dotenv';
dotenv.config();


 
async function indexDocument() {
  const loader = new PDFLoader('src/assets/mern.pdf');
  const docs = await loader.load();
  console.log(docs.length);

  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 , chunkOverlap: 200 });
  const splits = await textSplitter.splitDocuments(docs);
  console.log(splits);

  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    modelName: 'model: "text-embedding-004"'
  });
  

  console.log(embeddings);
}

indexDocument();