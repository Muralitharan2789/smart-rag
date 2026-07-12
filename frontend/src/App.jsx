import { useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = "http://localhost:8000";

function SourcesPanel({ sources }) {
  const [expandedIndex, setExpandedIndex] = useState(null);

  if (!sources || sources.length === 0) return null;

  return (
    <div style={{ marginTop: "0.5rem" }}>
      <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap" }}>
        {sources.map((src, i) => (
          <button
            key={i}
            onClick={() => setExpandedIndex(expandedIndex === i ? null : i)}
            style={{
              fontSize: "0.8rem",
              padding: "0.2rem 0.5rem",
              borderRadius: "12px",
              border: "1px solid #999",
              background: expandedIndex === i ? "#e0e0ff" : "#f0f0f0",
              cursor: "pointer",
            }}
          >
            Source {src.source_number}
          </button>
        ))}
      </div>

      {expandedIndex !== null && (
        <div
          style={{
            marginTop: "0.5rem",
            padding: "0.75rem",
            background: "#fafafa",
            border: "1px solid #ddd",
            borderRadius: "6px",
          }}
        >
          <div style={{ fontSize: "0.8rem", color: "#666", marginBottom: "0.4rem" }}>
            {sources[expandedIndex].document_name} · {sources[expandedIndex].chunk_type}
          </div>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {sources[expandedIndex].chunk_text_preview}
          </ReactMarkdown>
        </div>
      )}
    </div>
  );
}

function FileUpload({ onUploaded }) {
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState(null);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setStatus(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setStatus(
        `Uploaded: ${res.data.chunks_stored} chunks (${res.data.table_chunks} tables, ${res.data.text_chunks} text)`
      );
      if (onUploaded) onUploaded();
    } catch (err) {
      setStatus(`Upload failed: ${err.message}`);
    } finally {
      setUploading(false);
      e.target.value = ""; // allows re-uploading the same filename later if needed
    }
  };

  return (
    <div
      style={{
        marginBottom: "1rem",
        padding: "1rem",
        border: "1px dashed #999",
        borderRadius: "8px",
      }}
    >
      <input type="file" accept=".pdf,.docx" onChange={handleFileChange} disabled={uploading} />
      {uploading && <p style={{ color: "#888" }}>Uploading and indexing...</p>}
      {status && <p style={{ fontSize: "0.9rem" }}>{status}</p>}
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);

  const handleAsk = async () => {
    if (!question.trim()) return;

    const userMessage = { role: "user", text: question };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");
    setLoading(true);

    try {
      const res = await axios.post(`${API_BASE}/query`, {
        question: userMessage.text,
        top_k: 10,
        rerank_top_n: 5,
      });
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: res.data.answer, sources: res.data.sources },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `Error: ${err.message}`, sources: [] },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleAsk();
  };

  return (
    <div style={{ maxWidth: "700px", margin: "2rem auto", fontFamily: "sans-serif" }}>
      <h1>Smart RAG</h1>

      <FileUpload />

      <div
        style={{
          border: "1px solid #ccc",
          borderRadius: "8px",
          padding: "1rem",
          minHeight: "300px",
        }}
      >
        {messages.length === 0 && (
          <p style={{ color: "#888" }}>Ask a question about your uploaded documents.</p>
        )}
        {messages.map((msg, i) => (
          <div key={i} style={{ marginBottom: "1.5rem" }}>
            <strong>{msg.role === "user" ? "You" : "Smart RAG"}:</strong>
            <div>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
            </div>
            {msg.role === "assistant" && <SourcesPanel sources={msg.sources} />}
          </div>
        ))}
        {loading && <p style={{ color: "#888" }}>Thinking...</p>}
      </div>

      <div style={{ marginTop: "1rem", display: "flex", gap: "0.5rem" }}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question..."
          style={{ flex: 1, padding: "0.5rem" }}
        />
        <button onClick={handleAsk} disabled={loading}>
          Ask
        </button>
      </div>
    </div>
  );
}

export default App;
