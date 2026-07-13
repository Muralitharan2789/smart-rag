import { useState, useEffect, useRef } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

const API_BASE = "http://localhost:8000";

function StatusDot() {
  const [status, setStatus] = useState("checking");

  useEffect(() => {
    const check = () => {
      axios
        .get(`${API_BASE}/health`)
        .then((res) => setStatus(res.data.status === "ok" ? "online" : "degraded"))
        .catch(() => setStatus("offline"));
    };
    check();
    const interval = setInterval(check, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="status">
      <span className={`status-dot status-dot--${status}`} />
      <span className="status-label">
        {status === "online" ? "Backend connected" : status === "checking" ? "Connecting..." : "Backend unreachable"}
      </span>
    </div>
  );
}

function Exhibit({ source, index, isOpen, onToggle }) {
  return (
    <div className="exhibit">
      <button className={`exhibit-tab ${isOpen ? "exhibit-tab--open" : ""}`} onClick={onToggle}>
        Exhibit {source.source_number}
      </button>
      {isOpen && (
        <div className="exhibit-card">
          <div className="exhibit-card-header">
            <span className="exhibit-seal">✓ Verified in source</span>
            <span className="exhibit-meta">
              {source.document_name} · {source.chunk_type}
            </span>
          </div>
          <div className="exhibit-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{source.chunk_text_preview}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}

function SourcesRow({ sources }) {
  const [openIndex, setOpenIndex] = useState(null);
  if (!sources || sources.length === 0) return null;

  return (
    <div className="sources-row">
      <div className="exhibit-list">
        {sources.map((src, i) => (
          <Exhibit
            key={i}
            source={src}
            index={i}
            isOpen={openIndex === i}
            onToggle={() => setOpenIndex(openIndex === i ? null : i)}
          />
        ))}
      </div>
    </div>
  );
}

function FileUpload() {
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState(null);
  const inputRef = useRef(null);

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
      setStatus({
        type: "success",
        text: `${file.name} indexed — ${res.data.chunks_stored} chunks (${res.data.table_chunks} tables, ${res.data.text_chunks} text)`,
      });
    } catch (err) {
      setStatus({ type: "error", text: `Upload failed: ${err.message}` });
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  return (
    <div className="upload-zone">
      <div className="upload-zone-left">
        <span className="upload-icon">⬆</span>
        <div>
          <div className="upload-title">Add a document</div>
          <div className="upload-subtitle">PDF or DOCX — parsed, chunked, and indexed automatically</div>
        </div>
      </div>
      <button className="upload-button" onClick={() => inputRef.current.click()} disabled={uploading}>
        {uploading ? "Indexing..." : "Choose file"}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx"
        onChange={handleFileChange}
        style={{ display: "none" }}
      />
      {status && <div className={`upload-status upload-status--${status.type}`}>{status.text}</div>}
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleAsk = async () => {
    if (!question.trim() || loading) return;

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
        {
          role: "assistant",
          text: res.data.answer,
          sources: res.data.sources,
          timings: res.data.timings,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `Something went wrong reaching the backend: ${err.message}`, sources: [], error: true },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header-left">
          <div className="app-mark">SR</div>
          <div>
            <div className="app-title">Smart RAG</div>
            <div className="app-eyebrow">Table-aware, evidence-grounded Q&amp;A</div>
          </div>
        </div>
        <StatusDot />
      </header>

      <main className="app-main">
        <FileUpload />

        <div className="chat-panel">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-state-title">No questions asked yet</div>
              <div className="empty-state-subtitle">
                Upload a document above, then ask a question about its contents. Every
                answer is backed by numbered exhibits linking back to the exact source.
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`message message--${msg.role}`}>
              <div className="message-label">{msg.role === "user" ? "You" : "Smart RAG"}</div>
              <div className={`message-bubble ${msg.error ? "message-bubble--error" : ""}`}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
              </div>
              {msg.role === "assistant" && !msg.error && <SourcesRow sources={msg.sources} />}
            </div>
          ))}

          {loading && (
            <div className="message message--assistant">
              <div className="message-label">Smart RAG</div>
              <div className="message-bubble message-bubble--loading">
                <span className="loading-dot" />
                <span className="loading-dot" />
                <span className="loading-dot" />
              </div>
            </div>
          )}
          <div ref={scrollRef} />
        </div>

        <div className="composer">
          <textarea
            className="composer-input"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your uploaded documents..."
            rows={1}
          />
          <button className="composer-button" onClick={handleAsk} disabled={loading || !question.trim()}>
            Ask
          </button>
        </div>
      </main>
    </div>
  );
}

export default App;
