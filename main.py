import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pdfplumber
import PyPDF2
import re, os, threading
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class IntelligentPDFQA:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent PDF Q&A System")
        self.root.geometry("1000x800")

        # Model + FAISS
        self.model = SentenceTransformer("allenai/scibert_scivocab_uncased")
        self.index = None
        self.sentences = []   # (sentence, page_num)
        self.embeddings = None

        # Selected file
        self.selected_pdf = None

        # Build GUI
        self.create_gui()

    # ---------------- GUI ----------------
    def create_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # PDF selection frame
        pdf_frame = ttk.LabelFrame(main_frame, text="Select PDF Document", padding="5")
        pdf_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.file_path_var = tk.StringVar()
        ttk.Label(pdf_frame, text="Selected PDF:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(pdf_frame, textvariable=self.file_path_var, foreground="blue").grid(row=1, column=0, sticky=tk.W)

        ttk.Button(pdf_frame, text="Browse PDF", command=self.browse_pdf).grid(row=0, column=1, padx=(10, 0))
        ttk.Button(pdf_frame, text="Process PDF", command=self.process_pdf).grid(row=1, column=1, padx=(10, 0))

        # Progress bar
        self.progress = ttk.Progressbar(pdf_frame, mode='indeterminate')
        self.progress.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Document info frame
        info_frame = ttk.LabelFrame(main_frame, text="Document Information", padding="5")
        info_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.info_text = tk.Text(info_frame, height=4, wrap=tk.WORD, state=tk.DISABLED)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Question frame
        question_frame = ttk.LabelFrame(main_frame, text="Ask Intelligent Question", padding="5")
        question_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(question_frame, text="Your question:").grid(row=0, column=0, sticky=tk.W)
        self.question_entry = ttk.Entry(question_frame, width=60)
        self.question_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.question_entry.bind('<Return>', lambda e: self.search_answer())

        ttk.Button(question_frame, text="Search Answer", command=self.search_answer).grid(row=1, column=1)

        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Intelligent Answer", padding="5")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        self.answer_text = scrolledtext.ScrolledText(results_frame, height=20, wrap=tk.WORD)
        self.answer_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Status bar
        self.status_var = tk.StringVar(value="Ready - Please select and process a PDF document")
        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN).grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # Grid config
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)

    # ---------------- PDF Handling ----------------
    def browse_pdf(self):
        filename = filedialog.askopenfilename(
            title="Select PDF Document",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(os.path.basename(filename))
            self.selected_pdf = filename
            self.status_var.set("PDF selected - Click 'Process PDF' to analyze")

    def extract_pdf_sentences(self, filepath):
        """Extract sentences with page numbers."""
        content = []
        try:
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if not text:
                        continue
                    text = re.sub(r"\s+", " ", text.strip())
                    # split by sentence
                    for sent in re.split(r'(?<=[.؟?!])\s+', text):
                        sent = sent.strip()
                        if len(sent) > 20:  # skip very short lines
                            content.append((sent, page_num))
        except Exception as e:
            raise Exception(f"PDF extraction failed: {e}")
        return content

    def process_pdf(self):
        if not self.selected_pdf:
            messagebox.showwarning("Warning", "Please select a PDF file first")
            return

        self.progress.start(10)
        self.status_var.set("Processing PDF... This may take a while")
        self.root.update()

        def worker():
            try:
                self.sentences = self.extract_pdf_sentences(self.selected_pdf)
                if not self.sentences:
                    self.root.after(0, lambda: messagebox.showwarning("Warning", "No text found in PDF"))
                    return

                texts = [s for s, _ in self.sentences]
                self.embeddings = self.model.encode(texts, convert_to_numpy=True)

                dim = self.embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dim)
                self.index.add(self.embeddings)

                self.root.after(0, self.update_ui_after_processing)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Error processing PDF: {str(e)}"))
            finally:
                self.root.after(0, lambda: self.progress.stop())

        threading.Thread(target=worker, daemon=True).start()

    def update_ui_after_processing(self):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, f"📄 Pages: {len(set(p for _, p in self.sentences))} | 📝 Sentences: {len(self.sentences)}")
        self.info_text.config(state=tk.DISABLED)
        self.status_var.set("✅ PDF processed successfully!")

    # ---------------- Search ----------------
    def search_answer(self):
        if not self.index or not self.sentences:
            messagebox.showwarning("Warning", "Please process a PDF first")
            return

        query = self.question_entry.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a question")
            return

        self.status_var.set("🔍 Searching...")
        self.root.update()

        q_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, k=5)

        self.answer_text.delete(1.0, tk.END)
        self.answer_text.insert(tk.END, f"❓ Question: {query}\n\n")
        self.answer_text.insert(tk.END, f"🎯 Top {len(I[0])} answers:\n\n")

        for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
            sentence, page = self.sentences[idx]
            confidence = 1 - (dist / (dist + 1e-5))  # pseudo confidence
            self.answer_text.insert(tk.END, f"{rank}. [Page {page}] (Confidence: {confidence:.2f})\n")
            self.answer_text.insert(tk.END, f"{sentence}\n\n")

        self.status_var.set("✅ Search completed")

def main():
    root = tk.Tk()
    app = IntelligentPDFQA(root)
    root.mainloop()

if __name__ == "__main__":
    main()
