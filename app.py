from flask import Flask, render_template, request, redirect, url_for, session
import os, fitz, docx, cv2, numpy as np, uuid
from PIL import Image
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor

# Load Haar cascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
CACHE_FOLDER = 'cache'
os.makedirs(CACHE_FOLDER, exist_ok=True)

# ---------- Helper Functions ----------

def preprocess(text):
    return " ".join(text.lower().split()) if text.strip() else f"empty_{uuid.uuid4()}"

def cache_key(file_path):
    stat = os.stat(file_path)
    return f"{os.path.basename(file_path)}_{stat.st_size}"

def is_valid_image(file_path):
    """Validate if image contains text-based assignments (scanned/handwritten), not photos of living/non-living things."""
    try:
        img = cv2.imread(file_path)
        if img is None:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect human faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            return False  # Human face detected

        # Detect cat faces
        cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(cats) > 0:
            return False  # Cat face detected

        # Check for text using OCR
        text = pytesseract.image_to_string(Image.open(file_path))
        if not text.strip():
            return False  # No text found, likely a photo

        return True  # Valid if text present and no faces detected
    except Exception as e:
        print(f"Error validating image {file_path}: {e}")
        return False

def extract_text(file_path):
    """Extract text efficiently with caching and minimal OCR."""
    key = cache_key(file_path)
    cache_path = os.path.join(CACHE_FOLDER, key + ".txt")

    # Use cached version if available
    if os.path.exists(cache_path):
        return open(cache_path, encoding="utf-8").read()

    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            doc = fitz.open(file_path)
            for p in doc:
                t = p.get_text()
                if t.strip():
                    text += t
            # OCR only if no text found
            if not text.strip():
                for p in doc:
                    pix = p.get_pixmap()
                    img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), 1)
                    text += pytesseract.image_to_string(img)

        elif ext == ".docx":
            text = "\n".join(p.text for p in docx.Document(file_path).paragraphs)

        elif ext == ".txt":
            text = open(file_path, encoding="utf-8").read()

        elif ext in [".png", ".jpg", ".jpeg"]:
            text = pytesseract.image_to_string(Image.open(file_path))

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    text = preprocess(text)
    open(cache_path, "w", encoding="utf-8").write(text)
    return text

# ---------- Routes ----------

@app.route('/')
def landing():
    return render_template('captcha.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        files = request.files.getlist('students')
        if len(files) < 2:
            return render_template('upload.html', error="Upload at least 2 files.")

        names, paths, invalid_images = [], [], []
        for f in files[:64]:
            path = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(path)
            ext = os.path.splitext(f.filename)[1].lower()
            if ext in [".png", ".jpg", ".jpeg"]:
                if not is_valid_image(path):
                    invalid_images.append(f.filename)
                    os.remove(path)  # Delete invalid image
                    continue
            names.append(f.filename)
            paths.append(path)

        if invalid_images:
            error_msg = f"The following files are invalid: {', '.join(invalid_images)}. This is not an assignment."
            return render_template('upload.html', error=error_msg)

        # --- Parallel text extraction for speed ---
        with ThreadPoolExecutor() as executor:
            texts = list(executor.map(extract_text, paths))

        # --- TF-IDF with limited vocab for faster computation ---
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        vecs = vectorizer.fit_transform(texts)

        # --- Compute cosine similarity matrix ---
        sims = cosine_similarity(vecs)
        threshold = 0.7
        graph, sim_dict = defaultdict(set), {}

        n = len(names)
        for i in range(n):
            for j in range(i + 1, n):
                s = sims[i][j]
                sim_dict[frozenset([names[i], names[j]])] = s
                if s >= threshold:
                    graph[names[i]].add(names[j])
                    graph[names[j]].add(names[i])

        visited, groups = set(), []

        def dfs(node, g):
            visited.add(node)
            g.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, g)

        # Use different variable names to avoid overwriting n
        for filename in names:
            if filename not in visited and filename in graph:
                g = set()
                dfs(filename, g)
                groups.append(sorted(g))

        # --- Build results including non-similar files ---
        results = []
        group_index = 1

        # Real plagiarism groups (≥ threshold)
        for g in groups:
            sims_list = [sim_dict[frozenset([a, b])] for a, b in combinations(g, 2)]
            avg = f"{(sum(sims_list) / len(sims_list) * 100):.2f}" if sims_list else "N/A"
            results.append({
                'group': f"Group {group_index}",
                'files': ", ".join(g),
                'similarity': avg,
                'note': "Copied each other"
            })
            group_index += 1

        # Non-grouped files (below threshold)
        grouped_files = {f for g in groups for f in g}
        ungrouped = [f for f in names if f not in grouped_files]

        for file in ungrouped:
            idx = names.index(file)
            avg_sim = np.mean([sims[idx][j] for j in range(len(names)) if j != idx]) * 100
            results.append({
                'group': f"Group {group_index}",
                'files': file,
                'similarity': f"{avg_sim:.2f}",
                'note': "No strong similarity detected"
            })
            group_index += 1

        session['results'] = results
        return redirect(url_for('results'))

    return render_template('upload.html')

@app.route('/results')
def results():
    return render_template('results.html', results=session.get('results'))

# ---------- Run ----------
if __name__ == '__main__':
    app.run(debug=True, threaded=True)
