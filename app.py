import gradio as gr
import uuid
from pathlib import Path
from models import embeddings, vectordb, video
import shutil

BASE = Path("data/uploads")
BASE.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(fileobj, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)

    # NamedString: copy it to the destination path
    if hasattr(fileobj, "name") and Path(fileobj.name).exists():
        src_path = Path(fileobj.name)
        dest_path = dest_dir / src_path.name
        shutil.copy(src_path, dest_path)
        return str(dest_path)

    # String: directly save it as a file
    elif isinstance(fileobj, str):
        # generate a unique name
        dest_path = dest_dir / f"text_input_{uuid.uuid4().hex[:8]}.txt"
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(fileobj.strip())
        return str(dest_path)

    else:
        raise ValueError(f"Unsupported file object type: {type(fileobj)}")


def index_session_files(session_id, images, videos, text_files, direct_text,
                        frame_every=2, max_video_frames=40):
    """
    - Save uploaded files into data/uploads/{session_id}/
    - For each file, compute embedding and add to Chroma
    - direct_text (string) will be indexed into text collection
    """
    session_dir = BASE / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    indexed = {"images": [], "videos": [], "texts": []}

    # text files
    if text_files:
        txt_dir = session_dir / "texts"
        txt_dir.mkdir(exist_ok=True)
        for f in text_files:
            path = save_uploaded_file(f, txt_dir)
            content = Path(path).read_text(encoding="utf-8")
            vec = embeddings.text_to_vector(content)
            item_id = f"txt-{uuid.uuid4().hex[:8]}"
            meta = {"type": "text", "filepath": path, "session_id": session_id}
            vectordb.add_item("texts", item_id, vec, meta,
                              document=content[:200])
            indexed["texts"].append({"id": item_id, "path": path})

    # direct text (a short note, prompt or long text from textbox)
    if direct_text and direct_text.strip():
        txt_dir = session_dir / "texts"
        txt_dir.mkdir(exist_ok=True)
        path = save_uploaded_file(direct_text, txt_dir)
        content = direct_text.strip()
        vec = embeddings.text_to_vector(content)
        item_id = f"txt-{uuid.uuid4().hex[:8]}"
        meta = {"type": "text", "filepath": path, "session_id": session_id,
                "source": "direct_input"}
        vectordb.add_item("texts", item_id, vec, meta, document=content[:200])
        indexed["texts"].append({"id": item_id, "path": path})

    # images
    if images:
        img_dir = session_dir / "images"
        img_dir.mkdir(exist_ok=True)
        for f in images:
            path = save_uploaded_file(f, img_dir)
            vec = embeddings.image_to_vector(path)
            caption = embeddings.generate_caption(path)
            item_id = f"img-{uuid.uuid4().hex[:8]}"
            meta = {
                "type": "image",
                "filepath": str(path),
                "session_id": session_id,
                "caption": caption
            }
            vectordb.add_item("photos", item_id, vec, meta, document=caption)
            indexed["images"].append({"id": item_id,
                                      "path": path,
                                      "caption": caption})

    # videos: save, extract frames, embed frames
    if videos:
        video_dir = session_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        for f in videos:
            path = save_uploaded_file(f, video_dir)
            frames_dir = session_dir / "frames" / (Path(path).stem)
            frames = video.extract_frames(path, str(frames_dir),
                                          every_n_seconds=frame_every,
                                          max_frames=max_video_frames)
            for fr in frames:
                vec = embeddings.image_to_vector(fr)
                caption = embeddings.generate_caption(fr)
                item_id = f"vidframe-{uuid.uuid4().hex[:8]}"
                meta = {
                    "type": "video_frame",
                    "filepath": str(fr),
                    "video_source": str(path),
                    "session_id": session_id,
                    "caption": caption
                }
                vectordb.add_item("videos", item_id, vec,
                                  meta, document=caption)
                indexed["videos"].append({"id": item_id,
                                          "frame": fr,
                                          "source_video": path,
                                          "caption": caption})

    summary = {
        "session_id": session_id,
        "indexed_counts": {k: len(v) for k, v in indexed.items()},
        "details": indexed
    }
    return summary


def search_collection_ui(query_text, collection_name="texts", n_results=5):
    try:
        if not query_text.strip():
            return {"error": "Please enter a query text."}
        query_vec = embeddings.text_to_vector(query_text)
        res = vectordb.search_collection(collection_name, query_vec, n_results)
        # Pretty format results
        results = []
        for i in range(len(res["ids"][0])):
            results.append({
                "id": res["ids"][0][i],
                "metadata": res["metadatas"][0][i],
                "document": res["documents"][0][i],
                "distance": res["distances"][0][i]
                if "distances" in res
                else None,
            })
        return {"query": query_text, "results": results}
    except Exception as e:
        return {"error": str(e)}


with gr.Blocks() as demo:
    gr.Markdown("# StoryTrail AI — Upload & Index (MVP)")

    with gr.Row():
        with gr.Column(scale=1):
            images_in = gr.File(file_count="multiple",
                                label="Upload photos (jpg/png)",
                                file_types=["image"])
            videos_in = gr.File(file_count="multiple",
                                label="Upload videos (mp4/mov)",
                                file_types=["video"])
            texts_in = gr.File(file_count="multiple",
                               label="Upload text files (.txt/.md)",
                               file_types=[".txt", ".md"])
            direct_text = gr.Textbox(lines=6,
                                     placeholder="Or paste/write text here...",
                                     label="Direct text input")
            idx_btn = gr.Button("Index Uploads")
        with gr.Column(scale=1):
            result = gr.JSON(label="Indexing result")

    def on_index(images, videos, texts, dtext):
        sid = uuid.uuid4().hex[:8]
        summary = index_session_files(sid, images, videos, texts, dtext)
        return summary

    idx_btn.click(fn=on_index,
                  inputs=[images_in, videos_in, texts_in, direct_text],
                  outputs=[result])

    # New search UI
    with gr.Row():
        query_input = gr.Textbox(label="Search query",
                                 placeholder="Enter text to search...")
        collection_choice = gr.Dropdown(choices=["texts", "photos", "videos"],
                                        value="texts",
                                        label="Search Collection")
        n_results = gr.Slider(minimum=1, maximum=20, value=5, step=1,
                              label="Number of Results")
        search_btn = gr.Button("Search")
        search_result = gr.JSON(label="Search Results")

    search_btn.click(fn=search_collection_ui,
                     inputs=[query_input, collection_choice, n_results],
                     outputs=[search_result])

demo.launch()
