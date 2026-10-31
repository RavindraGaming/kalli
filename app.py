import streamlit as st
import sqlite3
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- CONFIG & DATABASE SETUP -----------------
st.set_page_config(page_title="Campus AI Lost & Found", page_icon="🔎", layout="wide")

DB_FILE = "lost_and_found.db"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            status TEXT NOT NULL,        -- 'Lost' or 'Found'
            category TEXT NOT NULL,
            location TEXT NOT NULL,
            description TEXT,
            contact TEXT NOT NULL,
            image_path TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # Populate with pre-seeded demo records if empty
    cursor.execute("SELECT COUNT(*) FROM items")
    if cursor.fetchone()[0] == 0:
        demo_items = [
            (
                "Blue Water Bottle",
                "Lost",
                "Personal Item",
                "Library",
                "1L Milton stainless steel blue insulated flask with minor dent at base.",
                "student1@college.edu",
                None,
                "2026-10-30 09:15:00"
            ),
            (
                "Black Headphones",
                "Found",
                "Electronics",
                "Canteen",
                "Boat over-ear wireless noise cancelling headset with cushion pads.",
                "canteen_manager@college.edu",
                None,
                "2026-10-31 10:00:00"
            ),
            (
                "Student ID Card",
                "Found",
                "Documents",
                "Block A",
                "Annamacharya college student card for Mechanical Dept with photo lanyard.",
                "security_desk@college.edu",
                None,
                "2026-10-31 10:45:00"
            ),
            (
                "Blue Insulated Flask",
                "Found",
                "Personal Item",
                "Library 2nd Floor",
                "Metallic blue water container left near reading table desk 14.",
                "librarian@college.edu",
                None,
                "2026-10-31 11:30:00"
            ),
            (
                "Wireless Bluetooth Headset",
                "Lost",
                "Electronics",
                "Cafeteria Food Court",
                "Black wireless Boat headset with black charging wire.",
                "audiophile@college.edu",
                None,
                "2026-10-31 11:45:00"
            )
        ]
        cursor.executemany("""
            INSERT INTO items (title, status, category, location, description, contact, image_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, demo_items)
        conn.commit()
    conn.close()


init_db()

# ----------------- AI MATCHING ENGINE -----------------
def compute_ai_similarity(lost_item, found_item):
    """
    Computes an AI semantic similarity score (0 to 100%) using TF-IDF n-grams
    over concatenated textual signatures (Title + Category + Location + Description).
    """
    # Hard filter: if categories differ completely, return 0
    if lost_item["category"] != found_item["category"]:
        return 0.0

    lost_text = f"{lost_item['title']} {lost_item['location']} {lost_item['description'] or ''}"
    found_text = f"{found_item['title']} {found_item['location']} {found_item['description'] or ''}"

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([lost_text, found_text])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(score) * 100, 1)
    except Exception:
        return 0.0


# ----------------- SIDEBAR & NAVIGATION -----------------
st.sidebar.title("🔎 Smart Lost & Found")
menu = st.sidebar.radio("Navigate", ["Browse Listings", "Report New Item", "🤖 AI Smart Matcher"])

# 1. BROWSE LISTINGS
if menu == "Browse Listings":
    st.title("📦 Campus Item Feed")
    
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox("Status Filter", ["All", "Lost", "Found"])
    with col2:
        cat_filter = st.selectbox("Category Filter", ["All", "Personal Item", "Electronics", "Documents", "Clothing", "Other"])

    conn = get_db()
    query = "SELECT * FROM items WHERE 1=1"
    params = []

    if status_filter != "All":
        query += " AND status = ?"
        params.append(status_filter)
    if cat_filter != "All":
        query += " AND category = ?"
        params.append(cat_filter)

    query += " ORDER BY id DESC"
    records = conn.execute(query, params).fetchall()
    conn.close()

    if not records:
        st.info("No matching records found in the database.")
    else:
        for r in records:
            with st.container(border=True):
                c_img, c_body = st.columns([1, 3])
                with c_img:
                    if r["image_path"] and os.path.exists(r["image_path"]):
                        st.image(r["image_path"], use_container_width=True)
                    else:
                        st.markdown("🖼️ *No Image Uploaded*")
                with c_body:
                    tag_color = "🔴" if r["status"] == "Lost" else "🟢"
                    st.markdown(f"### {tag_color} {r['title']} `[{r['status'].upper()}]`")
                    st.write(f"**Category:** {r['category']} | **Location:** {r['location']}")
                    st.write(f"**Details:** {r['description'] or 'No additional description provided.'}")
                    st.caption(f"Contact: {r['contact']} | Reported: {r['created_at']}")

# 2. REPORT NEW ITEM
elif menu == "Report New Item":
    st.title("📝 Report an Item (Lost or Found)")

    with st.form("item_report_form", clear_on_submit=True):
        col_a, col_b = st.columns(2)
        with col_a:
            title = st.text_input("Item Name*", placeholder="e.g., Student ID Card / Titan Watch")
            status = st.selectbox("Status*", ["Lost", "Found"])
            category = st.selectbox("Category*", ["Personal Item", "Electronics", "Documents", "Clothing", "Other"])
        with col_b:
            location = st.text_input("Location*", placeholder="e.g., Block A, 1st Floor Corridor")
            contact = st.text_input("Contact Info*", placeholder="e.g., phone or college email")
            photo = st.file_uploader("Upload Image (Optional)", type=["png", "jpg", "jpeg", "webp"])

        desc = st.text_area("Detailed Description", placeholder="Mention colors, serial numbers, labels, or stickers...")
        submit_btn = st.form_submit_button("Submit Item")

        if submit_btn:
            if not title.strip() or not location.strip() or not contact.strip():
                st.error("Please fill in all mandatory fields: Item Name, Location, and Contact Info.")
            else:
                saved_path = None
                if photo:
                    file_ext = photo.name.split(".")[-1]
                    filename = f"{int(datetime.now().timestamp())}_{title.replace(' ', '_')}.{file_ext}"
                    saved_path = os.path.join(UPLOAD_DIR, filename)
                    with open(saved_path, "wb") as f:
                        f.write(photo.getbuffer())

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn = get_db()
                conn.execute("""
                    INSERT INTO items (title, status, category, location, description, contact, image_path, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (title, status, category, location, desc, contact, saved_path, timestamp))
                conn.commit()
                conn.close()

                st.success(f"Report for '{title}' submitted successfully!")
                st.info("Check '🤖 AI Smart Matcher' to test real-time matches.")

# 3. AI SMART MATCHER
elif menu == "🤖 AI Smart Matcher":
    st.title("🤖 AI Smart Match Finder")
    st.markdown("Uses **TF-IDF + Cosine Similarity Vectorization** to cross-match keywords and descriptions across lost and found entries.")

    conn = get_db()
    lost_items = conn.execute("SELECT * FROM items WHERE status = 'Lost'").fetchall()
    found_items = conn.execute("SELECT * FROM items WHERE status = 'Found'").fetchall()
    conn.close()

    if not lost_items or not found_items:
        st.warning("You need at least one 'Lost' item and one 'Found' item registered to run the matcher.")
    else:
        lost_dict = {f"#{i['id']}: {i['title']} (Loc: {i['location']})": i for i in lost_items}
        selected_key = st.selectbox("Choose a reported LOST item to match against found inventory:", list(lost_dict.keys()))
        selected_lost = lost_dict[selected_key]

        st.markdown(f"**Target Item:** {selected_lost['title']} | **Category:** `{selected_lost['category']}` | **Location:** `{selected_lost['location']}`")

        # Run similarity across all found items
        scored_matches = []
        for found in found_items:
            score = compute_ai_similarity(selected_lost, found)
            if score > 0.0:
                scored_matches.append((score, found))

        scored_matches.sort(key=lambda x: x[0], reverse=True)

        st.divider()
        st.subheader("Match Results")

        if not scored_matches:
            st.info("No semantic matches found above the confidence threshold.")
        else:
            for score, found in scored_matches:
                score_badge = "🟢 Strong Match" if score >= 50 else ("🟡 Moderate Match" if score >= 20 else "⚪ Low Probability")
                with st.container(border=True):
                    col_score, col_info, col_img = st.columns([1, 3, 1])
                    with col_score:
                        st.metric(label="AI Confidence", value=f"{score}%")
                        st.caption(score_badge)
                    with col_info:
                        st.markdown(f"#### {found['title']}")
                        st.write(f"**Found at:** {found['location']} | **Date:** {found['created_at']}")
                        st.write(f"**Description:** {found['description']}")
                        st.markdown(f"**Claim Contact:** `{found['contact']}`")
                    with col_img:
                        if found["image_path"] and os.path.exists(found["image_path"]):
                            st.image(found["image_path"], caption="Found Item Photo", use_container_width=True)
                        else:
                            st.caption("No photo provided")