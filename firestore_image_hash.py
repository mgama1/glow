import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
import firebase_admin
from firebase_admin import credentials, firestore
import base64
import io
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv
import hashlib

# Page configuration
st.set_page_config(
    page_title="GLOW Reverse Search",
    page_icon="🖼️",
    layout="wide"
)
import os
import streamlit as st

def get_secret(key):
    # Try to get from Streamlit Secrets first
    try:
        return st.secrets[key]
    except KeyError:
        return os.getenv(key)


# Initialize Firebase (add your Firebase config)
@st.cache_resource
def init_firebase():
    """Initialize Firebase connection"""
    try:
        # If already initialized, return existing app
        app = firebase_admin.get_app()
        return firestore.client(app)
    except ValueError:
        # Initialize with service account key
        # Replace this with your Firebase service account key
        load_dotenv()

        cred = credentials.Certificate({
            "type": get_secret("FIREBASE_TYPE"),
            "project_id": get_secret("FIREBASE_PROJECT_ID"),
            "private_key_id": get_secret("FIREBASE_PRIVATE_KEY_ID"),
            "private_key": get_secret("FIREBASE_PRIVATE_KEY"),
            "client_email": get_secret("FIREBASE_CLIENT_EMAIL"),
            "client_id": get_secret("FIREBASE_CLIENT_ID"),
            "auth_uri": get_secret("FIREBASE_AUTH_URI"),
            "token_uri": get_secret("FIREBASE_TOKEN_URI"),
            "auth_provider_x509_cert_url": get_secret("FIREBASE_AUTH_PROVIDER_CERT_URL"),
            "client_x509_cert_url": get_secret("FIREBASE_CLIENT_CERT_URL")
        })


        firebase_admin.initialize_app(cred)
        return firestore.client()

def calculate_md5_hash(image):
    """Calculate MD5 hash of the original image"""
    try:
        # Convert image to bytes
        img_bytes = io.BytesIO()
        # Convert to RGB if necessary for consistent hashing
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        image.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        # Calculate MD5 hash
        md5_hash = hashlib.md5(img_bytes).hexdigest()
        return md5_hash
    except Exception as e:
        st.error(f"Error calculating MD5 hash: {str(e)}")
        return None

def compress_image_to_base64(image, max_size=256, quality=100):
    """Compress image and convert to base64"""
    # Create a copy to avoid modifying original
    img = image.copy()
    
    # Convert to RGB if necessary
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    
    # Resize while maintaining aspect ratio
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Save to bytes with compression
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality, optimize=True)
    buffer.seek(0)
    
    # Convert to base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_base64

def base64_to_image(base64_string):
    """Convert base64 string back to PIL Image"""
    img_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_bytes))

def calculate_hashes(image):
    """Calculate pHash, dHash, and wHash for an image"""
    try:
        phash = str(imagehash.phash(image))
        dhash = str(imagehash.dhash(image))
        whash = str(imagehash.whash(image))
        return phash, dhash, whash
    except Exception as e:
        st.error(f"Error calculating hashes: {str(e)}")
        return None, None, None

def check_existing_image(db, md5_hash):
    """Check if an image with the same MD5 hash already exists"""
    try:
        docs = db.collection('images').where('md5_hash', '==', md5_hash).limit(1).stream()
        for doc in docs:
            return doc.id, doc.to_dict()
        return None, None
    except Exception as e:
        st.error(f"Error checking existing image: {str(e)}")
        return None, None

def save_to_firestore(db, image_base64, price, phash, dhash, whash, original_filename, md5_hash):
    """Save image data to Firestore or update existing record"""
    try:
        # Check if image already exists
        existing_doc_id, existing_data = check_existing_image(db, md5_hash)
        
        if existing_doc_id:
            # Update existing record
            doc_ref = db.collection('images').document(existing_doc_id)
            doc_ref.update({
                'image_base64': image_base64,
                'price': price,
                'phash': phash,
                'dhash': dhash,
                'whash': whash,
                'original_filename': original_filename,
                'timestamp': datetime.now(),
                'md5_hash': md5_hash
            })
            return existing_doc_id, True  # True indicates it was an update
        else:
            # Create new record
            doc_id = str(uuid.uuid4())
            doc_ref = db.collection('images').document(doc_id)
            
            doc_ref.set({
                'image_base64': image_base64,
                'price': price,
                'phash': phash,
                'dhash': dhash,
                'whash': whash,
                'original_filename': original_filename,
                'timestamp': datetime.now(),
                'doc_id': doc_id,
                'md5_hash': md5_hash
            })
            return doc_id, False  # False indicates it was a new record
            
    except Exception as e:
        st.error(f"Error saving to Firestore: {str(e)}")
        return None, False

def load_from_firestore(db):
    """Load all images from Firestore"""
    try:
        docs = db.collection('images').stream()
        data = []
        
        for doc in docs:
            doc_data = doc.to_dict()
            data.append(doc_data)
        
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading from Firestore: {str(e)}")
        return pd.DataFrame()

def process_multiple_images(uploaded_files, price, db):
    """Process multiple images and save to Firestore"""
    results = []
    errors = []
    updates = []
    
    for uploaded_file in uploaded_files:
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Read and open image
            image = Image.open(uploaded_file)
            
            # Calculate MD5 hash first
            md5_hash = calculate_md5_hash(image)
            if not md5_hash:
                errors.append(f"Failed to calculate MD5 hash for {uploaded_file.name}")
                continue
            
            # Calculate hashes on original image
            phash, dhash, whash = calculate_hashes(image)
            
            if not all([phash, dhash, whash]):
                errors.append(f"Failed to calculate hashes for {uploaded_file.name}")
                continue
            
            # Compress image to base64
            image_base64 = compress_image_to_base64(image)
            
            # Save to Firestore (or update existing)
            doc_id, was_update = save_to_firestore(
                db, image_base64, price, phash, dhash, whash, uploaded_file.name, md5_hash
            )
            
            if doc_id:
                result_data = {
                    'file_name': uploaded_file.name,
                    'doc_id': doc_id,
                    'image': image,
                    'phash': phash,
                    'dhash': dhash,
                    'whash': whash,
                    'md5_hash': md5_hash,
                    'was_update': was_update
                }
                
                if was_update:
                    updates.append(result_data)
                else:
                    results.append(result_data)
            else:
                errors.append(f"Failed to save {uploaded_file.name} to Firestore")
                
        except Exception as e:
            errors.append(f"Error processing {uploaded_file.name}: {str(e)}")
    
    return results, updates, errors

def hamming_distance(hash1, hash2):
    """Calculate Hamming distance between two hash strings"""
    if len(hash1) != len(hash2):
        return float('inf')
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def find_similar_images(query_phash, query_dhash, query_whash, df, top_n=3):
    """Find the most similar images based on hash distances"""
    if df.empty:
        return []
    
    similarities = []
    
    for idx, row in df.iterrows():
        # Calculate distances for each hash type
        phash_dist = hamming_distance(query_phash, str(row['phash']))
        dhash_dist = hamming_distance(query_dhash, str(row['dhash']))
        whash_dist = hamming_distance(query_whash, str(row['whash']))
        
        # Calculate average distance
        avg_distance = (phash_dist + dhash_dist + whash_dist) / 3
        
        similarities.append({
            'index': idx,
            'doc_id': row['doc_id'],
            'image_base64': row['image_base64'],
            'price': row['price'],
            'phash': row['phash'],
            'dhash': row['dhash'],
            'whash': row['whash'],
            'timestamp': row['timestamp'],
            'original_filename': row['original_filename'],
            'phash_distance': phash_dist,
            'dhash_distance': dhash_dist,
            'whash_distance': whash_dist,
            'avg_distance': avg_distance
        })
    
    # Sort by average distance (lower is more similar)
    similarities.sort(key=lambda x: x['avg_distance'])
    
    return similarities[:top_n]

def render_sidebar(db):
    """Render the sidebar navigation"""
    with st.sidebar:
        st.title("GLOW Reverse Search")
        st.markdown("---")
        
        # Navigation
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "📤 Upload Images", "🔍 Find Similar Images"],
            key="navigation"
        )
        
        st.markdown("---")
        
        # Database info
        df = load_from_firestore(db)
        if not df.empty:
            st.subheader("📊 Database Info")
            st.metric("Total Images", len(df))
            if 'price' in df.columns:
                avg_price = df['price'].mean()
                st.metric("Avg Price", f"£{avg_price:.2f}")
        
        return page

def render_home_page(db):
    """Render the home page"""
    st.title("GLOW STORE")
    
    st.markdown("""
    Welcome to the **GLOW STORE** Reverse image search! This tool allows you to upload multiple images, calculate their perceptual hashes, and store them in Firestore for easy retrieval and comparison.
    """)
    
    st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <img src="https://scontent.fcai20-1.fna.fbcdn.net/v/t39.30808-6/499123096_1874969766690568_8210053830202425729_n.jpg?_nc_cat=108&ccb=1-7&_nc_sid=6ee11a&_nc_ohc=4ZIhfZ6tSG8Q7kNvwGnHGtH&_nc_oc=Adlx7UpxLzhvaMZ1vTDZj8qCO0t-n2VvM-7uH1FvnsooWeJNig_hYV9vRopSXXuSRm8&_nc_zt=23&_nc_ht=scontent.fcai20-1.fna&_nc_gid=WvmBtWo4pUlhj2vT7qumZA&oh=00_AfTT7zjxtIDWlPNX8R3aReVrTUV0HR2NE09FL3ckyFVp7Q&oe=6870D77E" style="max-width: 50%; height: auto;">
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick stats
    df = load_from_firestore(db)
    if not df.empty:
        st.subheader("📈 Quick Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Images", len(df))
        
        with col2:
            avg_price = df['price'].mean()
            st.metric("Average Price", f"£{avg_price:.2f}")
        
        with col3:
            unique_prices = df['price'].nunique()
            st.metric("Unique Prices", unique_prices)
        
        with col4:
            if 'timestamp' in df.columns:
                latest = df['timestamp'].max()
                st.metric("Latest Entry", str(latest)[:19])
            else:
                st.metric("Latest Entry", "N/A")
        
        # Recent additions
        st.subheader("🕒 Recent Additions")
        recent_df = df.sort_values('timestamp', ascending=False).head(5)
        display_df = recent_df[['original_filename', 'price', 'timestamp']].copy()
        display_df['timestamp'] = display_df['timestamp'].astype(str).str[:19]
        st.dataframe(display_df, use_container_width=True)
    
    else:
        st.info("📝 No images in database yet. Use the **Upload Images** page to get started!")
    
    st.markdown("---")
    st.markdown("Use the sidebar to navigate between different features.")

def render_upload_page(db):
    """Render the upload images page"""
    st.title("📤 Multi-Image Hash Calculator")
    st.markdown("Upload multiple images of the same item with a shared price. Images will be compressed to thumbnails and stored in Firestore.")
    
    # Initialize session state
    if 'uploader_counter' not in st.session_state:
        st.session_state.uploader_counter = 0
    if 'price_value' not in st.session_state:
        st.session_state.price_value = 0
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Images")
        
        # File uploader
        uploader_key = f"file_uploader_{st.session_state.uploader_counter}"
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'],
            help="Images will be compressed to 150x150 thumbnails",
            accept_multiple_files=True,
            key=uploader_key
        )
        
        price = st.number_input(
            "Enter price (shared for all images)",
            min_value=0,
            step=100,
            format="%d",
            help="This price will be applied to all uploaded images",
            value=st.session_state.price_value,
            key=f"price_input_{st.session_state.uploader_counter}"
        )
        
        process_button = st.button("Process All Images", type="primary", use_container_width=True)
        
        # Clear button
        if uploaded_files:
            if st.button("Clear Images", use_container_width=True):
                st.session_state.uploader_counter += 1
                st.session_state.price_value = 0
                st.rerun()
    
    with col2:
        # Show preview
        if uploaded_files:
            st.subheader(f"Image Previews ({len(uploaded_files)} files)")
            
            # Display images in a grid
            if len(uploaded_files) <= 4:
                cols = st.columns(min(len(uploaded_files), 2))
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        uploaded_file.seek(0)
                        image = Image.open(uploaded_file)
                        uploaded_file.seek(0)
                        cols[i % 2].image(image, caption=uploaded_file.name, use_column_width=True)
                    except Exception as e:
                        st.error(f"Error displaying {uploaded_file.name}: {str(e)}")
            else:
                # Show first 4 images
                cols = st.columns(2)
                for i in range(min(4, len(uploaded_files))):
                    try:
                        uploaded_files[i].seek(0)
                        image = Image.open(uploaded_files[i])
                        uploaded_files[i].seek(0)
                        cols[i % 2].image(image, caption=uploaded_files[i].name, use_column_width=True)
                    except Exception as e:
                        st.error(f"Error displaying {uploaded_files[i].name}: {str(e)}")
                
                if len(uploaded_files) > 4:
                    st.info(f"... and {len(uploaded_files) - 4} more images")
    
    # Process images
    if process_button and uploaded_files:
        with st.spinner(f"Processing {len(uploaded_files)} images..."):
            try:
                results, updates, errors = process_multiple_images(uploaded_files, price, db)
                
                if errors:
                    st.error("Some errors occurred:")
                    for error in errors:
                        st.error(f"• {error}")
                
                if updates:
                    st.warning(f"⚠️ {len(updates)} duplicate images were updated with new data:")
                    for update in updates:
                        st.info(f"• {update['file_name']} (MD5: {update['md5_hash'][:8]}...)")
                
                if results:
                    st.success(f"✅ {len(results)} new images processed and saved to Firestore!")
                    
                    # Display results
                    if len(results) > 1:
                        tabs = st.tabs([f"Image {i+1}" for i in range(len(results))])
                        
                        for i, (tab, result) in enumerate(zip(tabs, results)):
                            with tab:
                                st.subheader(f"Results for {result['file_name']}")
                                
                                # Display the image
                                st.image(result['image'], caption=result['file_name'], width=300)
                                
                                # Display hashes
                                hash_col1, hash_col2, hash_col3 = st.columns(3)
                                with hash_col1:
                                    st.metric("pHash", result['phash'])
                                with hash_col2:
                                    st.metric("dHash", result['dhash'])
                                with hash_col3:
                                    st.metric("wHash", result['whash'])
                                
                                st.info(f"Saved to Firestore with ID: {result['doc_id']}")
                                st.code(f"MD5: {result['md5_hash']}", language="text")
                    else:
                        # Single image
                        result = results[0]
                        st.subheader("Calculated Hashes")
                        hash_col1, hash_col2, hash_col3 = st.columns(3)
                        
                        with hash_col1:
                            st.metric("pHash", result['phash'])
                        with hash_col2:
                            st.metric("dHash", result['dhash'])
                        with hash_col3:
                            st.metric("wHash", result['whash'])
                        
                        st.info(f"Saved to Firestore with ID: {result['doc_id']}")
                        st.code(f"MD5: {result['md5_hash']}", language="text")
                    
                    # Clear after success
                    st.session_state.uploader_counter += 1
                    st.session_state.price_value = 0
                    st.rerun()
                
            except Exception as e:
                st.error(f"Error processing images: {str(e)}")
    
    elif process_button and not uploaded_files:
        st.warning("Please upload at least one image first!")
    
    # Display current database stats
    st.subheader("📊 Current Database")
    df = load_from_firestore(db)
    
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            avg_price = df['price'].mean()
            st.metric("Average Price", f"£{avg_price:.2f}")
        with col3:
            unique_prices = df['price'].nunique()
            st.metric("Unique Prices", unique_prices)
        with col4:
            latest_entry = str(df['timestamp'].max())[:19]
            st.metric("Latest Entry", latest_entry)
    else:
        st.info("No records in database yet. Upload images to get started!")

def render_search_page(db):
    """Render the search similar images page"""
    st.title("🔍 Find Similar Images")
    st.markdown("Upload an image to find the most similar images in your Firestore database.")
    
    # Check if database has images
    df = load_from_firestore(db)
    if df.empty:
        st.warning("⚠️ No images in database yet. Please upload some images first.")
        return
    
    # Initialize session state
    if 'search_uploader_counter' not in st.session_state:
        st.session_state.search_uploader_counter = 0
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Query Image")
        
        # File uploader
        uploader_key = f"search_uploader_{st.session_state.search_uploader_counter}"
        uploaded_file = st.file_uploader(
            "Choose a query image",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'],
            help="Upload an image to find similar images",
            key=uploader_key
        )
        
        # Number of results
        num_results = st.slider(
            "Number of similar images to show",
            min_value=1,
            max_value=min(10, len(df)),
            value=3
        )
        
        search_button = st.button("🔍 Find Similar Images", type="primary", use_container_width=True)
        
        if uploaded_file:
            if st.button("Clear Query Image", use_container_width=True):
                st.session_state.search_uploader_counter += 1
                st.rerun()
    
    with col2:
        # Show query image preview
        if uploaded_file:
            st.subheader("Query Image Preview")
            try:
                uploaded_file.seek(0)
                query_image = Image.open(uploaded_file)
                uploaded_file.seek(0)
                st.image(query_image, caption="Query Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying query image: {str(e)}")
        else:
            st.info("Upload a query image to see preview")
    
    # Process search
    if search_button and uploaded_file:
        with st.spinner("Searching for similar images..."):
            try:
                # Process query image
                uploaded_file.seek(0)
                query_image = Image.open(uploaded_file)
                uploaded_file.seek(0)
                
                if query_image.mode in ('RGBA', 'P'):
                    query_image = query_image.convert('RGB')
                
                # Calculate hashes
                query_phash, query_dhash, query_whash = calculate_hashes(query_image)
                
                if not all([query_phash, query_dhash, query_whash]):
                    st.error("Failed to calculate hashes for query image")
                    return
                
                # Display query hashes
                st.subheader("Query Image Hashes")
                hash_col1, hash_col2, hash_col3 = st.columns(3)
                with hash_col1:
                    st.metric("pHash", query_phash)
                with hash_col2:
                    st.metric("dHash", query_dhash)
                with hash_col3:
                    st.metric("wHash", query_whash)
                
                # Find similar images
                similarities = find_similar_images(query_phash, query_dhash, query_whash, df, num_results)
                
                if similarities:
                    st.subheader(f"🎯 Top {len(similarities)} Similar Images")
                    
                    # Display results
                    for i, sim in enumerate(similarities):
                        with st.container():
                            st.markdown(f"### Result #{i+1}")
                            
                            # Create columns
                            img_col, details_col = st.columns([1, 1])
                            
                            with img_col:
                                # Display image from base64
                                try:
                                    img = base64_to_image(sim['image_base64'])
                                    st.image(img, caption=f"Distance: {sim['avg_distance']:.2f}", use_column_width=True)
                                except Exception as e:
                                    st.error(f"Error displaying image: {str(e)}")
                            
                            with details_col:
                                st.markdown(f"**Filename:** {sim['original_filename']}")
                                st.markdown(f"**Price:** £{sim['price']:.2f}")
                                st.markdown(f"**Timestamp:** {str(sim['timestamp'])[:19]}")
                                st.markdown(f"**Distance:** {sim['avg_distance']:.2f}")
                                
                                # Hash distances
                                st.markdown("**Hash Distances:**")
                                distance_col1, distance_col2, distance_col3 = st.columns(3)
                                with distance_col1:
                                    st.metric("pHash", f"{sim['phash_distance']}")
                                with distance_col2:
                                    st.metric("dHash", f"{sim['dhash_distance']}")
                                with distance_col3:
                                    st.metric("wHash", f"{sim['whash_distance']}")
                                
                                # Similarity percentage
                                max_distance = 64
                                similarity_pct = max(0, (max_distance - sim['avg_distance']) / max_distance * 100)
                                st.progress(similarity_pct / 100)
                                st.markdown(f"**Similarity:** {similarity_pct:.1f}%")
                            
                            st.markdown("---")
                    
                    # Summary
                    st.subheader("📈 Search Summary")
                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                    
                    with summary_col1:
                        st.metric("Images Searched", len(df))
                    with summary_col2:
                        best_match = similarities[0]['avg_distance']
                        st.metric("Best Match", f"{best_match:.2f}")
                    with summary_col3:
                        avg_distance = sum(sim['avg_distance'] for sim in similarities) / len(similarities)
                        st.metric("Average Distance", f"{avg_distance:.2f}")
                    with summary_col4:
                        avg_price = sum(sim['price'] for sim in similarities) / len(similarities)
                        st.metric("Avg Price", f"£{avg_price:.2f}")
                    
                else:
                    st.info("No similar images found.")
                    
            except Exception as e:
                st.error(f"Error during search: {str(e)}")
    
    elif search_button and not uploaded_file:
        st.warning("Please upload a query image first!")

# Main application
def main():
    # Initialize Firebase
    try:
        db = init_firebase()
        st.success("✅ Connected to Firestore", icon="🔥")
    except Exception as e:
        st.error(f"❌ Failed to connect to Firestore: {str(e)}")
        st.stop()
    
    # Render sidebar navigation
    page = render_sidebar(db)
    
    # Render the appropriate page
    if page == "🏠 Home":
        render_home_page(db)
    elif page == "📤 Upload Images":
        render_upload_page(db)
    elif page == "🔍 Find Similar Images":
        render_search_page(db)

if __name__ == "__main__":
    main()