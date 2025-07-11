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

def calculate_cryptographic_hash(image_bytes):
    """Calculate SHA-256 hash of image bytes for duplicate detection"""
    return hashlib.sha256(image_bytes).hexdigest()

def image_to_bytes(image):
    """Convert PIL Image to bytes for hashing"""
    buffer = io.BytesIO()
    # Save in a consistent format for hashing
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    image.save(buffer, format='PNG')
    return buffer.getvalue()

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

def check_duplicate_image(db, crypto_hash):
    """Check if an image with the same cryptographic hash already exists"""
    try:
        # Query for documents with the same crypto_hash
        docs = db.collection('images').where('crypto_hash', '==', crypto_hash).limit(1).stream()
        
        for doc in docs:
            return doc.to_dict(), doc.id
        
        return None, None
    except Exception as e:
        st.error(f"Error checking for duplicates: {str(e)}")
        return None, None

def update_price_in_firestore(db, doc_id, new_price, original_filename):
    """Update the price of an existing image in Firestore"""
    try:
        doc_ref = db.collection('images').document(doc_id)
        doc_ref.update({
            'price': new_price,
            'last_updated': datetime.now(),
            'updated_from_file': original_filename
        })
        return True
    except Exception as e:
        st.error(f"Error updating price in Firestore: {str(e)}")
        return False

def save_to_firestore(db, image_base64, price, phash, dhash, whash, crypto_hash, original_filename):
    """Save image data to Firestore"""
    try:
        doc_id = str(uuid.uuid4())
        doc_ref = db.collection('images').document(doc_id)
        
        doc_ref.set({
            'image_base64': image_base64,
            'price': price,
            'phash': phash,
            'dhash': dhash,
            'whash': whash,
            'crypto_hash': crypto_hash,
            'original_filename': original_filename,
            'timestamp': datetime.now(),
            'last_updated': datetime.now(),
            'doc_id': doc_id
        })
        
        return doc_id
    except Exception as e:
        st.error(f"Error saving to Firestore: {str(e)}")
        return None

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
    """Process multiple images and save to Firestore or update existing ones"""
    results = []
    errors = []
    duplicates_updated = []
    
    for uploaded_file in uploaded_files:
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Read and open image
            image = Image.open(uploaded_file)
            
            # Calculate cryptographic hash for duplicate detection
            image_bytes = image_to_bytes(image)
            crypto_hash = calculate_cryptographic_hash(image_bytes)
            
            # Check if image already exists
            existing_doc, existing_doc_id = check_duplicate_image(db, crypto_hash)
            
            if existing_doc:
                # Image already exists, update price
                old_price = existing_doc.get('price', 0)
                if update_price_in_firestore(db, existing_doc_id, price, uploaded_file.name):
                    duplicates_updated.append({
                        'file_name': uploaded_file.name,
                        'doc_id': existing_doc_id,
                        'old_price': old_price,
                        'new_price': price,
                        'crypto_hash': crypto_hash,
                        'original_filename': existing_doc.get('original_filename', 'Unknown')
                    })
                else:
                    errors.append(f"Failed to update price for duplicate image {uploaded_file.name}")
                continue
            
            # New image, calculate perceptual hashes
            phash, dhash, whash = calculate_hashes(image)
            
            if not all([phash, dhash, whash]):
                errors.append(f"Failed to calculate perceptual hashes for {uploaded_file.name}")
                continue
            
            # Compress image to base64
            image_base64 = compress_image_to_base64(image)
            
            # Save to Firestore
            doc_id = save_to_firestore(
                db, image_base64, price, phash, dhash, whash, crypto_hash, uploaded_file.name
            )
            
            if doc_id:
                results.append({
                    'file_name': uploaded_file.name,
                    'doc_id': doc_id,
                    'image': image,
                    'phash': phash,
                    'dhash': dhash,
                    'whash': whash,
                    'crypto_hash': crypto_hash
                })
            else:
                errors.append(f"Failed to save {uploaded_file.name} to Firestore")
                
        except Exception as e:
            errors.append(f"Error processing {uploaded_file.name}: {str(e)}")
    
    return results, errors, duplicates_updated

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
            'crypto_hash': row.get('crypto_hash', 'N/A'),
            'timestamp': row['timestamp'],
            'last_updated': row.get('last_updated', row['timestamp']),
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
            
            # Show duplicate detection info
            if 'crypto_hash' in df.columns:
                unique_hashes = df['crypto_hash'].nunique()
                total_images = len(df)
                st.metric("Unique Images", unique_hashes)
                if unique_hashes < total_images:
                    st.metric("Duplicates", total_images - unique_hashes)
        
        return page

def render_home_page(db):
    """Render the home page"""
    st.title("GLOW STORE")
    
    st.markdown("""
    Welcome to the **GLOW STORE** Reverse image search! This tool allows you to upload multiple images, calculate their perceptual hashes, and store them in Firestore for easy retrieval and comparison.
    
    **✨ New Features:**
    - **Duplicate Detection**: Automatically detects duplicate images using cryptographic hashing
    - **Price Updates**: Updates prices for existing images instead of creating duplicates
    - **Enhanced Tracking**: Tracks when images were last updated
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
            if 'crypto_hash' in df.columns:
                unique_images = df['crypto_hash'].nunique()
                st.metric("Unique Images", unique_images)
            else:
                unique_prices = df['price'].nunique()
                st.metric("Unique Prices", unique_prices)
        
        with col4:
            if 'last_updated' in df.columns:
                latest = df['last_updated'].max()
                st.metric("Latest Update", str(latest)[:19])
            elif 'timestamp' in df.columns:
                latest = df['timestamp'].max()
                st.metric("Latest Entry", str(latest)[:19])
            else:
                st.metric("Latest Entry", "N/A")
        
        # Show duplicate information if available
        if 'crypto_hash' in df.columns:
            total_images = len(df)
            unique_images = df['crypto_hash'].nunique()
            duplicates = total_images - unique_images
            
            if duplicates > 0:
                st.warning(f"⚠️ Found {duplicates} duplicate image(s) in database. Use the upload feature to update prices.")
        
        # Recent additions
        st.subheader("🕒 Recent Activity")
        if 'last_updated' in df.columns:
            recent_df = df.sort_values('last_updated', ascending=False).head(5)
            display_df = recent_df[['original_filename', 'price', 'last_updated']].copy()
            display_df['last_updated'] = display_df['last_updated'].astype(str).str[:19]
            display_df = display_df.rename(columns={'last_updated': 'last_activity'})
        else:
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
    st.markdown("""
    Upload multiple images of the same item with a shared price. Images will be compressed to thumbnails and stored in Firestore.
    
    **🔍 Duplicate Detection**: If an image already exists in the database, its price will be updated instead of creating a duplicate.
    """)
    
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
            help="Images will be compressed to 150x150 thumbnails. Duplicates will have their prices updated.",
            accept_multiple_files=True,
            key=uploader_key
        )
        
        price = st.number_input(
            "Enter price (shared for all images)",
            min_value=0,
            step=100,
            format="%d",
            help="This price will be applied to all uploaded images or used to update existing duplicates",
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
                results, errors, duplicates_updated = process_multiple_images(uploaded_files, price, db)
                
                if errors:
                    st.error("Some errors occurred:")
                    for error in errors:
                        st.error(f"• {error}")
                
                # Show duplicate updates
                if duplicates_updated:
                    st.warning(f"🔄 {len(duplicates_updated)} duplicate image(s) found and updated!")
                    
                    with st.expander("View Updated Duplicates", expanded=True):
                        for dup in duplicates_updated:
                            st.markdown(f"""
                            **File:** {dup['file_name']}  
                            **Original File:** {dup['original_filename']}  
                            **Price Updated:** £{dup['old_price']:.2f} → £{dup['new_price']:.2f}  
                            **Document ID:** {dup['doc_id']}
                            """)
                            st.markdown("---")
                
                # Show new images
                if results:
                    st.success(f"✅ {len(results)} new image(s) processed and saved to Firestore!")
                    
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
                                
                                # Display cryptographic hash
                                st.code(f"Crypto Hash: {result['crypto_hash']}", language="text")
                                
                                st.info(f"Saved to Firestore with ID: {result['doc_id']}")
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
                        
                        # Display cryptographic hash
                        st.code(f"Crypto Hash: {result['crypto_hash']}", language="text")
                        
                        st.info(f"Saved to Firestore with ID: {result['doc_id']}")
                
                # Show summary
                if results or duplicates_updated:
                    st.subheader("📊 Processing Summary")
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    with summary_col1:
                        st.metric("New Images", len(results))
                    with summary_col2:
                        st.metric("Duplicates Updated", len(duplicates_updated))
                    with summary_col3:
                        st.metric("Total Processed", len(results) + len(duplicates_updated))
                    
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
            if 'crypto_hash' in df.columns:
                unique_images = df['crypto_hash'].nunique()
                st.metric("Unique Images", unique_images)
            else:
                unique_prices = df['price'].nunique()
                st.metric("Unique Prices", unique_prices)
        with col4:
            if 'last_updated' in df.columns:
                latest_entry = str(df['last_updated'].max())[:19]
                st.metric("Latest Update", latest_entry)
            else:
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
                
                # Also calculate crypto hash for exact duplicate detection
                query_image_bytes = image_to_bytes(query_image)
                query_crypto_hash = calculate_cryptographic_hash(query_image_bytes)
                
                if not all([query_phash, query_dhash, query_whash]):
                    st.error("Failed to calculate hashes for query image")
                    return
                
                # Check for exact duplicate first
                existing_doc, existing_doc_id = check_duplicate_image(db, query_crypto_hash)
                if existing_doc:
                    st.success("🎯 Exact duplicate found!")
                    st.markdown(f"""
                    **Original File:** {existing_doc.get('original_filename', 'Unknown')}  
                    **Price:** £{existing_doc.get('price', 0):.2f}  
                    **Document ID:** {existing_doc_id}
                    """)
                    st.markdown("---")
                
                # Display query hashes
                st.subheader("Query Image Hashes")
                hash_col1, hash_col2, hash_col3 = st.columns(3)
                with hash_col1:
                    st.metric("pHash", query_phash)
                with hash_col2:
                    st.metric("dHash", query_dhash)
                with hash_col3:
                    st.metric("wHash", query_whash)
                
                # Display crypto hash
                st.code(f"Crypto Hash: {query_crypto_hash}", language="text")
                
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
                                if 'last_updated' in sim:
                                    st.markdown(f"**Last Updated:** {str(sim['last_updated'])[:19]}")
                                st.markdown(f"**Distance:** {sim['avg_distance']:.2f}")
                                
                                # Show if it's an exact duplicate
                                if sim['crypto_hash'] == query_crypto_hash:
                                    st.success("🎯 Exact Duplicate!")
                                
                                # Hash distances
                                st.markdown("**Hash Distances:**")
                                distance_col1, distance_col2, distance_col3 = st.columns(3)
                                with distance_col1:
                                    st.metric("pHash", f"{sim['phash_distance']}")
                                with distance_col2:
                                    st.metric("dHash", f"{sim['dhash_distance']}")
                                with distance_col3:
                                    st.metric("wHash", f"{sim['whash_distance']}")
                                
                                # Show crypto hash
                                with st.expander("Show Crypto Hash"):
                                    st.code(sim['crypto_hash'], language="text")
                                
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
                    
                    # Check for exact duplicates in results
                    exact_duplicates = [sim for sim in similarities if sim['crypto_hash'] == query_crypto_hash]
                    if exact_duplicates:
                        st.info(f"🔍 Found {len(exact_duplicates)} exact duplicate(s) in results")
                    
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