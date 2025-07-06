import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
import os
from datetime import datetime
import hashlib
import io

# Page configuration
st.set_page_config(
    page_title="GLOW Reverse Search",
    page_icon="🖼️",
    layout="wide"
)

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

def save_uploaded_image(file_bytes, original_filename):
    """Save uploaded image to a local directory and return the path"""
    # Create uploads directory if it doesn't exist
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
    
    # Generate a unique filename based on file content hash
    file_hash = hashlib.md5(file_bytes).hexdigest()
    file_extension = original_filename.split('.')[-1].lower()
    filename = f"{file_hash}.{file_extension}"
    filepath = os.path.join(uploads_dir, filename)
    
    # Save the file using the bytes
    try:
        with open(filepath, "wb") as f:
            f.write(file_bytes)
        
        # Verify the file was saved correctly by trying to open it
        with Image.open(filepath) as test_img:
            test_img.verify()
        
        return filepath
    except Exception as e:
        st.error(f"Error saving image {original_filename}: {str(e)}")
        # Clean up the file if it was partially written
        if os.path.exists(filepath):
            os.remove(filepath)
        return None

def load_or_create_db():
    """Load existing CSV database or create a new one"""
    db_file = "db.csv"
    if os.path.exists(db_file):
        try:
            df = pd.read_csv(db_file)
            # Ensure all required columns exist
            required_columns = ['image_path', 'price', 'phash', 'dhash', 'whash', 'timestamp']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ''
            return df
        except Exception as e:
            st.error(f"Error loading database: {str(e)}")
            # Create new dataframe if loading fails
            return pd.DataFrame(columns=['image_path', 'price', 'phash', 'dhash', 'whash', 'timestamp'])
    else:
        return pd.DataFrame(columns=['image_path', 'price', 'phash', 'dhash', 'whash', 'timestamp'])

def save_to_db(df):
    """Save dataframe to CSV"""
    try:
        df.to_csv("db.csv", index=False)
        return True
    except Exception as e:
        st.error(f"Error saving to database: {str(e)}")
        return False

def update_or_insert_record(df, image_path, price, phash, dhash, whash):
    """Update existing record or insert new one"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if image_path already exists
    existing_row = df[df['image_path'] == image_path]
    
    if not existing_row.empty:
        # Update existing record
        df.loc[df['image_path'] == image_path, ['price', 'phash', 'dhash', 'whash', 'timestamp']] = [price, phash, dhash, whash, timestamp]
        return df, "updated"
    else:
        # Insert new record
        new_row = pd.DataFrame({
            'image_path': [image_path],
            'price': [price],
            'phash': [phash],
            'dhash': [dhash],
            'whash': [whash],
            'timestamp': [timestamp]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        return df, "inserted"

def process_multiple_images(uploaded_files, price):
    """Process multiple images and return results"""
    results = []
    errors = []
    
    for uploaded_file in uploaded_files:
        try:
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            
            # Read the file bytes
            file_bytes = uploaded_file.read()
            
            # Validate that we have actual data
            if not file_bytes:
                errors.append(f"Empty file or could not read: {uploaded_file.name}")
                continue
            
            # Reset file pointer again for potential reuse
            uploaded_file.seek(0)
            
            # Open image directly from bytes
            image = Image.open(io.BytesIO(file_bytes))
            
            # Convert to RGB if necessary (handles RGBA, P mode images)
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
            
            # Save the uploaded image
            image_path = save_uploaded_image(file_bytes, uploaded_file.name)
            
            if image_path is None:
                errors.append(f"Failed to save image: {uploaded_file.name}")
                continue
            
            # Calculate hashes
            phash, dhash, whash = calculate_hashes(image)
            
            if phash and dhash and whash:
                results.append({
                    'file_name': uploaded_file.name,
                    'image_path': image_path,
                    'image': image,
                    'phash': phash,
                    'dhash': dhash,
                    'whash': whash
                })
            else:
                errors.append(f"Failed to calculate hashes for {uploaded_file.name}")
                
        except Exception as e:
            errors.append(f"Error processing {uploaded_file.name}: {str(e)}")
    
    return results, errors

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
        
        # Calculate average distance (you can adjust weighting here)
        avg_distance = (phash_dist + dhash_dist + whash_dist) / 3
        
        similarities.append({
            'index': idx,
            'image_path': row['image_path'],
            'price': row['price'],
            'phash': row['phash'],
            'dhash': row['dhash'],
            'whash': row['whash'],
            'timestamp': row['timestamp'],
            'phash_distance': phash_dist,
            'dhash_distance': dhash_dist,
            'whash_distance': whash_dist,
            'avg_distance': avg_distance
        })
    
    # Sort by average distance (lower is more similar)
    similarities.sort(key=lambda x: x['avg_distance'])
    
    return similarities[:top_n]

# Navigation
def render_sidebar():
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
        df = load_or_create_db()
        if not df.empty:
            st.subheader("📊 Database Info")
            st.metric("Total Images", len(df))
            if 'price' in df.columns and df['price'].dtype in ['float64', 'int64']:
                avg_price = df['price'].mean()
                st.metric("Avg Price", f"£{avg_price:.2f}")
        
        return page

def render_home_page():
    """Render the home page"""
    st.title("GLOW STORE")
    
    st.markdown("""
    Welcome to the **GLOW STORE** Reverse image search! This tool allows you to upload multiple images of the same item, calculate their perceptual hashes, and store them in a database for easy retrieval and comparison.
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
    df = load_or_create_db()
    if not df.empty:
        st.subheader("📈 Quick Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Images", len(df))
        
        with col2:
            if 'price' in df.columns and df['price'].dtype in ['float64', 'int64']:
                avg_price = df['price'].mean()
                st.metric("Average Price", f"£{avg_price:.2f}")
            else:
                st.metric("Average Price", "N/A")
        
        with col3:
            if 'price' in df.columns:
                unique_prices = df['price'].nunique()
                st.metric("Unique Prices", unique_prices)
            else:
                st.metric("Unique Prices", "N/A")
        
        with col4:
            if 'timestamp' in df.columns:
                latest = df['timestamp'].max()
                st.metric("Latest Entry", latest)
            else:
                st.metric("Latest Entry", "N/A")
        
        # Recent additions
        st.subheader("🕒 Recent Additions")
        if 'timestamp' in df.columns:
            recent_df = df.sort_values('timestamp', ascending=False).head(5)
            st.dataframe(recent_df[['image_path', 'price', 'timestamp']], use_container_width=True)
        else:
            st.info("No timestamp data available")
    
    else:
        st.info("📝 No images in database yet. Use the **Upload Images** page to get started!")
    
    st.markdown("---")
    st.markdown("Use the sidebar to navigate between different features.")

def render_upload_page():
    """Render the upload images page"""
    st.title("📤 Multi-Image Hash Calculator")
    st.markdown("Upload multiple images of the same item with a shared price. Each image will get its own row in the database with the same price and individual hashes.")
    # Initialize session state
    if 'uploader_counter' not in st.session_state:
        st.session_state.uploader_counter = 0
    if 'price_value' not in st.session_state:
        st.session_state.price_value = 0
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Images")
        
        # File uploader with key that changes to clear it
        uploader_key = f"file_uploader_{st.session_state.uploader_counter}"
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'],
            help="Supported formats: PNG, JPG, JPEG, GIF, BMP, TIFF",
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
        
        # Add a clear button
        if uploaded_files:
            if st.button("Clear Images", use_container_width=True):
                # Increment counter to change the uploader key, effectively clearing it
                st.session_state.uploader_counter += 1
                st.session_state.price_value = 0 # Reset price
                st.rerun()
    
    with col2:
        # Show preview whenever there are uploaded files
        if uploaded_files:
            st.subheader(f"Image Previews ({len(uploaded_files)} files)")
            
            # Display images in a grid
            if len(uploaded_files) == 1:
                try:
                    uploaded_files[0].seek(0)
                    image = Image.open(uploaded_files[0])
                    uploaded_files[0].seek(0)
                    st.image(image, caption=uploaded_files[0].name, use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying {uploaded_files[0].name}: {str(e)}")
            elif len(uploaded_files) <= 4:
                # Display 2x2 grid for up to 4 images
                if len(uploaded_files) <= 2:
                    cols = st.columns(len(uploaded_files))
                else:
                    cols = st.columns(2)
                    
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Reset file pointer before reading
                        uploaded_file.seek(0)
                        image = Image.open(uploaded_file)
                        # Reset again after opening
                        uploaded_file.seek(0)
                        if len(uploaded_files) <= 2:
                            cols[i].image(image, caption=uploaded_file.name, use_column_width=True)
                        else:
                            cols[i % 2].image(image, caption=uploaded_file.name, use_column_width=True)
                    except Exception as e:
                        st.error(f"Error displaying {uploaded_file.name}: {str(e)}")
            else:
                # For more than 4 images, show first 4 and indicate how many more
                cols = st.columns(2)
                for i in range(min(4, len(uploaded_files))):
                    try:
                        # Reset file pointer before reading
                        uploaded_files[i].seek(0)
                        image = Image.open(uploaded_files[i])
                        # Reset again after opening
                        uploaded_files[i].seek(0)
                        cols[i % 2].image(image, caption=uploaded_files[i].name, use_column_width=True)
                    except Exception as e:
                        st.error(f"Error displaying {uploaded_files[i].name}: {str(e)}")
                
                if len(uploaded_files) > 4:
                    st.info(f"... and {len(uploaded_files) - 4} more images")
    
    # Process the images when button is clicked
    if process_button and uploaded_files:
        with st.spinner(f"Processing {len(uploaded_files)} images..."):
            try:
                # Process all images
                results, errors = process_multiple_images(uploaded_files, price)
                
                if errors:
                    st.error("Some errors occurred:")
                    for error in errors:
                        st.error(f"• {error}")
                
                if results:
                    # Load database
                    df = load_or_create_db()
                    
                    # Process each result
                    updated_count = 0
                    inserted_count = 0
                    
                    for result in results:
                        df, action = update_or_insert_record(
                            df, 
                            result['image_path'], 
                            price, 
                            result['phash'], 
                            result['dhash'], 
                            result['whash']
                        )
                        
                        if action == "updated":
                            updated_count += 1
                        else:
                            inserted_count += 1
                    
                    # Save to database
                    if save_to_db(df):
                        # Display success message
                        success_msg = []
                        if inserted_count > 0:
                            success_msg.append(f"{inserted_count} new records added")
                        if updated_count > 0:
                            success_msg.append(f"{updated_count} records updated")
                        
                        st.success(f"✅ {' and '.join(success_msg)} successfully!")
                        
                        # Display results in tabs
                        if len(results) > 1:
                            tabs = st.tabs([f"Image {i+1}" for i in range(len(results))])
                            
                            for i, (tab, result) in enumerate(zip(tabs, results)):
                                with tab:
                                    st.subheader(f"Results for {result['file_name']}")
                                    
                                    # Display the image
                                    st.image(result['image'], caption=result['file_name'], width=300)
                                    
                                    # Display hashes in columns
                                    hash_col1, hash_col2, hash_col3 = st.columns(3)
                                    with hash_col1:
                                        st.metric("pHash", result['phash'])
                                    with hash_col2:
                                        st.metric("dHash", result['dhash'])
                                    with hash_col3:
                                        st.metric("wHash", result['whash'])
                                    
                                    st.info(f"Image saved to: {result['image_path']}")
                        else:
                            # Single image results
                            result = results[0]
                            st.subheader("Calculated Hashes")
                            hash_col1, hash_col2, hash_col3 = st.columns(3)
                            
                            with hash_col1:
                                st.metric("pHash", result['phash'])
                            with hash_col2:
                                st.metric("dHash", result['dhash'])
                            with hash_col3:
                                st.metric("wHash", result['whash'])
                            
                            st.info(f"Image saved to: {result['image_path']}")
                        
                        # Clear the uploader and reset price after successful processing
                        st.session_state.uploader_counter += 1
                        st.session_state.price_value = 0
                        
                        # Force a rerun to clear the uploader
                        st.rerun()
                    
            except Exception as e:
                st.error(f"Error processing images: {str(e)}")
    
    elif process_button and not uploaded_files:
        st.warning("Please upload at least one image first!")
    
    # Display current database
    st.subheader("📊 Current Database")
    df = load_or_create_db()
    
    if not df.empty:
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            avg_price = df['price'].mean() if 'price' in df.columns and df['price'].dtype in ['float64', 'int64'] else 0
            st.metric("Average Price", f"£{avg_price:.2f}")
        with col3:
            unique_prices = df['price'].nunique() if 'price' in df.columns else 0
            st.metric("Unique Prices", unique_prices)
        with col4:
            latest_entry = df['timestamp'].max() if 'timestamp' in df.columns else "N/A"
            st.metric("Latest Entry", latest_entry)
        
      
    else:
        st.info("No records in database yet. Upload images to get started!")

def render_search_page():
    """Render the search similar images page"""
    st.title("🔍 Find Similar Images")
    st.markdown("Upload an image to find the most similar images in your database based on perceptual hashing.")
    
    # Check if database has images
    df = load_or_create_db()
    if df.empty:
        st.warning("⚠️ No images in database yet. Please upload some images first using the **Upload Images** page.")
        return
    
    # Initialize session state for search
    if 'search_uploader_counter' not in st.session_state:
        st.session_state.search_uploader_counter = 0
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Query Image")
        
        # File uploader for query image
        uploader_key = f"search_uploader_{st.session_state.search_uploader_counter}"
        uploaded_file = st.file_uploader(
            "Choose a query image",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'],
            help="Upload an image to find similar images in your database",
            key=uploader_key
        )
        
        # Number of results to show
        num_results = st.slider(
            "Number of similar images to show",
            min_value=1,
            max_value=min(10, len(df)),
            value=3,
            help="Maximum number of similar images to display"
        )
        
        # Hash weighting options
        st.subheader("Advanced Options")
        with st.expander("Hash Weighting"):
            st.markdown("Adjust the importance of different hash types:")
            phash_weight = st.slider("pHash Weight", 0.0, 2.0, 1.0, 0.1)
            dhash_weight = st.slider("dHash Weight", 0.0, 2.0, 1.0, 0.1)
            whash_weight = st.slider("wHash Weight", 0.0, 2.0, 1.0, 0.1)
        
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
    
    # Process search when button is clicked
    if search_button and uploaded_file:
        with st.spinner("Calculating hashes and finding similar images..."):
            try:
                # Process query image
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                uploaded_file.seek(0)
                
                # Open and process query image
                query_image = Image.open(io.BytesIO(file_bytes))
                if query_image.mode in ('RGBA', 'P'):
                    query_image = query_image.convert('RGB')
                
                # Calculate hashes for query image
                query_phash, query_dhash, query_whash = calculate_hashes(query_image)
                
                if not all([query_phash, query_dhash, query_whash]):
                    st.error("Failed to calculate hashes for query image")
                    return
                
                # Display query image hashes
                st.subheader("Query Image Hashes")
                hash_col1, hash_col2, hash_col3 = st.columns(3)
                with hash_col1:
                    st.metric("pHash", query_phash)
                with hash_col2:
                    st.metric("dHash", query_dhash)
                with hash_col3:
                    st.metric("wHash", query_whash)
                
                # Find similar images with weighted distances
                similarities = []
                
                for idx, row in df.iterrows():
                    # Calculate distances for each hash type
                    phash_dist = hamming_distance(query_phash, str(row['phash']))
                    dhash_dist = hamming_distance(query_dhash, str(row['dhash']))
                    whash_dist = hamming_distance(query_whash, str(row['whash']))
                    
                    # Calculate weighted average distance
                    total_weight = phash_weight + dhash_weight + whash_weight
                    if total_weight > 0:
                        weighted_distance = (
                            phash_dist * phash_weight + 
                            dhash_dist * dhash_weight + 
                            whash_dist * whash_weight
                        ) / total_weight
                    else:
                        weighted_distance = (phash_dist + dhash_dist + whash_dist) / 3
                    
                    similarities.append({
                        'index': idx,
                        'image_path': row['image_path'],
                        'price': row['price'],
                        'phash': row['phash'],
                        'dhash': row['dhash'],
                        'whash': row['whash'],
                        'timestamp': row['timestamp'],
                        'phash_distance': phash_dist,
                        'dhash_distance': dhash_dist,
                        'whash_distance': whash_dist,
                        'weighted_distance': weighted_distance
                    })
                
                # Sort by weighted distance (lower is more similar)
                similarities.sort(key=lambda x: x['weighted_distance'])
                top_similarities = similarities[:num_results]
                
                if top_similarities:
                    st.subheader(f"🎯 Top {len(top_similarities)} Similar Images")
                    
                    # Display results
                    for i, sim in enumerate(top_similarities):
                        with st.container():
                            st.markdown(f"### Result #{i+1}")
                            
                            # Create columns for image and details
                            img_col, details_col = st.columns([1, 1])
                            
                            with img_col:
                                # Try to display the image
                                try:
                                    if os.path.exists(sim['image_path']):
                                        with Image.open(sim['image_path']) as img:
                                            st.image(img, caption=f"Similarity Score: {sim['weighted_distance']:.2f}", use_column_width=True)
                                    else:
                                        st.error(f"Image not found: {sim['image_path']}")
                                except Exception as e:
                                    st.error(f"Error loading image: {str(e)}")
                            
                            with details_col:
                                st.markdown(f"**Image Path:** `{sim['image_path']}`")
                                st.markdown(f"**Price:** £{sim['price']:.2f}")
                                st.markdown(f"**Timestamp:** {sim['timestamp']}")
                                st.markdown(f"**Overall Distance:** {sim['weighted_distance']:.2f}")
                                
                                # Hash distances
                                st.markdown("**Hash Distances:**")
                                distance_col1, distance_col2, distance_col3 = st.columns(3)
                                with distance_col1:
                                    st.metric("pHash", f"{sim['phash_distance']}")
                                with distance_col2:
                                    st.metric("dHash", f"{sim['dhash_distance']}")
                                with distance_col3:
                                    st.metric("wHash", f"{sim['whash_distance']}")
                                
                                # Similarity percentage (rough estimate)
                                max_distance = 64  # Maximum possible hamming distance for 64-bit hashes
                                similarity_pct = max(0, (max_distance - sim['weighted_distance']) / max_distance * 100)
                                st.progress(similarity_pct / 100)
                                st.markdown(f"**Estimated Similarity:** {similarity_pct:.1f}%")
                            
                            st.markdown("---")
                    
                    # Summary statistics
                    st.subheader("📈 Search Summary")
                    
                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                    
                    with summary_col1:
                        st.metric("Images Searched", len(df))
                    
                    with summary_col2:
                        best_match = top_similarities[0]['weighted_distance']
                        st.metric("Best Match Score", f"{best_match:.2f}")
                    
                    with summary_col3:
                        avg_distance = sum(sim['weighted_distance'] for sim in top_similarities) / len(top_similarities)
                        st.metric("Average Distance", f"{avg_distance:.2f}")
                    
                    with summary_col4:
                        price_range = [sim['price'] for sim in top_similarities if pd.notnull(sim['price'])]
                        if price_range:
                            avg_price = sum(price_range) / len(price_range)
                            st.metric("Avg Price of Matches", f"£{avg_price:.2f}")
                        else:
                            st.metric("Avg Price of Matches", "N/A")
                    
                    # Export results
                    st.subheader("📁 Export Results")
                    results_df = pd.DataFrame(top_similarities)
                    results_csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Search Results",
                        data=results_csv,
                        file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.info("No similar images found in the database.")
                    
            except Exception as e:
                st.error(f"Error during search: {str(e)}")
    
    elif search_button and not uploaded_file:
        st.warning("Please upload a query image first!")
    
   
# Main application
def main():
    # Render sidebar navigation
    page = render_sidebar()
    
    # Render the appropriate page
    if page == "🏠 Home":
        render_home_page()
    elif page == "📤 Upload Images":
        render_upload_page()
    elif page == "🔍 Find Similar Images":
        render_search_page()

if __name__ == "__main__":
    main()