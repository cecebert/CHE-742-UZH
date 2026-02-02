import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("ROI Selector with Grey Intensity Analysis")
st.subheader('CHE 742 spectrophotometry experiments')
st.divider()

# 1. Upload Area
uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image with PIL
    img = Image.open(uploaded_file)
    width, height = img.size

    st.subheader("Draw your ROI on the image")
    
    # 2. Canvas Configuration
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # ROI color (transparent orange)
        stroke_width=2,
        stroke_color="#fff",
        background_image=img,               # Uploaded image as background
        update_streamlit=True,
        height=height,                      # Adapt canvas height to image
        width=width,                        # Adapt canvas width to image
        drawing_mode="rect",                # Rectangle mode
        key="canvas",
    )

    # 3. ROI Coordinates Output
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects:
            st.write("Detected ROI coordinates:")
            for obj in objects:
                # Coordinates extracted from Fabric.js JSON (used by canvas)
                roi_data = {
                    "x": obj["left"],
                    "y": obj["top"],
                    "w": obj["width"],
                    "h": obj["height"]
                }
                st.json(roi_data)
                
                # 4. Grey Intensity Analysis
                st.subheader("📊 Grey Intensity Analysis")
                
                # Convert image to numpy array
                img_array = np.array(img)
                
                # Calculate ROI coordinates
                x = int(roi_data["x"])
                y = int(roi_data["y"])
                w = int(roi_data["w"])
                h = int(roi_data["h"])
                
                # Ensure coordinates are within image bounds
                if x >= 0 and y >= 0 and w > 0 and h > 0:
                    # Extract ROI region
                    x_end = min(x + w, width)
                    y_end = min(y + h, height)
                    
                    roi_region = img_array[y:y_end, x:x_end]
                    
                    if roi_region.size > 0:
                        # Show ROI image
                        roi_img = Image.fromarray(roi_region)
                        st.image(roi_img, caption=f"Extracted ROI: {w}x{h} pixels", use_column_width=False)
                        
                        # Convert to greyscale
                        if len(roi_region.shape) == 3:  # Color image (RGB/RGBA)
                            # Average of RGB channels for greyscale
                            roi_gray = np.mean(roi_region[:, :, :3], axis=2).astype(np.uint8)
                        else:  # Already greyscale
                            roi_gray = roi_region
                        
                        # Calculate basic statistics
                        min_val = np.min(roi_gray)
                        max_val = np.max(roi_gray)
                        mean_val = np.mean(roi_gray)
                        std_val = np.std(roi_gray)
                        median_val = np.median(roi_gray)
                        
                        # Show statistics in a table
                        stats_df = pd.DataFrame({
                            "Statistic": ["Minimum", "Maximum", "Mean", "Standard Deviation", "Median"],
                            "Value": [f"{min_val:.2f}", f"{max_val:.2f}", f"{mean_val:.2f}", f"{std_val:.2f}", f"{median_val:.2f}"]
                        })
                        
                        st.write("**Intensity Statistics:**")
                        st.table(stats_df)
                        
                        # Show histogram
                        st.write("**Intensity Histogram:**")
                        hist_values, hist_bins = np.histogram(roi_gray.flatten(), bins=256, range=[0, 256])
                        st.bar_chart(hist_values)
                        
                        # Additional information
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Number of Pixels", f"{roi_gray.size:,}")
                        with col2:
                            st.metric("Intensity Range", f"{max_val - min_val:.1f}")
                        with col3:
                            st.metric("Variance", f"{np.var(roi_gray):.2f}")
                        
                        # 5. NEW: Vertical Intensity Profile
                        st.subheader("📈 Vertical Intensity Profile")
                        
                        # Calculate mean intensity for each row (average across horizontal axis)
                        vertical_profile = np.mean(roi_gray, axis=1)
                        
                        # Create figure for the profile plot
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Plot 1: Intensity profile vs vertical position
                        y_positions = np.arange(len(vertical_profile))
                        ax1.plot(vertical_profile, y_positions, 'b-', linewidth=2)
                        ax1.set_xlabel('Mean Intensity')
                        ax1.set_ylabel('Vertical Position (pixels)')
                        ax1.set_title('Vertical Intensity Profile')
                        ax1.grid(True, alpha=0.3)
                        ax1.invert_yaxis()  # Invert y-axis to match image coordinates
                        
                        # Add statistical lines
                        ax1.axvline(x=mean_val, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.1f}')
                        ax1.axvline(x=median_val, color='g', linestyle='--', alpha=0.7, label=f'Median: {median_val:.1f}')
                        ax1.legend()
                        
                        # Plot 2: ROI with profile visualization
                        ax2.imshow(roi_gray, cmap='gray', aspect='auto')
                        ax2.set_title('ROI with Intensity Profile')
                        ax2.set_xlabel('Horizontal Position')
                        ax2.set_ylabel('Vertical Position')
                        
                        # Overlay profile on ROI image
                        profile_normalized = (vertical_profile - min_val) / (max_val - min_val + 1e-8)
                        profile_scaled = profile_normalized * w * 0.3  # Scale for visualization
                        
                        for i, intensity in enumerate(profile_scaled):
                            ax2.plot([0, intensity], [i, i], 'r-', alpha=0.3, linewidth=0.5)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Profile statistics
                        st.write("**Vertical Profile Statistics:**")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Profile Min", f"{np.min(vertical_profile):.1f}")
                        with col2:
                            st.metric("Profile Max", f"{np.max(vertical_profile):.1f}")
                        with col3:
                            st.metric("Profile Mean", f"{np.mean(vertical_profile):.1f}")
                        with col4:
                            st.metric("Profile Std", f"{np.std(vertical_profile):.1f}")
                        
                        # Show profile data as table
                        st.write("**First 10 rows of Profile Data:**")
                        profile_df = pd.DataFrame({
                            'Row': np.arange(1, min(11, len(vertical_profile))),
                            'Y Position': np.arange(y, min(y+10, y_end)),
                            'Mean Intensity': vertical_profile[:10]
                        })
                        st.table(profile_df)
                        
                        # Option to show heatmap
                        if st.checkbox("Show intensity heatmap"):
                            # Normalize values for visualization
                            heatmap_normalized = (roi_gray - min_val) / (max_val - min_val + 1e-8)
                            
                            # Create colored heatmap
                            import matplotlib.cm as cm
                            colormap = cm.viridis
                            heatmap_colored = (colormap(heatmap_normalized) * 255).astype(np.uint8)
                            
                            heatmap_img = Image.fromarray(heatmap_colored[:, :, :3])
                            st.image(heatmap_img, caption="Intensity Heatmap", use_column_width=True)
                        
                        # Option to export data
                        if st.button("Export Statistical Data"):
                            # Create complete dataframe
                            export_df = pd.DataFrame({
                                "Statistic": ["Minimum", "Maximum", "Mean", "Standard Deviation", "Median", "Variance", "Number of Pixels",
                                             "Profile Minimum", "Profile Maximum", "Profile Mean", "Profile Std"],
                                "Value": [min_val, max_val, mean_val, std_val, median_val, np.var(roi_gray), roi_gray.size,
                                         np.min(vertical_profile), np.max(vertical_profile), np.mean(vertical_profile), np.std(vertical_profile)],
                                "X_Coordinate": [x] * 11,
                                "Y_Coordinate": [y] * 11,
                                "Width": [w] * 11,
                                "Height": [h] * 11
                            })
                            
                            # Also export full profile data
                            profile_full_df = pd.DataFrame({
                                'Row_Index': np.arange(len(vertical_profile)),
                                'Y_Position': np.arange(y, y_end),
                                'Mean_Intensity': vertical_profile
                            })
                            
                            # Convert to CSV
                            csv_stats = export_df.to_csv(index=False)
                            csv_profile = profile_full_df.to_csv(index=False)
                            
                            # Create download buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="Download Statistics CSV",
                                    data=csv_stats,
                                    file_name="roi_intensity_statistics.csv",
                                    mime="text/csv"
                                )
                            with col2:
                                st.download_button(
                                    label="Download Full Profile CSV",
                                    data=csv_profile,
                                    file_name="roi_vertical_profile.csv",
                                    mime="text/csv"
                                )
                    else:
                        st.warning("The selected ROI is outside image boundaries.")
                else:
                    st.warning("Invalid ROI coordinates.")
else:
    st.info("Start by uploading an image file above.")
st.divider()
st.markdown('<p style="font-size:12px;">Developed by Cesare Berton, 2026</p>', unsafe_allow_html=True)