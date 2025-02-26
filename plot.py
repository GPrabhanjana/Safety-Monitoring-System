import os
import sys
import importlib.util
import matplotlib.pyplot as plt
import numpy as np
import traceback
from tkinter import Tk, filedialog, Button, Label, Frame, Entry, messagebox, Text, Scrollbar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import inspect
import json
import re

class ImprovedGraphGridViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Improved Graph Grid Viewer")
        self.root.geometry("1200x800")
        
        self.graph_files = []
        self.graphs = []
        self.grid_size = (2, 2)  # Default grid size (rows, cols)
        self.sample_data_path = ""
        
        self.setup_ui()
        
    def setup_ui(self):
        # Top frame for controls
        control_frame = Frame(self.root, pady=10, padx=10)
        control_frame.pack(fill='x')
        
        # Grid size controls
        grid_frame = Frame(control_frame)
        grid_frame.pack(side='left', padx=10)
        
        Label(grid_frame, text="Grid Size:").grid(row=0, column=0)
        Label(grid_frame, text="Rows:").grid(row=0, column=1)
        self.rows_entry = Entry(grid_frame, width=5)
        self.rows_entry.insert(0, str(self.grid_size[0]))
        self.rows_entry.grid(row=0, column=2)
        
        Label(grid_frame, text="Cols:").grid(row=0, column=3)
        self.cols_entry = Entry(grid_frame, width=5)
        self.cols_entry.insert(0, str(self.grid_size[1]))
        self.cols_entry.grid(row=0, column=4)
        
        Button(grid_frame, text="Update Grid", command=self.update_grid_size).grid(row=0, column=5, padx=5)
        
        # File selection buttons
        file_frame = Frame(control_frame)
        file_frame.pack(side='left', padx=20)
        
        Button(file_frame, text="Add Graph Files", command=self.add_graph_files).pack(side='left', padx=5)
        Button(file_frame, text="Set Sample Data", command=self.set_sample_data).pack(side='left', padx=5)
        Button(file_frame, text="Clear All Files", command=self.clear_files).pack(side='left', padx=5)
        
        # Display controls
        display_frame = Frame(control_frame)
        display_frame.pack(side='right', padx=10)
        
        Button(display_frame, text="Display Graphs", command=self.display_graphs).pack(side='left', padx=5)
        Button(display_frame, text="Save Layout", command=self.save_layout).pack(side='left', padx=5)
        
        # Files list frame
        self.files_frame = Frame(self.root, pady=10, padx=10, bg='#f0f0f0')
        self.files_frame.pack(fill='x')
        
        # Files list with scrollbar
        files_list_frame = Frame(self.files_frame, bg='#f0f0f0')
        files_list_frame.pack(fill='x', expand=True)
        
        Label(files_list_frame, text="Selected Graph Files:", bg='#f0f0f0', font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.files_text = Text(files_list_frame, height=5, width=100, bg='#f0f0f0', wrap='word')
        self.files_text.pack(side='left', fill='x', expand=True, pady=5)
        scrollbar = Scrollbar(files_list_frame, command=self.files_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.files_text.config(yscrollcommand=scrollbar.set)
        self.files_text.insert('1.0', "No files selected")
        self.files_text.config(state='disabled')
        
        # Sample data info
        self.sample_data_label = Label(self.files_frame, text="Sample Data: None", 
                                     bg='#f0f0f0', anchor='w', justify='left',
                                     font=('Arial', 9, 'italic'))
        self.sample_data_label.pack(anchor='w', fill='x')
        
        # Graph display area
        self.graph_frame = Frame(self.root)
        self.graph_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_bar = Label(self.root, text="Ready", bd=1, relief='sunken', anchor='w')
        self.status_bar.pack(side='bottom', fill='x')
        
    def update_grid_size(self):
        try:
            rows = int(self.rows_entry.get())
            cols = int(self.cols_entry.get())
            if rows <= 0 or cols <= 0:
                raise ValueError("Rows and columns must be positive integers")
            self.grid_size = (rows, cols)
            self.status_bar.config(text=f"Grid size updated to {rows}×{cols}")
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
    
    def add_graph_files(self):
        files = filedialog.askopenfilenames(
            title="Select Python Graph Files",
            filetypes=[("Python files", "*.py")],
            initialdir=os.getcwd()
        )
        
        if not files:
            return
            
        for file in files:
            if file not in self.graph_files:
                self.graph_files.append(file)
        
        self.update_files_list()
    
    def set_sample_data(self):
        file_path = filedialog.askopenfilename(
            title="Select Sample Data File (JSON)",
            filetypes=[("JSON files", "*.json")],
            initialdir=os.getcwd()
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'r') as f:
                # Just try to load it to validate it's JSON
                json_data = json.load(f)
                sample_count = len(json_data) if isinstance(json_data, list) else 1
                
            self.sample_data_path = file_path
            self.sample_data_label.config(
                text=f"Sample Data: {os.path.basename(file_path)} ({sample_count} items)"
            )
            self.status_bar.config(text=f"Sample data set to {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Invalid Data File", f"Error loading JSON data: {str(e)}")
    
    def update_files_list(self):
        self.files_text.config(state='normal')
        self.files_text.delete('1.0', tk.END)
        
        if not self.graph_files:
            self.files_text.insert('1.0', "No files selected")
        else:
            files_text = "\n".join([f"{i+1}. {os.path.basename(f)}" for i, f in enumerate(self.graph_files)])
            self.files_text.insert('1.0', files_text)
        
        self.files_text.config(state='disabled')
    
    def clear_files(self):
        self.graph_files = []
        self.update_files_list()
        self.sample_data_path = ""
        self.sample_data_label.config(text="Sample Data: None")
        self.status_bar.config(text="All files cleared")
    
    def load_graph_file(self, file_path):
        """
        Modified: Carefully load a graph file and capture only the main visualizations
        """
        try:
            # Close any existing figures to start clean
            plt.close('all')
            
            # Extract the filename without extension to determine which function to call
            base_name = os.path.basename(file_path).lower().replace(".py", "")
            
            # Create a spec and import the module
            spec = importlib.util.spec_from_file_location("graph_module", file_path)
            module = importlib.util.module_from_spec(spec)
            
            # Define a custom show function to capture the figure
            original_show = plt.show
            captured_figs = []
            
            def custom_show(*args, **kwargs):
                # Get the current figure and append it to our list
                fig = plt.gcf()
                captured_figs.append(fig)
                # Don't actually show it
                return None
            
            # Replace show function
            plt.show = custom_show
            
            try:
                # Execute the module - modified to prevent running all code
                spec.loader.exec_module(module)
                
                # Based on the filename, call the appropriate function
                if "kde" in base_name and hasattr(module, 'analyze_crime_hotspots'):
                    if self.sample_data_path:
                        module.analyze_crime_hotspots(self.sample_data_path, "temp_output.json")
                    else:
                        messagebox.showinfo("Missing Data", "Please set a sample data file.")
                        return []
                
                elif "kmeans" in base_name and hasattr(module, 'CrimeClusterAnalyzer'):
                    if self.sample_data_path:
                        analyzer = module.CrimeClusterAnalyzer()
                        analyzer.load_data_from_json(self.sample_data_path)
                        analyzer.determine_optimal_clusters()
                        analyzer.perform_clustering()
                    else:
                        messagebox.showinfo("Missing Data", "Please set a sample data file.")
                        return []
                
                # If no specific function matches, try calling main if it exists
                elif hasattr(module, 'main') and self.sample_data_path:
                    sig = inspect.signature(module.main)
                    if len(sig.parameters) >= 1:
                        module.main(self.sample_data_path, "temp_output.json")
                    else:
                        module.main()
                
                # Collect all figures that were generated
                if captured_figs:
                    return captured_figs
                
                # If no figures were captured, get the current figures
                current_figs = [plt.figure(num) for num in plt.get_fignums()]
                if current_figs:
                    return current_figs
                    
                print(f"No matplotlib figures were generated from {file_path}")
                return []
                    
            finally:
                # Restore the original show function
                plt.show = original_show
                
        except Exception as e:
            print(f"Error loading {os.path.basename(file_path)}: {str(e)}")
            traceback.print_exc()
            return []
    
    def display_graphs(self):
        # Clear the current graph frame
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        
        if not self.graph_files:
            messagebox.showinfo("No Files", "Please add graph files first.")
            return
            
        if not self.sample_data_path:
            messagebox.showinfo("No Sample Data", "Please set a sample data file.")
            return
            
        # Load graphs from files
        self.graphs = []
        for file_path in self.graph_files:
            plt.close('all')  # Ensure we start fresh for each file
            figures = self.load_graph_file(file_path)
            if figures:
                self.graphs.extend(figures)
                self.status_bar.config(text=f"Loaded {len(figures)} graphs from {os.path.basename(file_path)}")
            else:
                self.status_bar.config(text=f"No graphs found in {os.path.basename(file_path)}")
        
        if not self.graphs:
            messagebox.showinfo("No Graphs", 
                              "No valid graphs were loaded from the specified files.\n\n"
                              "Try setting a sample data file if your scripts need input data.")
            return
        
        # Calculate grid dimensions if needed
        rows, cols = self.grid_size
        
        # If we have more graphs than cells, create a larger grid
        if len(self.graphs) > rows * cols:
            cols = max(cols, int(np.ceil(np.sqrt(len(self.graphs)))))
            rows = int(np.ceil(len(self.graphs) / cols))
            messagebox.showinfo("Grid Size Adjusted", 
                               f"Grid size adjusted to {rows}×{cols} to fit all graphs.")
            self.rows_entry.delete(0, tk.END)
            self.rows_entry.insert(0, str(rows))
            self.cols_entry.delete(0, tk.END)
            self.cols_entry.insert(0, str(cols))
            self.grid_size = (rows, cols)
        
        # Create a grid of frames for the graphs
        frames = []
        for r in range(rows):
            frame_row = []
            for c in range(cols):
                frame = Frame(self.graph_frame, borderwidth=1, relief="ridge")
                frame.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
                frame_row.append(frame)
            frames.append(frame_row)
        
        # Make grid cells expandable
        for r in range(rows):
            self.graph_frame.grid_rowconfigure(r, weight=1)
        for c in range(cols):
            self.graph_frame.grid_columnconfigure(c, weight=1)
        
        # Place graphs in the grid
        for i, fig in enumerate(self.graphs):
            if i >= rows * cols:
                break
                
            r, c = i // cols, i % cols
            
            # Create title label
            title = fig._suptitle.get_text() if hasattr(fig, '_suptitle') and fig._suptitle else f"Graph {i+1}"
            Label(frames[r][c], text=title, bg='#e8e8e8').pack(fill='x')
            
            # Create graph canvas
            canvas = FigureCanvasTkAgg(fig, master=frames[r][c])
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.status_bar.config(text=f"Displayed {min(len(self.graphs), rows*cols)} graphs in a {rows}×{cols} grid")
    
    def save_layout(self):
        if not self.graphs:
            messagebox.showinfo("No Graphs", "No graphs to save")
            return
            
        save_path = filedialog.asksaveasfilename(
            title="Save Layout",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF File", "*.pdf"), ("SVG File", "*.svg")]
        )
        
        if not save_path:
            return
        
        rows, cols = self.grid_size
        num_figs = min(len(self.graphs), rows * cols)
        
        try:
            fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
            
            # Make axes a 2D array even if it's a single subplot
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
                
            # Turn off all axes initially
            for ax_row in axes:
                for ax in ax_row:
                    ax.axis('off')
            
            # Copy content from each graph to the new figure
            for i, src_fig in enumerate(self.graphs):
                if i >= rows * cols:
                    break
                    
                r, c = i // cols, i % cols
                
                # Get title if available
                title = src_fig._suptitle.get_text() if hasattr(src_fig, '_suptitle') and src_fig._suptitle else f"Graph {i+1}"
                
                # Clear the target axes
                axes[r][c].clear()
                
                # Copy content by saving and then reading the figure
                tmp_file = f"temp_fig_{i}.png"
                src_fig.savefig(tmp_file, dpi=300)
                img = plt.imread(tmp_file)
                axes[r][c].imshow(img)
                axes[r][c].set_title(title)
                axes[r][c].axis('off')
                
                # Remove temp file
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            
            plt.tight_layout()
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            
            self.status_bar.config(text=f"Layout saved to {save_path}")
            messagebox.showinfo("Success", f"Grid layout saved to {save_path}")
            
        except Exception as e:
            messagebox.showerror("Error Saving", f"Error: {str(e)}")
            traceback.print_exc()

def main():
    root = Tk()
    app = ImprovedGraphGridViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()