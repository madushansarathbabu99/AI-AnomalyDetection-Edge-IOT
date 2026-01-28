#type: ignore


import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import subprocess
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import webbrowser
from datetime import datetime

# Import  models and utilities
try:
    from src.data_generator import generate_synthetic_iot_data
    from src.train_all_models import train_all_models
    from src.models.isolation_forest_model import IsolationForestDetector
    from src.models.autoencoder_model import AutoencoderDetector
    from src.models.lstm_model import LSTMDetector
    from utils.config import (
        DATA_PATH, SCALER_PATH, SEQUENCE_LENGTH, RANDOM_SEED, FEATURE_COLUMNS
    )
    import joblib
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all source files are in the correct directory structure.")
    sys.exit(1)

np.random.seed(RANDOM_SEED)


class AnomalyDetectionGUI:
    """Main GUI application for Edge IoT Anomaly Detection System."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Edge IoT Anomaly Detection System - Control Center")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        # State variables
        self.models_loaded = False
        self.training_complete = False
        self.detection_running = False
        self.streamlit_process = None
        
        # Detection parameters
        self.delay_var = tk.DoubleVar(value=0.5)
        self.inject_rate_var = tk.IntVar(value=15)
        
        # Detection data
        self.detection_queue = queue.Queue()
        self.detection_history = {
            'timestamps': [],
            'cpu': [],
            'memory': [],
            'net_in': [],
            'net_out': [],
            'temperature': [],
            'failed_auth': [],
            'iso_pred': [],
            'ae_pred': [],
            'lstm_pred': [],
            'actual': []
        }
        self.max_history = 100
        
        # Model evaluation results
        self.evaluation_results = None
        
        # Models
        self.iso_model = None
        self.ae_model = None
        self.lstm_model = None
        self.scaler = None
        self.sequence_buffer = []
        
        # Setup GUI
        self.setup_styles()
        self.create_widgets()
        
        # Start queue processor
        self.process_queue()
    
    def setup_styles(self):
        """Setup custom styles for widgets."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        bg_dark = '#1e1e1e'
        bg_medium = '#2d2d2d'
        bg_light = '#3d3d3d'
        fg_color = '#ffffff'
        accent_color = '#0078d4'
        success_color = '#28a745'
        warning_color = '#ffc107'
        danger_color = '#dc3545'
        
        style.configure('Dark.TFrame', background=bg_dark)
        style.configure('Medium.TFrame', background=bg_medium)
        style.configure('Light.TFrame', background=bg_light)
        
        style.configure('Title.TLabel', 
                       background=bg_dark, 
                       foreground=fg_color, 
                       font=('Segoe UI', 16, 'bold'))
        
        style.configure('Header.TLabel', 
                       background=bg_medium, 
                       foreground=fg_color, 
                       font=('Segoe UI', 12, 'bold'))
        
        style.configure('Status.TLabel', 
                       background=bg_medium, 
                       foreground=fg_color, 
                       font=('Segoe UI', 10))
        
        style.configure('Success.TLabel', 
                       background=bg_medium, 
                       foreground=success_color, 
                       font=('Segoe UI', 10, 'bold'))
        
        style.configure('Danger.TLabel', 
                       background=bg_medium, 
                       foreground=danger_color, 
                       font=('Segoe UI', 10, 'bold'))
    
    def create_widgets(self):
        """Create all GUI widgets with scrollable canvas."""
        
        # Main container with scrollbar
        main_container = ttk.Frame(self.root, style='Dark.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(main_container, bg='#1e1e1e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        
        # Create scrollable frame
        scrollable_frame = ttk.Frame(canvas, style='Dark.TFrame')
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux scroll up
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))   # Linux scroll down
        
        # Content container inside scrollable frame
        content_container = ttk.Frame(scrollable_frame, style='Dark.TFrame')
        content_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_frame = ttk.Frame(content_container, style='Dark.TFrame')
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(
            title_frame,
            text="Edge IoT Anomaly Detection System - Control Center",
            style='Title.TLabel'
        )
        title_label.pack(side=tk.LEFT)
        
        # Create Notebook (Tabs)
        self.notebook = ttk.Notebook(content_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Configure notebook style to fit container
        style = ttk.Style()
        style.configure('TNotebook', background='#1e1e1e')
        style.configure('TNotebook.Tab', padding=[20, 10], font=('Segoe UI', 10, 'bold'))
        
        # Tab 1: Training & Control
        control_tab = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(control_tab, text="  🎛️ Control Panel  ")
        
        # Tab 2: Live Monitoring
        monitor_tab = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(monitor_tab, text="  📊 Live Monitoring  ")
        
        # Tab 3: Model Evaluation
        eval_tab = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(eval_tab, text="  📈 Model Evaluation  ")
        
        # Tab 4: Metrics Dashboard
        metrics_tab = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(metrics_tab, text="  📉 Metrics Dashboard  ")
        
        # Populate tabs
        self.create_control_tab(control_tab)
        self.create_monitoring_tab(monitor_tab)
        self.create_evaluation_tab(eval_tab)
        self.create_metrics_tab(metrics_tab)
        
        # Store canvas reference for cleanup
        self.scroll_canvas = canvas
    
    def create_control_tab(self, parent):
        """Create control panel tab with training and detection controls."""
        # Create three main sections
        top_section = ttk.Frame(parent, style='Dark.TFrame')
        top_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Training and Control
        left_panel = ttk.Frame(top_section, style='Medium.TFrame')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5), pady=0)
        
        # Right panel - Logs
        right_panel = ttk.Frame(top_section, style='Medium.TFrame')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create sections
        self.create_training_section(left_panel)
        self.create_control_section(left_panel)
        self.create_dashboard_section(left_panel)
        self.create_log_section(right_panel)
    
    def create_monitoring_tab(self, parent):
        """Create live monitoring tab with real-time graphs."""
        self.create_visualization_section(parent)
    
    def create_evaluation_tab(self, parent):
        """Create model evaluation results tab with detailed individual model results."""
        # Create scrollable container for evaluation tab
        eval_canvas = tk.Canvas(parent, bg='#1e1e1e', highlightthickness=0)
        eval_scrollbar = ttk.Scrollbar(parent, orient="vertical", command=eval_canvas.yview)
        eval_scrollable = ttk.Frame(eval_canvas, style='Dark.TFrame')
        
        eval_scrollable.bind(
            "<Configure>",
            lambda e: eval_canvas.configure(scrollregion=eval_canvas.bbox("all"))
        )
        
        eval_canvas.create_window((0, 0), window=eval_scrollable, anchor="nw")
        eval_canvas.configure(yscrollcommand=eval_scrollbar.set)
        
        eval_scrollbar.pack(side="right", fill="y")
        eval_canvas.pack(side="left", fill="both", expand=True)
        
        # Mouse wheel scrolling for eval tab
        def _on_eval_mousewheel(event):
            eval_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        eval_canvas.bind_all("<Button-4>", lambda e: eval_canvas.yview_scroll(-1, "units"))
        eval_canvas.bind_all("<Button-5>", lambda e: eval_canvas.yview_scroll(1, "units"))
        
        main_frame = ttk.Frame(eval_scrollable, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title = ttk.Label(
            title_frame,
            text="Model Evaluation Results",
            style='Title.TLabel'
        )
        title.pack(side=tk.LEFT)
        
        # Evaluate button
        eval_btn = tk.Button(
            title_frame,
            text="🔄 Run Evaluation",
            command=self.run_evaluation,
            bg='#0078d4',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            padx=25,
            pady=10,
            cursor='hand2',
            state=tk.DISABLED
        )
        eval_btn.pack(side=tk.RIGHT)
        self.eval_button = eval_btn
        
        # Info label
        self.eval_info_label = ttk.Label(
            main_frame,
            text="Train and evaluate models to see detailed results here",
            style='Status.TLabel',
            font=('Segoe UI', 11)
        )
        self.eval_info_label.pack(pady=20)
        
        # Results container
        self.eval_results_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        self.eval_results_frame.pack(fill=tk.BOTH, expand=True)
    
    def create_metrics_tab(self, parent):
        """Create real-time metrics dashboard tab."""
        main_frame = ttk.Frame(parent, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title = ttk.Label(
            main_frame,
            text="Real-Time Metrics Dashboard",
            style='Title.TLabel'
        )
        title.pack(pady=(0, 20))
        
        # Metrics grid
        metrics_container = ttk.Frame(main_frame, style='Dark.TFrame')
        metrics_container.pack(fill=tk.BOTH, expand=True)
        
        # Create metric cards
        self.metric_cards = {}
        
        # Row 1: Current readings
        row1 = ttk.Frame(metrics_container, style='Dark.TFrame')
        row1.pack(fill=tk.X, pady=(0, 10))
        
        metrics_row1 = [
            ('CPU Usage', 'cpu', '%', '#00bfff'),
            ('Memory Usage', 'memory', '%', '#ffc107'),
            ('Temperature', 'temperature', '°C', '#ff6b6b')
        ]
        
        for i, (label, key, unit, color) in enumerate(metrics_row1):
            card = self.create_metric_card(row1, label, unit, color)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            self.metric_cards[key] = card
        
        # Row 2: Network metrics
        row2 = ttk.Frame(metrics_container, style='Dark.TFrame')
        row2.pack(fill=tk.X, pady=(0, 10))
        
        metrics_row2 = [
            ('Network In', 'net_in', 'KB/s', '#20c997'),
            ('Network Out', 'net_out', 'KB/s', '#6610f2'),
            ('Failed Auth', 'failed_auth', 'count', '#e83e8c')
        ]
        
        for i, (label, key, unit, color) in enumerate(metrics_row2):
            card = self.create_metric_card(row2, label, unit, color)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            self.metric_cards[key] = card
        
        # Row 3: Detection statistics
        row3 = ttk.Frame(metrics_container, style='Dark.TFrame')
        row3.pack(fill=tk.X, pady=(0, 10))
        
        # Model agreement card
        agreement_frame = ttk.LabelFrame(
            row3,
            text="  Model Agreement  ",
            style='Medium.TFrame'
        )
        agreement_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.agreement_label = ttk.Label(
            agreement_frame,
            text="--",
            style='Title.TLabel',
            font=('Segoe UI', 24, 'bold')
        )
        self.agreement_label.pack(pady=20)
        
        ttk.Label(
            agreement_frame,
            text="Models in agreement",
            style='Status.TLabel'
        ).pack()
        
        # Detection confidence card
        confidence_frame = ttk.LabelFrame(
            row3,
            text="  Detection Confidence  ",
            style='Medium.TFrame'
        )
        confidence_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.confidence_label = ttk.Label(
            confidence_frame,
            text="--",
            style='Title.TLabel',
            font=('Segoe UI', 24, 'bold')
        )
        self.confidence_label.pack(pady=20)
        
        ttk.Label(
            confidence_frame,
            text="Confidence level",
            style='Status.TLabel'
        ).pack()
        
        # Anomaly history visualization
        history_frame = ttk.LabelFrame(
            metrics_container,
            text="  Anomaly Detection History (Last 50 Samples)  ",
            style='Medium.TFrame'
        )
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Create history canvas
        self.history_fig = Figure(figsize=(12, 3), facecolor='#2d2d2d')
        self.history_ax = self.history_fig.add_subplot(111)
        self.history_ax.set_facecolor('#1e1e1e')
        self.history_ax.tick_params(colors='white', labelsize=8)
        self.history_ax.spines['bottom'].set_color('white')
        self.history_ax.spines['left'].set_color('white')
        self.history_ax.spines['top'].set_visible(False)
        self.history_ax.spines['right'].set_visible(False)
        self.history_ax.grid(True, alpha=0.2, color='white')
        self.history_ax.set_title('Detection Timeline', color='white', fontsize=10)
        self.history_ax.set_ylabel('Status', color='white', fontsize=9)
        
        self.history_canvas = FigureCanvasTkAgg(self.history_fig, master=history_frame)
        self.history_canvas.draw()
        self.history_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_metric_card(self, parent, label, unit, color):
        """Create a metric display card."""
        card = ttk.LabelFrame(parent, text=f"  {label}  ", style='Medium.TFrame')
        
        value_label = tk.Label(
            card,
            text="--",
            font=('Segoe UI', 28, 'bold'),
            bg='#2d2d2d',
            fg=color
        )
        value_label.pack(pady=(10, 5))
        
        unit_label = ttk.Label(
            card,
            text=unit,
            style='Status.TLabel',
            font=('Segoe UI', 10)
        )
        unit_label.pack(pady=(0, 10))
        
        # Store references
        card.value_label = value_label
        card.unit = unit
        
        return card
    def create_training_section(self, parent):
        """Create training control section."""
        frame = ttk.LabelFrame(
            parent,
            text="  Training & Model Management  ",
            style='Medium.TFrame'
        )
        frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Model status indicators
        status_frame = ttk.Frame(frame, style='Medium.TFrame')
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        models = ['Isolation Forest', 'Autoencoder', 'LSTM']
        self.model_status_labels = {}
        
        for i, model in enumerate(models):
            model_frame = ttk.Frame(status_frame, style='Medium.TFrame')
            model_frame.grid(row=i, column=0, sticky='w', pady=5)
            
            status_label = ttk.Label(
                model_frame,
                text="⚪",
                style='Status.TLabel',
                font=('Segoe UI', 14)
            )
            status_label.pack(side=tk.LEFT, padx=(0, 10))
            
            name_label = ttk.Label(
                model_frame,
                text=model,
                style='Status.TLabel'
            )
            name_label.pack(side=tk.LEFT)
            
            self.model_status_labels[model] = status_label
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            frame,
            mode='indeterminate',
            length=350
        )
        self.progress_bar.pack(padx=10, pady=(0, 10))
        
        # Training status
        self.training_status_label = ttk.Label(
            frame,
            text="Status: Ready to train",
            style='Status.TLabel'
        )
        self.training_status_label.pack(padx=10, pady=(0, 10))
        
        # Buttons
        button_frame = ttk.Frame(frame, style='Medium.TFrame')
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.generate_btn = tk.Button(
            button_frame,
            text="📊 Generate Data",
            command=self.generate_data,
            bg='#0078d4',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.generate_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.train_btn = tk.Button(
            button_frame,
            text="🚀 Train All Models",
            command=self.start_training,
            bg='#28a745',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.train_btn.pack(fill=tk.X)
    
    def create_control_section(self, parent):
        """Create detection control section."""
        frame = ttk.LabelFrame(
            parent,
            text="  Real-Time Detection Control  ",
            style='Medium.TFrame'
        )
        frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Parameters
        params_frame = ttk.Frame(frame, style='Medium.TFrame')
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Delay timer
        delay_frame = ttk.Frame(params_frame, style='Medium.TFrame')
        delay_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(
            delay_frame,
            text="Sample Delay (seconds):",
            style='Status.TLabel'
        ).pack(side=tk.LEFT)
        
        self.delay_scale = tk.Scale(
            delay_frame,
            from_=0.1,
            to=2.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.delay_var,
            bg='#2d2d2d',
            fg='white',
            highlightthickness=0,
            troughcolor='#1e1e1e',
            length=200
        )
        self.delay_scale.pack(side=tk.RIGHT)
        
        # Inject rate
        inject_frame = ttk.Frame(params_frame, style='Medium.TFrame')
        inject_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(
            inject_frame,
            text="Anomaly Inject Rate (%):",
            style='Status.TLabel'
        ).pack(side=tk.LEFT)
        
        self.inject_scale = tk.Scale(
            inject_frame,
            from_=0,
            to=50,
            resolution=5,
            orient=tk.HORIZONTAL,
            variable=self.inject_rate_var,
            bg='#2d2d2d',
            fg='white',
            highlightthickness=0,
            troughcolor='#1e1e1e',
            length=200
        )
        self.inject_scale.pack(side=tk.RIGHT)
        
        # Status indicator
        self.detection_status_frame = ttk.Frame(frame, style='Medium.TFrame')
        self.detection_status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.detection_status_indicator = tk.Label(
            self.detection_status_frame,
            text="●",
            font=('Segoe UI', 20),
            bg='#2d2d2d',
            fg='#6c757d'
        )
        self.detection_status_indicator.pack(side=tk.LEFT, padx=(0, 10))
        
        self.detection_status_text = ttk.Label(
            self.detection_status_frame,
            text="Detection: Stopped",
            style='Status.TLabel'
        )
        self.detection_status_text.pack(side=tk.LEFT)
        
        # Control buttons
        button_frame = ttk.Frame(frame, style='Medium.TFrame')
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.start_btn = tk.Button(
            button_frame,
            text="▶ START DETECTION",
            command=self.start_detection,
            bg='#28a745',
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=12,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.start_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.stop_btn = tk.Button(
            button_frame,
            text="⏹ STOP DETECTION",
            command=self.stop_detection,
            bg='#dc3545',
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=12,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.stop_btn.pack(fill=tk.X)
        
        # Statistics
        stats_frame = ttk.Frame(frame, style='Medium.TFrame')
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.stats_labels = {}
        stats = [
            ('Samples Processed:', 'samples'),
            ('Anomalies Detected:', 'anomalies'),
            ('Detection Rate:', 'rate')
        ]
        
        for i, (label_text, key) in enumerate(stats):
            stat_frame = ttk.Frame(stats_frame, style='Medium.TFrame')
            stat_frame.grid(row=i, column=0, sticky='w', pady=2)
            
            ttk.Label(
                stat_frame,
                text=label_text,
                style='Status.TLabel'
            ).pack(side=tk.LEFT)
            
            value_label = ttk.Label(
                stat_frame,
                text="0",
                style='Success.TLabel'
            )
            value_label.pack(side=tk.RIGHT)
            
            self.stats_labels[key] = value_label
    
    def create_dashboard_section(self, parent):
        """Create dashboard integration section."""
        frame = ttk.LabelFrame(
            parent,
            text="  Dashboard Integration  ",
            style='Medium.TFrame'
        )
        frame.pack(fill=tk.X, padx=10, pady=10)
        
        info_label = ttk.Label(
            frame,
            text="Launch the interactive Streamlit dashboard\nfor advanced analytics and visualization.",
            style='Status.TLabel',
            justify=tk.CENTER
        )
        info_label.pack(padx=10, pady=10)
        
        self.dashboard_status_label = ttk.Label(
            frame,
            text="Dashboard: Not Running",
            style='Status.TLabel'
        )
        self.dashboard_status_label.pack(padx=10, pady=(0, 10))
        
        button_frame = ttk.Frame(frame, style='Medium.TFrame')
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.launch_dashboard_btn = tk.Button(
            button_frame,
            text="🚀 Launch Dashboard",
            command=self.launch_dashboard,
            bg='#0078d4',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.launch_dashboard_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.stop_dashboard_btn = tk.Button(
            button_frame,
            text="⏹ Stop Dashboard",
            command=self.stop_dashboard,
            bg='#6c757d',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.stop_dashboard_btn.pack(fill=tk.X)
    
    def create_visualization_section(self, parent):
        """Create real-time visualization section."""
        frame = ttk.Frame(parent, style='Dark.TFrame')
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title and status
        header_frame = ttk.Frame(frame, style='Dark.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            header_frame,
            text="Real-Time System Monitoring",
            style='Header.TLabel'
        ).pack(side=tk.LEFT)
        
        self.monitor_status = ttk.Label(
            header_frame,
            text="● Waiting for detection to start",
            style='Status.TLabel',
            foreground='#6c757d'
        )
        self.monitor_status.pack(side=tk.RIGHT)
        
        # Create matplotlib figure with 3 subplots
        self.fig = Figure(figsize=(14, 8), facecolor='#2d2d2d')
        
        self.ax1 = self.fig.add_subplot(2, 3, 1)
        self.ax2 = self.fig.add_subplot(2, 3, 2)
        self.ax3 = self.fig.add_subplot(2, 3, 3)
        self.ax4 = self.fig.add_subplot(2, 3, 4)
        self.ax5 = self.fig.add_subplot(2, 3, 5)

        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white', labelsize=8)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, color='white')
        
        self.ax1.set_title('CPU Usage (%)', color='white', fontsize=11, pad=10, fontweight='bold')
        self.ax2.set_title('Memory Usage (%)', color='white', fontsize=11, pad=10, fontweight='bold')
        self.ax3.set_title('Temperature (°C)', color='white', fontsize=11, pad=10, fontweight='bold')
        self.ax4.set_title('Network In (KB/s)', color='white', fontsize=11, pad=10, fontweight='bold')
        self.ax5.set_title('Failed Auth Attempts', color='white', fontsize=11, pad=10, fontweight='bold')
      
        
        self.fig.tight_layout(pad=3.0)
        
        # Embed in tkinter
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Model predictions display
        pred_frame = ttk.LabelFrame(
            frame,
            text="  Current Model Predictions  ",
            style='Medium.TFrame'
        )
        pred_frame.pack(fill=tk.X, pady=(10, 0))
        
        pred_container = ttk.Frame(pred_frame, style='Medium.TFrame')
        pred_container.pack(fill=tk.X, padx=10, pady=10)
        
        self.prediction_labels = {}
        models = ['Isolation Forest', 'Autoencoder', 'LSTM']
        
        for i, model in enumerate(models):
            model_pred_frame = ttk.Frame(pred_container, style='Medium.TFrame')
            model_pred_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)
            
            ttk.Label(
                model_pred_frame,
                text=model,
                style='Status.TLabel',
                font=('Segoe UI', 10, 'bold')
            ).pack(pady=(5, 0))
            
            pred_label = tk.Label(
                model_pred_frame,
                text="⚪ Waiting",
                font=('Segoe UI', 12, 'bold'),
                bg='#2d2d2d',
                fg='#6c757d'
            )
            pred_label.pack(pady=(10, 5))
            
            self.prediction_labels[model] = pred_label
    
    def create_log_section(self, parent):
        """Create log display section."""
        frame = ttk.LabelFrame(
            parent,
            text="  System Logs  ",
            style='Medium.TFrame'
        )
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        frame.configure(height=150)
        
        self.log_text = scrolledtext.ScrolledText(
            frame,
            height=20,
            bg='#1e1e1e',
            fg='#00ff00',
            font=('Consolas', 9),
            insertbackground='white',
            relief=tk.FLAT
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log("System initialized. Ready for operation.")
    
    def log(self, message):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def update_model_status(self, model, status):
        """Update model status indicator."""
        symbols = {
            'pending': '⚪',
            'training': '🔄',
            'success': '🟢',
            'error': '🔴'
        }
        self.model_status_labels[model].config(text=symbols.get(status, '⚪'))
    
    def generate_data(self):
        """Generate synthetic dataset."""
        if os.path.exists(DATA_PATH):
            response = messagebox.askyesno(
                "Dataset Exists",
                "Dataset already exists. Regenerate?"
            )
            if not response:
                return
        
        def run_generation():
            try:
                self.log("Generating synthetic IoT dataset...")
                self.training_status_label.config(text="Status: Generating data...")
                self.generate_btn.config(state=tk.DISABLED)
                
                generate_synthetic_iot_data(save=True)
                
                self.log("✓ Dataset generation complete!")
                self.training_status_label.config(text="Status: Data ready for training")
                
            except Exception as e:
                self.log(f"✗ Error generating data: {str(e)}")
                self.training_status_label.config(text="Status: Error")
            finally:
                self.generate_btn.config(state=tk.NORMAL)
        
        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()
    
    def start_training(self):
        """Start model training in background thread."""
        if not os.path.exists(DATA_PATH):
            messagebox.showerror("Error", "Please generate data first!")
            return
        
        def run_training():
            try:
                self.log("Starting training pipeline...")
                self.progress_bar.start()
                self.train_btn.config(state=tk.DISABLED)
                self.training_status_label.config(text="Status: Training in progress...")
                
                # Update status for each model
                for model in ['Isolation Forest', 'Autoencoder', 'LSTM']:
                    self.update_model_status(model, 'pending')
                
                # Train Isolation Forest
                self.update_model_status('Isolation Forest', 'training')
                self.log("Training Isolation Forest...")
                time.sleep(0.5)  # Visual feedback
                
                # Train all models
                result = train_all_models()
                
                self.update_model_status('Isolation Forest', 'success')
                self.log("✓ Isolation Forest trained successfully")
                
                self.update_model_status('Autoencoder', 'success')
                self.log("✓ Autoencoder trained successfully")
                
                self.update_model_status('LSTM', 'success')
                self.log("✓ LSTM trained successfully")
                
                self.training_complete = True
                self.training_status_label.config(text="Status: Training complete!")
                self.log("All models trained successfully!")
                
                # Enable detection controls
                self.start_btn.config(state=tk.NORMAL)
                self.eval_button.config(state=tk.NORMAL)
                
                # Load models
                self.load_models()
                
            except Exception as e:
                self.log(f"✗ Error during training: {str(e)}")
                self.training_status_label.config(text="Status: Training failed")
                for model in ['Isolation Forest', 'Autoencoder', 'LSTM']:
                    self.update_model_status(model, 'error')
            finally:
                self.progress_bar.stop()
                self.train_btn.config(state=tk.NORMAL)
        
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
    
    def run_evaluation(self):
        """Run model evaluation and display results."""
        def evaluate():
            try:
                self.log("Running model evaluation...")
                self.eval_button.config(state=tk.DISABLED)
                
                from src.evaluate_all_models import evaluate_all_models
                evaluator = evaluate_all_models()
                
                self.evaluation_results = evaluator.results
                self.display_evaluation_results()
                
                self.log("✓ Evaluation complete!")
                
            except Exception as e:
                self.log(f"✗ Error during evaluation: {str(e)}")
                messagebox.showerror("Error", f"Evaluation failed: {str(e)}")
            finally:
                self.eval_button.config(state=tk.NORMAL)
        
        thread = threading.Thread(target=evaluate, daemon=True)
        thread.start()
    
    def display_evaluation_results(self):
        """Display detailed evaluation results for each model in the evaluation tab."""
        # Clear existing results
        for widget in self.eval_results_frame.winfo_children():
            widget.destroy()
        
        self.eval_info_label.config(text="Detailed Model Evaluation Results")
        
        if not self.evaluation_results:
            return
        
        # Create results display
        results_container = ttk.Frame(self.eval_results_frame, style='Dark.TFrame')
        results_container.pack(fill=tk.BOTH, expand=True)
        
        # ===== INDIVIDUAL MODEL RESULTS =====
        for idx, (model_name, results) in enumerate(self.evaluation_results.items()):
            # Model frame
            model_frame = ttk.LabelFrame(
                results_container,
                text=f"  {model_name} - Detailed Evaluation  ",
                style='Medium.TFrame'
            )
            model_frame.pack(fill=tk.X, padx=5, pady=10)
            
            inner_frame = ttk.Frame(model_frame, style='Medium.TFrame')
            inner_frame.pack(fill=tk.X, padx=15, pady=15)
            
            # Split into two columns
            left_col = ttk.Frame(inner_frame, style='Medium.TFrame')
            left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
            
            right_col = ttk.Frame(inner_frame, style='Medium.TFrame')
            right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
            
            # Left column - Key Metrics
            metrics_frame = ttk.LabelFrame(
                left_col,
                text="  Key Performance Metrics  ",
                style='Medium.TFrame'
            )
            metrics_frame.pack(fill=tk.X, pady=(0, 10))
            
            metrics_display = ttk.Frame(metrics_frame, style='Medium.TFrame')
            metrics_display.pack(padx=15, pady=15)
            
            key_metrics = [
                ('Accuracy', results['accuracy'], '#00bfff'),
                ('Precision', results['precision'], '#28a745'),
                ('Recall', results['recall'], '#ffc107'),
                ('F1-Score', results['f1_score'], '#e83e8c'),
            ]
            
            for i, (metric_name, value, color) in enumerate(key_metrics):
                metric_row = ttk.Frame(metrics_display, style='Medium.TFrame')
                metric_row.grid(row=i, column=0, sticky='ew', pady=5)
                
                ttk.Label(
                    metric_row,
                    text=f"{metric_name}:",
                    style='Status.TLabel',
                    font=('Segoe UI', 11),
                    width=15
                ).pack(side=tk.LEFT)
                
                value_label = tk.Label(
                    metric_row,
                    text=f"{value:.4f}",
                    font=('Segoe UI', 14, 'bold'),
                    bg='#2d2d2d',
                    fg=color
                )
                value_label.pack(side=tk.LEFT, padx=10)
                
                # Progress bar visualization
                progress_pct = value * 100
                progress_canvas = tk.Canvas(
                    metric_row,
                    width=100,
                    height=12,
                    bg='#1e1e1e',
                    highlightthickness=0
                )
                progress_canvas.pack(side=tk.LEFT, padx=5)
                progress_canvas.create_rectangle(
                    0, 0, progress_pct, 12,
                    fill=color,
                    outline=''
                )
            
            # Additional metrics
            add_metrics_frame = ttk.LabelFrame(
                left_col,
                text="  Additional Metrics  ",
                style='Medium.TFrame'
            )
            add_metrics_frame.pack(fill=tk.X)
            
            add_metrics_display = ttk.Frame(add_metrics_frame, style='Medium.TFrame')
            add_metrics_display.pack(padx=15, pady=15)
            
            additional = [
                ('ROC-AUC Score', f"{results['roc_auc']:.4f}" if results['roc_auc'] else "N/A"),
                ('False Positive Rate', f"{results['false_positive_rate']:.4f}"),
                ('False Negative Rate', f"{results['false_negative_rate']:.4f}"),
                ('Avg Inference Time', f"{results['avg_inference_time_ms']:.2f} ms" if results['avg_inference_time_ms'] else "N/A"),
            ]
            
            for i, (metric_name, value) in enumerate(additional):
                metric_row = ttk.Frame(add_metrics_display, style='Medium.TFrame')
                metric_row.grid(row=i, column=0, sticky='w', pady=3)
                
                ttk.Label(
                    metric_row,
                    text=f"{metric_name}:",
                    style='Status.TLabel',
                    font=('Segoe UI', 10),
                    width=20
                ).pack(side=tk.LEFT)
                
                ttk.Label(
                    metric_row,
                    text=value,
                    style='Success.TLabel',
                    font=('Segoe UI', 10)
                ).pack(side=tk.LEFT, padx=10)
            
            # Right column - Confusion Matrix
            cm_frame = ttk.LabelFrame(
                right_col,
                text="  Confusion Matrix  ",
                style='Medium.TFrame'
            )
            cm_frame.pack(fill=tk.BOTH, expand=True)
            
            cm_display = ttk.Frame(cm_frame, style='Medium.TFrame')
            cm_display.pack(padx=20, pady=20)
            
            # Confusion matrix headers
            ttk.Label(
                cm_display,
                text="",
                style='Status.TLabel',
                width=15
            ).grid(row=0, column=0)
            
            ttk.Label(
                cm_display,
                text="Predicted Normal",
                style='Header.TLabel',
                font=('Segoe UI', 10, 'bold'),
                width=15
            ).grid(row=0, column=1, padx=5, pady=5)
            
            ttk.Label(
                cm_display,
                text="Predicted Anomaly",
                style='Header.TLabel',
                font=('Segoe UI', 10, 'bold'),
                width=15
            ).grid(row=0, column=2, padx=5, pady=5)
            
            ttk.Label(
                cm_display,
                text="Actual Normal",
                style='Header.TLabel',
                font=('Segoe UI', 10, 'bold'),
                width=15
            ).grid(row=1, column=0, padx=5, pady=5)
            
            ttk.Label(
                cm_display,
                text="Actual Anomaly",
                style='Header.TLabel',
                font=('Segoe UI', 10, 'bold'),
                width=15
            ).grid(row=2, column=0, padx=5, pady=5)
            
            # Confusion matrix values
            tn = results['true_negatives']
            fp = results['false_positives']
            fn = results['false_negatives']
            tp = results['true_positives']
            
            # True Negative (correct normal)
            tn_frame = tk.Frame(cm_display, bg='#28a745', relief=tk.RAISED, bd=2)
            tn_frame.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')
            tk.Label(
                tn_frame,
                text=str(tn),
                font=('Segoe UI', 18, 'bold'),
                bg='#28a745',
                fg='white',
                width=8,
                height=2
            ).pack()
            tk.Label(
                tn_frame,
                text="True Negative",
                font=('Segoe UI', 8),
                bg='#28a745',
                fg='white'
            ).pack()
            
            # False Positive (wrong)
            fp_frame = tk.Frame(cm_display, bg='#dc3545', relief=tk.RAISED, bd=2)
            fp_frame.grid(row=1, column=2, padx=5, pady=5, sticky='nsew')
            tk.Label(
                fp_frame,
                text=str(fp),
                font=('Segoe UI', 18, 'bold'),
                bg='#dc3545',
                fg='white',
                width=8,
                height=2
            ).pack()
            tk.Label(
                fp_frame,
                text="False Positive",
                font=('Segoe UI', 8),
                bg='#dc3545',
                fg='white'
            ).pack()
            
            # False Negative (wrong)
            fn_frame = tk.Frame(cm_display, bg='#dc3545', relief=tk.RAISED, bd=2)
            fn_frame.grid(row=2, column=1, padx=5, pady=5, sticky='nsew')
            tk.Label(
                fn_frame,
                text=str(fn),
                font=('Segoe UI', 18, 'bold'),
                bg='#dc3545',
                fg='white',
                width=8,
                height=2
            ).pack()
            tk.Label(
                fn_frame,
                text="False Negative",
                font=('Segoe UI', 8),
                bg='#dc3545',
                fg='white'
            ).pack()
            
            # True Positive (correct anomaly)
            tp_frame = tk.Frame(cm_display, bg='#28a745', relief=tk.RAISED, bd=2)
            tp_frame.grid(row=2, column=2, padx=5, pady=5, sticky='nsew')
            tk.Label(
                tp_frame,
                text=str(tp),
                font=('Segoe UI', 18, 'bold'),
                bg='#28a745',
                fg='white',
                width=8,
                height=2
            ).pack()
            tk.Label(
                tp_frame,
                text="True Positive",
                font=('Segoe UI', 8),
                bg='#28a745',
                fg='white'
            ).pack()
            
            # Classification Report section below confusion matrix
            report_frame = ttk.LabelFrame(
                right_col,
                text="  Classification Report  ",
                style='Medium.TFrame'
            )
            report_frame.pack(fill=tk.X, pady=(10, 0))
            
            report_display = ttk.Frame(report_frame, style='Medium.TFrame')
            report_display.pack(padx=15, pady=15)
            
            # Calculate derived metrics for classification report
            total = tn + fp + fn + tp
            accuracy = (tn + tp) / total if total > 0 else 0
            
            report_text = f"""
Classification Summary:
• Total Samples: {total}
• Correctly Classified: {tn + tp} ({accuracy*100:.2f}%)
• Misclassified: {fp + fn} ({(1-accuracy)*100:.2f}%)

Per-Class Performance:
• Normal Class: {tn} correct, {fp} false alarms
• Anomaly Class: {tp} detected, {fn} missed
"""
            
            report_label = tk.Label(
                report_display,
                text=report_text,
                font=('Consolas', 9),
                bg='#2d2d2d',
                fg='#00ff00',
                justify=tk.LEFT,
                anchor='w'
            )
            report_label.pack(fill=tk.X)
        
        # ===== MODEL COMPARISON SUMMARY =====
        separator = ttk.Separator(results_container, orient='horizontal')
        separator.pack(fill=tk.X, pady=20)
        
        comparison_frame = ttk.LabelFrame(
            results_container,
            text="  Model Comparison Summary  ",
            style='Medium.TFrame'
        )
        comparison_frame.pack(fill=tk.X, padx=5, pady=10)
        
        comparison_display = ttk.Frame(comparison_frame, style='Medium.TFrame')
        comparison_display.pack(padx=20, pady=20)
        
        # Comparison table
        headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Inference (ms)']
        
        # Header row
        for i, header in enumerate(headers):
            label = ttk.Label(
                comparison_display,
                text=header,
                style='Header.TLabel',
                font=('Segoe UI', 10, 'bold'),
                width=15
            )
            label.grid(row=0, column=i, padx=8, pady=10, sticky='w')
        
        # Separator
        ttk.Separator(comparison_display, orient='horizontal').grid(
            row=1, column=0, columnspan=len(headers), sticky='ew', pady=5
        )
        
        # Data rows with highlighting best values
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'avg_inference_time_ms']
        best_values = {}
        
        for metric in metrics_to_compare:
            values = [
                self.evaluation_results[model][metric] 
                for model in self.evaluation_results 
                if self.evaluation_results[model][metric] is not None
            ]
            if values:
                if metric == 'avg_inference_time_ms':
                    best_values[metric] = min(values)  # Lower is better for inference time
                else:
                    best_values[metric] = max(values)  # Higher is better for other metrics
        
        for row_idx, (model_name, results) in enumerate(self.evaluation_results.items(), start=2):
            values = [
                model_name,
                f"{results['accuracy']:.4f}",
                f"{results['precision']:.4f}",
                f"{results['recall']:.4f}",
                f"{results['f1_score']:.4f}",
                f"{results['roc_auc']:.4f}" if results['roc_auc'] else "N/A",
                f"{results['avg_inference_time_ms']:.2f}" if results['avg_inference_time_ms'] else "N/A"
            ]
            
            for col_idx, (value, metric) in enumerate(zip(values, ['model'] + metrics_to_compare)):
                # Determine if this is the best value
                is_best = False
                if metric != 'model' and metric in best_values:
                    try:
                        numeric_value = float(value.replace(' ms', '')) if 'ms' not in value or value != "N/A" else None
                        if numeric_value is not None and abs(numeric_value - best_values[metric]) < 0.0001:
                            is_best = True
                    except:
                        pass
                
                label = tk.Label(
                    comparison_display,
                    text=value,
                    font=('Segoe UI', 10, 'bold' if is_best else 'normal'),
                    bg='#28a745' if is_best else '#2d2d2d',
                    fg='white',
                    width=15,
                    anchor='w',
                    padx=5,
                    pady=8
                )
                label.grid(row=row_idx, column=col_idx, padx=8, pady=2, sticky='w')
        
        # Best model recommendation
        best_model_frame = ttk.Frame(comparison_display, style='Medium.TFrame')
        best_model_frame.grid(row=row_idx+2, column=0, columnspan=len(headers), pady=(20, 0), sticky='ew')
        
        # Determine best overall model based on F1-score
        best_model = max(
            self.evaluation_results.items(),
            key=lambda x: x[1]['f1_score']
        )
        
        recommendation = tk.Label(
            best_model_frame,
            text=f"🏆 Recommended Model: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.4f})",
            font=('Segoe UI', 12, 'bold'),
            bg='#0078d4',
            fg='white',
            padx=20,
            pady=15
        )
        recommendation.pack(fill=tk.X)
    
    def load_models(self):
        """Load trained models."""
        try:
            self.log("Loading trained models...")
            
            self.scaler = joblib.load(SCALER_PATH)
            
            self.iso_model = IsolationForestDetector()
            self.iso_model.load()
            
            self.ae_model = AutoencoderDetector()
            self.ae_model.load()
            
            self.lstm_model = LSTMDetector()
            self.lstm_model.load()
            
            self.models_loaded = True
            self.log("✓ All models loaded successfully")
            
        except Exception as e:
            self.log(f"✗ Error loading models: {str(e)}")
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def start_detection(self):
        """Start real-time detection."""
        if not self.models_loaded:
            messagebox.showerror("Error", "Models not loaded. Please train first!")
            return
        
        self.detection_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.detection_status_indicator.config(fg='#28a745')
        self.detection_status_text.config(text="Detection: Running")
        self.monitor_status.config(text="● Detection Active", foreground='#28a745')
        
        # Reset statistics
        for key in ['samples', 'anomalies', 'rate']:
            self.stats_labels[key].config(text="0")
        
        # Clear history
        for key in self.detection_history:
            self.detection_history[key].clear()
        
        self.sequence_buffer.clear()
        
        self.log("Real-time detection started")
        
        # Start detection thread
        thread = threading.Thread(target=self.detection_loop, daemon=True)
        thread.start()
    
    def stop_detection(self):
        """Stop real-time detection."""
        self.detection_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.detection_status_indicator.config(fg='#6c757d')
        self.detection_status_text.config(text="Detection: Stopped")
        self.monitor_status.config(text="● Waiting for detection to start", foreground='#6c757d')
        self.log("Real-time detection stopped")
    
    def detection_loop(self):
        """Main detection loop running in background thread."""
        sample_count = 0
        anomaly_count = 0
        
        while self.detection_running:
            try:
                # Get parameters
                delay = self.delay_var.get()
                inject_rate = self.inject_rate_var.get()
                
                # Determine if this should be an anomaly
                should_inject = np.random.rand() * 100 < inject_rate
                
                # Generate sample
                if should_inject:
                    sample = np.array([[
                        np.random.normal(85, 10),
                        np.random.normal(90, 8),
                        np.random.normal(750, 120),
                        np.random.normal(700, 140),
                        np.random.normal(75, 7),
                        np.random.poisson(12)
                    ]])
                    actual_label = 1
                else:
                    sample = np.array([[
                        np.random.normal(38, 8),
                        np.random.normal(42, 10),
                        np.random.normal(220, 50),
                        np.random.normal(200, 45),
                        np.random.normal(46, 4),
                        np.random.poisson(1)
                    ]])
                    actual_label = 0
                
                sample_scaled = self.scaler.transform(sample)
                
                # Get predictions
                pred_iso = self.iso_model.predict(sample_scaled)[0]
                pred_ae = self.ae_model.predict(sample_scaled)[0]
                
                # LSTM prediction
                self.sequence_buffer.append(sample_scaled[0])
                if len(self.sequence_buffer) > SEQUENCE_LENGTH:
                    self.sequence_buffer.pop(0)
                
                if len(self.sequence_buffer) == SEQUENCE_LENGTH:
                    seq = np.array([self.sequence_buffer])
                    pred_lstm = self.lstm_model.predict(seq)[0]
                else:
                    pred_lstm = -1  # Not ready
                
                # Update statistics
                sample_count += 1
                if actual_label == 1:
                    anomaly_count += 1
                
                # Queue data for GUI update
                self.detection_queue.put({
                    'sample': sample[0],
                    'predictions': {
                        'iso': pred_iso,
                        'ae': pred_ae,
                        'lstm': pred_lstm
                    },
                    'actual': actual_label,
                    'stats': {
                        'samples': sample_count,
                        'anomalies': anomaly_count
                    }
                })
                
                time.sleep(delay)
                
            except Exception as e:
                self.log(f"Error in detection loop: {str(e)}")
                break
    
    def process_queue(self):
        """Process detection queue and update GUI."""
        try:
            while not self.detection_queue.empty():
                data = self.detection_queue.get_nowait()
                self.update_detection_display(data)
        except queue.Empty:
            pass
        finally:
            # Schedule next check
            self.root.after(100, self.process_queue)
    
    def update_detection_display(self, data):
        """Update detection visualization and statistics."""
        sample = data['sample']
        predictions = data['predictions']
        actual = data['actual']
        stats = data['stats']
        
        # Update statistics
        self.stats_labels['samples'].config(text=str(stats['samples']))
        self.stats_labels['anomalies'].config(text=str(stats['anomalies']))
        
        rate = (stats['anomalies'] / stats['samples'] * 100) if stats['samples'] > 0 else 0
        self.stats_labels['rate'].config(text=f"{rate:.1f}%")
        
        # Update history
        self.detection_history['timestamps'].append(stats['samples'])
        self.detection_history['cpu'].append(sample[0])
        self.detection_history['memory'].append(sample[1])
        self.detection_history['net_in'].append(sample[2])
        self.detection_history['net_out'].append(sample[3])
        self.detection_history['temperature'].append(sample[4])
        self.detection_history['failed_auth'].append(sample[5])
        self.detection_history['iso_pred'].append(predictions['iso'])
        self.detection_history['ae_pred'].append(predictions['ae'])
        self.detection_history['lstm_pred'].append(predictions['lstm'])
        self.detection_history['actual'].append(actual)
        
        # Trim history
        if len(self.detection_history['timestamps']) > self.max_history:
            for key in self.detection_history:
                self.detection_history[key].pop(0)
        
        # Update prediction labels
        def format_prediction(pred, actual):
            if pred == -1:
                return "⏳ Buffering", '#6c757d'
            elif pred == 1:
                return "🔴 ANOMALY", '#dc3545'
            else:
                return "🟢 Normal", '#28a745'
        
        for model_name, pred in [('Isolation Forest', predictions['iso']),
                                  ('Autoencoder', predictions['ae']),
                                  ('LSTM', predictions['lstm'])]:
            text, color = format_prediction(pred, actual)
            self.prediction_labels[model_name].config(text=text, fg=color)
        
        # Update plots
        self.update_plots()
        
        # Update metrics dashboard
        self.update_metrics_dashboard(sample, predictions, actual)
        
        # Log if anomaly detected
        if actual == 1:
            detections = []
            if predictions['iso'] == 1:
                detections.append('ISO')
            if predictions['ae'] == 1:
                detections.append('AE')
            if predictions['lstm'] == 1:
                detections.append('LSTM')
            
            if detections:
                self.log(f"⚠ Anomaly detected by: {', '.join(detections)}")
    
    def update_metrics_dashboard(self, sample, predictions, actual):
        """Update the metrics dashboard tab."""
        # Update metric cards
        metrics_update = {
            'cpu': sample[0],
            'memory': sample[1],
            'net_in': sample[2],
            'net_out': sample[3],
            'temperature': sample[4],
            'failed_auth': sample[5]
        }
        
        for key, value in metrics_update.items():
            if key in self.metric_cards:
                self.metric_cards[key].value_label.config(text=f"{value:.1f}")
        
        # Calculate model agreement
        preds = [predictions['iso'], predictions['ae'], predictions['lstm']]
        valid_preds = [p for p in preds if p != -1]
        
        if valid_preds:
            agreement = sum(p == actual for p in valid_preds)
            total = len(valid_preds)
            self.agreement_label.config(text=f"{agreement}/{total}")
            
            # Calculate confidence (how many agree on majority)
            if len(valid_preds) >= 2:
                from collections import Counter
                pred_counts = Counter(valid_preds)
                consensus, count = pred_counts.most_common(1)[0]
                confidence = (count / len(valid_preds)) * 100
                
                if confidence >= 66:
                    color = '#28a745'
                elif confidence >= 50:
                    color = '#ffc107'
                else:
                    color = '#dc3545'
                
                self.confidence_label.config(text=f"{confidence:.0f}%", foreground=color)
            else:
                self.confidence_label.config(text="--", foreground='#6c757d')
        
        # Update detection history plot
        self.update_history_plot()
    
    def update_history_plot(self):
        """Update the anomaly detection history visualization."""
        if not self.detection_history['timestamps']:
            return
        
        timestamps = self.detection_history['timestamps'][-50:]  # Last 50 samples
        actual = self.detection_history['actual'][-50:]
        iso_pred = self.detection_history['iso_pred'][-50:]
        ae_pred = self.detection_history['ae_pred'][-50:]
        lstm_pred = self.detection_history['lstm_pred'][-50:]
        
        self.history_ax.clear()
        
        # Plot detection results
        self.history_ax.plot(timestamps, actual, 'r-', linewidth=3, label='Actual', alpha=0.3)
        self.history_ax.plot(timestamps, iso_pred, 'b-', linewidth=2, label='Isolation Forest', marker='o', markersize=4)
        self.history_ax.plot(timestamps, ae_pred, 'g-', linewidth=2, label='Autoencoder', marker='s', markersize=4)
        
        # Only plot LSTM if data available
        lstm_valid = [p if p != -1 else None for p in lstm_pred]
        if any(p is not None for p in lstm_valid):
            self.history_ax.plot(timestamps, lstm_valid, 'y-', linewidth=2, label='LSTM', marker='^', markersize=4)
        
        self.history_ax.set_ylim(-0.2, 1.2)
        self.history_ax.set_yticks([0, 1])
        self.history_ax.set_yticklabels(['Normal', 'Anomaly'])
        
        # Styling
        self.history_ax.set_facecolor('#1e1e1e')
        self.history_ax.tick_params(colors='white', labelsize=8)
        self.history_ax.spines['bottom'].set_color('white')
        self.history_ax.spines['left'].set_color('white')
        self.history_ax.spines['top'].set_visible(False)
        self.history_ax.spines['right'].set_visible(False)
        self.history_ax.grid(True, alpha=0.2, color='white')
        self.history_ax.set_title('Detection Timeline', color='white', fontsize=10)
        self.history_ax.set_ylabel('Status', color='white', fontsize=9)
        self.history_ax.legend(loc='upper left', facecolor='#2d2d2d', edgecolor='white', labelcolor='white', fontsize=8)
        
        self.history_canvas.draw()
    
    def update_plots(self):
        """Update real-time plots."""
        if not self.detection_history['timestamps']:
            return
        
        timestamps = self.detection_history['timestamps']
        
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()
        
        # Plot CPU
        self.ax1.plot(timestamps, self.detection_history['cpu'], 
                     color='#00bfff', linewidth=2)
        
        # Highlight anomalies
        anomaly_indices = [i for i, a in enumerate(self.detection_history['actual']) if a == 1]
        if anomaly_indices:
            anomaly_times = [timestamps[i] for i in anomaly_indices]
            anomaly_cpu = [self.detection_history['cpu'][i] for i in anomaly_indices]
            self.ax1.scatter(anomaly_times, anomaly_cpu, color='#dc3545', 
                           s=100, marker='x', linewidths=3, zorder=5)
        
        # Plot Memory
        self.ax2.plot(timestamps, self.detection_history['memory'], 
                     color='#ffc107', linewidth=2)
        if anomaly_indices:
            anomaly_mem = [self.detection_history['memory'][i] for i in anomaly_indices]
            self.ax2.scatter(anomaly_times, anomaly_mem, color='#dc3545', 
                           s=100, marker='x', linewidths=3, zorder=5)
        
        # Plot Temperature
        self.ax3.plot(timestamps, self.detection_history['temperature'], 
                     color='#ff6b6b', linewidth=2)
        if anomaly_indices:
            anomaly_temp = [self.detection_history['temperature'][i] for i in anomaly_indices]
            self.ax3.scatter(anomaly_times, anomaly_temp, color='#dc3545', 
                           s=100, marker='x', linewidths=3, zorder=5)
            
        # Plot Network IN
        self.ax4.plot(timestamps, self.detection_history['net_in'], 
                     color='#ff6b6b', linewidth=2)
        if anomaly_indices:
            anomaly_temp = [self.detection_history['net_in'][i] for i in anomaly_indices]
            self.ax4.scatter(anomaly_times, anomaly_temp, color='#dc3545', 
                           s=100, marker='x', linewidths=3, zorder=5)
        
        # Plot Failed Auth Attempts
        self.ax5.plot(timestamps, self.detection_history['failed_auth'], 
                     color='#ff6b6b', linewidth=2)
        if anomaly_indices:
            anomaly_failed_auth = [self.detection_history['failed_auth'][i] for i in anomaly_indices]
            self.ax5.scatter(anomaly_times, anomaly_failed_auth, color='#dc3545', 
                           s=100, marker='x', linewidths=3, zorder=5)

        # Styling
        for ax, title in [(self.ax1, 'CPU Usage (%)'),
                          (self.ax2, 'Memory Usage (%)'),
                          (self.ax3, 'Temperature (°C)'),
                          (self.ax4, 'Network In (KB/s)'),
                          (self.ax5, 'Failed Auth Attempts')]:
            
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white', labelsize=8)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, color='white')
            ax.set_title(title, color='white', fontsize=10, pad=10)
            
            if len(timestamps) > 1:
                ax.set_xlim(timestamps[0], timestamps[-1])
        
    
        
        self.fig.tight_layout(pad=3.0)
        self.plot_canvas.draw()
    
    def launch_dashboard(self):
        """Launch Streamlit dashboard."""
        try:
            self.log("Launching Streamlit dashboard...")
            
            # Check if dashboard file exists
            dashboard_path = os.path.join('src', 'dashboard.py')
            if not os.path.exists(dashboard_path):
                messagebox.showerror("Error", "Dashboard file not found!")
                return
            
            # Launch Streamlit in subprocess
            self.streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", dashboard_path],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
            
            self.log("Dashboard launching on http://localhost:8501")
            self.dashboard_status_label.config(text="Dashboard: Running on localhost:8501")
            self.launch_dashboard_btn.config(state=tk.DISABLED)
            self.stop_dashboard_btn.config(state=tk.NORMAL)
            
            # Open browser after short delay
            self.root.after(3000, lambda: webbrowser.open('http://localhost:8501'))
            
        except Exception as e:
            self.log(f"✗ Error launching dashboard: {str(e)}")
            messagebox.showerror("Error", f"Failed to launch dashboard: {str(e)}")
    
    def stop_dashboard(self):
        """Stop Streamlit dashboard."""
        try:
            if self.streamlit_process:
                self.streamlit_process.terminate()
                self.streamlit_process.wait(timeout=5)
                self.streamlit_process = None
                
                self.log("Dashboard stopped")
                self.dashboard_status_label.config(text="Dashboard: Not Running")
                self.launch_dashboard_btn.config(state=tk.NORMAL)
                self.stop_dashboard_btn.config(state=tk.DISABLED)
            
        except Exception as e:
            self.log(f"Error stopping dashboard: {str(e)}")
    
    def on_closing(self):
        """Handle window closing."""
        if self.detection_running:
            self.stop_detection()
        
        if self.streamlit_process:
            self.stop_dashboard()
        
        self.root.destroy()


def main():
    """Main entry point for GUI application."""
    root = tk.Tk()
    app = AnomalyDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()


