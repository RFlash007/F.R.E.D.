import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox, PhotoImage
import threading
from queue import Queue
from datetime import datetime
import json
import os
import time
import math
import random

class ChatUI:
    def __init__(self, chat_callback):
        self.root = tk.Tk()
        self.root.title("F.R.E.D. Neural Interface")
        self.root.geometry("1200x800")
        self.root.configure(bg='#0a1520')
        
        # Set constant transparency
        self.root.attributes('-alpha', 0.95)
        
        # Set window icon
        try:
            self.root.iconbitmap("assets/fred_icon.ico")
        except:
            pass
        
        # Initialize thought process visualization
        self.thinking_particles = []
        self.particle_canvas = None
        
        # Message queue for thread-safe UI updates
        self.msg_queue = Queue()
        self.chat_callback = chat_callback
        self.conversation_history = []
        
        # Create custom styles and widgets
        self._create_styles()
        self._create_widgets()
        self._setup_layout()
        self._start_msg_checker()
        
        # Keyboard shortcuts
        self.root.bind("<Control-c>", lambda e: self._clear_chat())
        self.root.bind("<Control-q>", lambda e: self.root.quit())

    def _create_styles(self):
        style = ttk.Style()
        # Use a modern theme as a base; clam works well for custom colors
        style.theme_use('clam')
        
        # Enhanced color palette - deep blues with accent glow
        self.colors = {
            'bg_dark': '#0a1520',
            'bg_medium': '#0d1a2a', 
            'bg_light': '#132638',
            'accent': '#00bfff',
            'accent_bright': '#00dfff',
            'accent_dim': '#007dc5',
            'highlight': '#ff7b00',  # New orange highlight for alerts
            'success': '#00c853',    # Success green
            'warning': '#ffab00'     # Warning amber
        }
        
        # Main frames styling with darker, more premium look
        style.configure('Neural.TFrame', background=self.colors['bg_dark'])
        style.configure('Sidebar.TFrame', background=self.colors['bg_medium'])
        style.configure('TSeparator', background=self.colors['accent'])
        
        # Action buttons styling with enhanced holographic look
        style.configure('Action.TButton',
                        font=('Rajdhani', 11, 'bold'),
                        padding=8,
                        background=self.colors['bg_medium'],
                        foreground=self.colors['accent'])
        style.map('Action.TButton',
                  background=[('active', self.colors['bg_light'])],
                  foreground=[('active', self.colors['accent_bright'])])
        
        # Holographic buttons alternative style
        style.configure('Hologram.TButton',
                        font=('Rajdhani', 11, 'bold'),
                        padding=8,
                        background=self.colors['bg_medium'],
                        foreground=self.colors['accent_bright'])
        style.map('Hologram.TButton',
                  background=[('active', self.colors['bg_medium'])],
                  foreground=[('active', '#ffffff')])
        
        # Entry field styling with enhanced neon effects
        style.configure('Neural.TEntry',
                        fieldbackground=self.colors['bg_medium'],
                        foreground=self.colors['accent'],
                        insertcolor=self.colors['accent_bright'],
                        borderwidth=0)
        
        # Custom label styles for different UI elements
        style.configure('Neural.TLabel',
                        background=self.colors['bg_dark'],
                        foreground=self.colors['accent'],
                        font=('Rajdhani', 11))
        
        style.configure('Title.TLabel',
                        background=self.colors['bg_medium'],
                        foreground=self.colors['accent_bright'],
                        font=('Rajdhani', 24, 'bold'))
        
        style.configure('Status.TLabel',
                        background=self.colors['bg_dark'],
                        foreground=self.colors['accent'],
                        font=('Rajdhani', 10))
        
        style.configure('Metric.TLabel',
                        background=self.colors['bg_medium'],
                        foreground=self.colors['accent'],
                        font=('Rajdhani', 9))
        
    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Update colors for menus
        file_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg_medium'], fg=self.colors['accent'],
                            activebackground=self.colors['bg_light'], activeforeground=self.colors['accent_bright'])
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Clear Chat", command=self._clear_chat)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        edit_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg_medium'], fg=self.colors['accent'],
                            activebackground=self.colors['bg_light'], activeforeground=self.colors['accent_bright'])
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Copy", command=lambda: self.chat_display.event_generate("<<Copy>>"))
        edit_menu.add_command(label="Paste", command=lambda: self.input_field.event_generate("<<Paste>>"))
        
        # Add a new help menu with F.R.E.D. commands
        help_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg_medium'], fg=self.colors['accent'],
                            activebackground=self.colors['bg_light'], activeforeground=self.colors['accent_bright'])
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About F.R.E.D.", command=self._show_about)
        help_menu.add_command(label="Commands", command=self._show_commands)
    
    def _create_header(self):
        """Create header with advanced HUD-style elements and dynamic status indicators"""
        self.header_frame = ttk.Frame(self.main_frame, style='Neural.TFrame')
        
        # Left side: F.R.E.D. status indicator with pulse effect
        self.status_indicator_frame = ttk.Frame(self.header_frame, style='Neural.TFrame')
        self.status_indicator_frame.pack(side='left', fill='y')
        
        self.status_canvas = tk.Canvas(
            self.status_indicator_frame,
            width=30,
            height=30,
            bg=self.colors['bg_dark'],
            highlightthickness=0
        )
        self.status_canvas.pack(side='left', padx=10)
        
        # Create status circle with pulsing effect
        self.status_circle = self.status_canvas.create_oval(
            5, 5, 25, 25,
            fill=self.colors['accent'],
            outline=self.colors['accent_bright'],
            width=2
        )
        
        # F.R.E.D. Current Status
        self.status_text = ttk.Label(
            self.status_indicator_frame,
            text="F.R.E.D. STATUS: ONLINE",
            style='Neural.TLabel',
            font=('Rajdhani', 10, 'bold')
        )
        self.status_text.pack(side='left', padx=5)
        
        # Center: System metrics display (CPU, Memory)
        self.metrics_frame = ttk.Frame(self.header_frame, style='Neural.TFrame')
        self.metrics_frame.pack(side='left', fill='y', padx=20)
        
        # Create CPU and memory indicators
        self.cpu_frame = ttk.Frame(self.metrics_frame, style='Neural.TFrame')
        self.cpu_frame.pack(side='left', padx=10)
        
        self.cpu_label = ttk.Label(
            self.cpu_frame,
            text="CPU 0%",
            style='Metric.TLabel'
        )
        self.cpu_label.pack(anchor='w')
        
        self.cpu_canvas = tk.Canvas(
            self.cpu_frame,
            width=100,
            height=15,
            bg=self.colors['bg_dark'],
            highlightthickness=0
        )
        self.cpu_canvas.pack()
        self.cpu_bar = self.cpu_canvas.create_rectangle(
            0, 0, 10, 15,
            fill=self.colors['accent'],
            width=0
        )
        
        self.memory_frame = ttk.Frame(self.metrics_frame, style='Neural.TFrame')
        self.memory_frame.pack(side='left', padx=10)
        
        self.memory_label = ttk.Label(
            self.memory_frame,
            text="MEM 0%",
            style='Metric.TLabel'
        )
        self.memory_label.pack(anchor='w')
        
        self.memory_canvas = tk.Canvas(
            self.memory_frame,
            width=100,
            height=15,
            bg=self.colors['bg_dark'],
            highlightthickness=0
        )
        self.memory_canvas.pack()
        self.memory_bar = self.memory_canvas.create_rectangle(
            0, 0, 10, 15,
            fill=self.colors['accent'],
            width=0
        )
        
        # Right side: Time display with digital clock effect
        self.time_frame = ttk.Frame(self.header_frame, style='Neural.TFrame')
        self.time_frame.pack(side='right', fill='y')
        
        self.date_label = ttk.Label(
            self.time_frame,
            text="",
            foreground=self.colors['accent_dim'],
            background=self.colors['bg_dark'],
            font=('Rajdhani', 9)
        )
        self.date_label.pack(side='top', anchor='e', padx=10)
        
        self.time_label = ttk.Label(
            self.time_frame,
            text="",
            foreground=self.colors['accent_bright'],
            background=self.colors['bg_dark'],
            font=('Rajdhani', 14, 'bold')
        )
        self.time_label.pack(side='bottom', anchor='e', padx=10)
        
        # Start animations and updates
        self._update_time()
        self._update_metrics()

    def _create_widgets(self):
        # Create main container
        self.container = ttk.Frame(self.root, style='Neural.TFrame')
        
        # Create sidebar with JARVIS-inspired design
        self.sidebar = ttk.Frame(self.container, style='Sidebar.TFrame', width=250)
        
        # Add F.R.E.D. logo/title with glowing effect
        self.logo_frame = ttk.Frame(self.sidebar, style='Sidebar.TFrame')
        self.logo_canvas = tk.Canvas(
            self.logo_frame,
            width=200,
            height=200,
            bg='#0d1a2a',
            highlightthickness=0
        )
        # Enhanced reactor logo with additional glow and particle effects
        self.create_reactor_logo()
        
        self.sidebar_title = ttk.Label(
            self.sidebar,
            text="F.R.E.D.",
            foreground='#00bfff',
            background='#0d1a2a',
            font=('Rajdhani', 24, 'bold')
        )
        
        # Quick action buttons with icons
        self.actions_frame = ttk.Frame(self.sidebar, style='Sidebar.TFrame')
        self.create_action_buttons()
        
        # Main chat area
        self.main_frame = ttk.Frame(self.container, padding="20", style='Neural.TFrame')
        
        self._create_menu()
        self._create_header()
        
        # Chat display with custom scrollbar and futuristic font
        self.chat_frame = ttk.Frame(self.main_frame, style='Neural.TFrame')
        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            width=80,
            height=30,
            font=("Consolas", 11),
            bg='#0d1a2a',
            fg='#00bfff',
            insertbackground='#00dfff',
            relief='flat',
            borderwidth=0,
            padx=20,
            pady=20
        )
        # Custom scrollbar styling
        self.chat_display.vbar.configure(
            troughcolor='#0a1520',
            bg='#00bfff',
            activebackground='#00dfff',
            width=8
        )
        
        # Input area with glowing effect
        self.input_container = ttk.Frame(self.main_frame, style='Neural.TFrame')
        self.input_frame = ttk.Frame(
            self.input_container,
            style='Neural.TFrame',
            padding=(10, 5)
        )
        
        self.input_field = ttk.Entry(
            self.input_frame,
            width=70,
            font=("Rajdhani", 11),
            style='Neural.TEntry'
        )
        self.input_field.bind("<Return>", self._on_send)
        
        # Holographic send button with neon glow effect
        self.send_button = tk.Button(
            self.input_frame,
            text="TRANSMIT",
            command=self._on_send,
            font=('Rajdhani', 11, 'bold'),
            bg='#0d1a2a',
            fg='#00bfff',
            activebackground='#102030',
            activeforeground='#00dfff',
            relief='flat',
            padx=15,
            pady=5,
            cursor='hand2',
            bd=0
        )
        self.send_button.bind("<Enter>", self._on_button_hover)
        self.send_button.bind("<Leave>", self._on_button_leave)
        
        # Status bar with additional system info
        self.status_frame = ttk.Frame(self.main_frame, style='Neural.TFrame')
        self.status_bar = ttk.Label(
            self.status_frame,
            text="SYSTEMS READY",
            style='Neural.TLabel'
        )
        
        # Create particle canvas for thinking animation (overlayed on chat area)
        self.particle_canvas = tk.Canvas(
            self.chat_frame,
            bg='#0d1a2a',
            highlightthickness=0,
            width=self.chat_frame.winfo_width(),
            height=50
        )
        
        # Voice feedback indicator with pulsating neon ring
        self.voice_indicator = tk.Canvas(
            self.input_frame,
            width=20,
            height=20,
            bg='#0d1a2a',
            highlightthickness=0
        )
        self.voice_indicator.create_oval(
            2, 2, 18, 18,
            fill='#0d1a2a',
            outline='#00bfff',
            width=2,
            tags='voice_ring'
        )
        
        # Add neural network visualization with dynamic neon nodes
        self.neural_canvas = tk.Canvas(
            self.chat_frame,
            bg='#0d1a2a',
            highlightthickness=0,
            width=150,
            height=150
        )
        self._create_neural_network()
        
        # Environment analysis display with updated neon bars
        self.env_frame = ttk.Frame(self.main_frame, style='Neural.TFrame')
        self.env_canvas = tk.Canvas(
            self.env_frame,
            bg='#0d1a2a',
            highlightthickness=0,
            width=200,
            height=30
        )
        self._create_env_display()
        
        # Visualization overlays – hexagonal grid background and data stream effect
        self.visualization_frame = ttk.Frame(self.chat_frame, style='Neural.TFrame')
        
        self.hex_canvas = tk.Canvas(
            self.visualization_frame,
            bg='#0d1a2a',
            highlightthickness=0,
            height=self.chat_frame.winfo_height()
        )
        self._create_hex_grid()
        
        self.stream_canvas = tk.Canvas(
            self.visualization_frame,
            bg='#0d1a2a',
            highlightthickness=0,
            width=30,
            height=self.chat_frame.winfo_height()
        )
        self._create_data_stream()
        
        # Frequency analyzer visualization for added tech flair
        self.freq_canvas = tk.Canvas(
            self.chat_frame,
            bg='#0d1a2a',
            highlightthickness=0,
            height=40
        )
        self._create_frequency_analyzer()

    def create_reactor_logo(self):
        """Create an enhanced 3D Arc Reactor inspired logo with glowing, rotating layers"""
        center_x, center_y = 100, 100
        self.reactor_layers = []
        
        # Create multiple rotating layers
        for i in range(5):
            layer = {
                'rings': [],
                'particles': [],
                'z': i * 10,  # Z-depth for 3D effect
                'rotation': random.random() * 360
            }
            # Adjust radius for each layer to add depth and glow
            radius = 40 - (i * 5)
            for j in range(3):
                # Draw multiple overlapping arcs for a glowing effect
                ring = self.logo_canvas.create_arc(
                    center_x - radius - (j * 3),
                    center_y - radius - (j * 3),
                    center_x + radius + (j * 3),
                    center_y + radius + (j * 3),
                    start=0,
                    extent=300,
                    outline=self._adjust_color_brightness('#00bfff', 0.7 - (i * 0.1)),
                    width=2,
                    style='arc'
                )
                layer['rings'].append(ring)
            # Add orbiting particles with subtle oscillations
            for _ in range(6):
                angle = random.random() * 360
                particle = self.logo_canvas.create_oval(
                    0, 0, 4, 4,
                    fill='#00bfff',
                    outline='#00bfff'
                )
                layer['particles'].append({
                    'id': particle,
                    'angle': angle,
                    'speed': 1 + random.random() * 2,
                    'radius': radius
                })
            self.reactor_layers.append(layer)
        
        # Create core elements with pulsating neon glow
        self.reactor_core = {
            'inner': self.logo_canvas.create_oval(
                center_x - 15, center_y - 15,
                center_x + 15, center_y + 15,
                fill='#00bfff',
                outline='#00bfff'
            ),
            'outer': self.logo_canvas.create_oval(
                center_x - 20, center_y - 20,
                center_x + 20, center_y + 20,
                fill='',
                outline='#00bfff',
                width=2
            ),
            'energy_rings': []
        }
        for i in range(3):
            ring = self.logo_canvas.create_oval(
                center_x - 25 - (i * 5),
                center_y - 25 - (i * 5),
                center_x + 25 + (i * 5),
                center_y + 25 + (i * 5),
                fill='',
                outline=self._adjust_color_brightness('#00bfff', 0.5),
                width=1
            )
            self.reactor_core['energy_rings'].append(ring)
        
        self._animate_enhanced_reactor()

    def _animate_enhanced_reactor(self):
        """Animate the enhanced reactor with rotating layers, orbiting particles and pulsating core"""
        if not hasattr(self, 'reactor_layers'):
            return
            
        center_x, center_y = 100, 100
        t = time.time()
        
        for layer in self.reactor_layers:
            layer['rotation'] += 0.5 * (1 + layer['z'] / 50)
            for ring in layer['rings']:
                self.logo_canvas.itemconfig(ring, start=layer['rotation'])
            for particle in layer['particles']:
                particle['angle'] += particle['speed']
                x = center_x + math.cos(math.radians(particle['angle'])) * particle['radius']
                y = center_y + math.sin(math.radians(particle['angle'])) * particle['radius'] * 0.3
                y += math.sin(t * 2 + particle['angle']) * 5
                self.logo_canvas.coords(
                    particle['id'],
                    x - 2, y - 2, x + 2, y + 2
                )
                opacity = 0.5 + math.sin(math.radians(particle['angle'])) * 0.5
                color = self._adjust_color_brightness('#00bfff', opacity)
                self.logo_canvas.itemconfig(particle['id'], fill=color, outline=color)
        
        # Core pulsing effect
        core_pulse = abs(math.sin(t * 2)) * 0.3 + 0.7
        core_color = self._adjust_color_brightness('#00bfff', core_pulse)
        self.logo_canvas.itemconfig(self.reactor_core['inner'], fill=core_color, outline=core_color)
        
        # Energy rings pulsate and scale for a dynamic effect
        for i, ring in enumerate(self.reactor_core['energy_rings']):
            ring_pulse = abs(math.sin(t * 2 + i * math.pi / 3))
            ring_color = self._adjust_color_brightness('#00bfff', ring_pulse * 0.5)
            self.logo_canvas.itemconfig(ring, outline=ring_color)
            scale = 1 + ring_pulse * 0.1
            self.logo_canvas.coords(
                ring,
                center_x - (25 + i * 5) * scale,
                center_y - (25 + i * 5) * scale,
                center_x + (25 + i * 5) * scale,
                center_y + (25 + i * 5) * scale
            )
        
        self.root.after(20, self._animate_enhanced_reactor)

    def create_action_buttons(self):
        """Create holographic action buttons with enhanced visual effects"""
        # Add a system status display above the buttons
        self.system_status_frame = ttk.Frame(self.actions_frame, style='Sidebar.TFrame')
        self.system_status_frame.pack(fill='x', padx=10, pady=10)
        
        # System status indicator
        status_label = ttk.Label(
            self.system_status_frame,
            text="SYSTEM STATUS",
            style='Neural.TLabel',
            font=('Rajdhani', 9)
        )
        status_label.pack(anchor='w')
        
        # Create circular status indicators
        self.systems_canvas = tk.Canvas(
            self.system_status_frame,
            width=200,
            height=30,
            bg=self.colors['bg_medium'],
            highlightthickness=0
        )
        self.systems_canvas.pack(fill='x', pady=5)
        
        # Create status indicators for different systems
        self.system_indicators = {}
        systems = ['CORE', 'MEMORY', 'NETWORK', 'SENSORS']
        
        for i, system in enumerate(systems):
            x_pos = 20 + i * 50
            # Background circle
            self.systems_canvas.create_oval(
                x_pos - 8, 15 - 8,
                x_pos + 8, 15 + 8,
                fill=self.colors['bg_dark'],
                outline=self.colors['accent_dim'],
                width=1
            )
            # Status indicator
            indicator = self.systems_canvas.create_oval(
                x_pos - 5, 15 - 5,
                x_pos + 5, 15 + 5,
                fill=self.colors['success'],
                outline=''
            )
            # Label
            self.systems_canvas.create_text(
                x_pos, 30,
                text=system,
                fill=self.colors['accent_dim'],
                font=('Rajdhani', 7)
            )
            self.system_indicators[system] = indicator
        
        # Simulate occasional status changes for dynamic effect
        self._animate_system_indicators()
        
        # Action buttons with holographic design
        actions = [
            ("🔍 SCAN", lambda: self._on_send(None, "Scan systems"), "Analyze system status"),
            ("💡 ASSIST", lambda: self._on_send(None, "What can you help me with?"), "Get assistance"),
            ("🔄 RESET", self._clear_chat, "Clear current session"),
            ("⚙️ SETTINGS", self._show_settings, "Configure F.R.E.D. settings")
        ]
        
        for text, command, tooltip in actions:
            btn_frame = ttk.Frame(self.actions_frame, style='Sidebar.TFrame')
            btn_frame.pack(fill='x', padx=10, pady=3)
            
            # Create a canvas for the holographic button effect
            btn_canvas = tk.Canvas(
                btn_frame,
                height=40,
                bg=self.colors['bg_medium'],
                highlightthickness=0
            )
            btn_canvas.pack(fill='x')
            
            # Create button directly on canvas
            btn_rect = btn_canvas.create_rectangle(
                0, 0, 200, 40,
                fill=self.colors['bg_medium'],
                outline=self.colors['accent'],
                width=1,
                tags=f"button_{text}"
            )
            
            # Add decoration lines for tech feel
            btn_canvas.create_line(
                0, 0, 10, 0,
                fill=self.colors['accent_bright'],
                width=2,
                tags=f"button_{text}"
            )
            btn_canvas.create_line(
                0, 0, 0, 10,
                fill=self.colors['accent_bright'],
                width=2,
                tags=f"button_{text}"
            )
            btn_canvas.create_line(
                200, 40, 190, 40,
                fill=self.colors['accent_bright'],
                width=2,
                tags=f"button_{text}"
            )
            btn_canvas.create_line(
                200, 40, 200, 30,
                fill=self.colors['accent_bright'],
                width=2,
                tags=f"button_{text}"
            )
            
            # Add text with glow effect
            glow = btn_canvas.create_text(
                100, 20,
                text=text,
                fill=self.colors['accent_bright'],
                font=('Rajdhani', 11, 'bold'),
                tags=f"button_{text}"
            )
            
            # Add hover effects
            def on_enter(e, canvas=btn_canvas, tags=f"button_{text}"):
                canvas.itemconfig(tags, fill=self.colors['bg_light'])
            
            def on_leave(e, canvas=btn_canvas, tags=f"button_{text}", rect=btn_rect):
                canvas.itemconfig(tags, fill=self.colors['bg_medium'])
                canvas.itemconfig(rect, fill=self.colors['bg_medium'])
            
            def on_click(e, cmd=command, canvas=btn_canvas, tags=f"button_{text}"):
                canvas.itemconfig(tags, fill=self.colors['bg_dark'])
                self.root.after(100, cmd)
            
            btn_canvas.tag_bind(f"button_{text}", "<Enter>", on_enter)
            btn_canvas.tag_bind(f"button_{text}", "<Leave>", on_leave)
            btn_canvas.tag_bind(f"button_{text}", "<Button-1>", on_click)
            
            self._create_tooltip(btn_canvas, tooltip)

    def _create_tooltip(self, widget, text):
        """Create a floating tooltip for widgets"""
        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 20
            
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            
            label = ttk.Label(
                self.tooltip,
                text=text,
                justify='left',
                background='#0d1a2a',
                foreground='#00bfff',
                relief='solid',
                borderwidth=1,
                font=("Rajdhani", 9)
            )
            label.pack()
        
        def leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()
        
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)

    def _on_button_hover(self, event):
        self.send_button.configure(bg='#102030', fg='#00dfff')

    def _on_button_leave(self, event):
        self.send_button.configure(bg='#0d1a2a', fg='#00bfff')

    def _update_metrics(self):
        """Update system metrics with smooth animations"""
        try:
            import psutil
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            
            if hasattr(self, 'cpu_bar') and hasattr(self, 'memory_bar'):
                # Smoothly animate CPU bar
                current_cpu_width = self.cpu_canvas.coords(self.cpu_bar)[2] 
                target_cpu_width = (cpu / 100) * 100
                delta_cpu = (target_cpu_width - current_cpu_width) * 0.2
                self.cpu_canvas.coords(self.cpu_bar, 0, 0, current_cpu_width + delta_cpu, 15)
                
                # Change CPU bar color based on usage
                if cpu > 80:
                    cpu_color = self.colors['highlight']
                elif cpu > 50:
                    cpu_color = self.colors['warning']
                else:
                    cpu_color = self.colors['accent']
                self.cpu_canvas.itemconfig(self.cpu_bar, fill=cpu_color)
                
                # Smoothly animate memory bar
                current_mem_width = self.memory_canvas.coords(self.memory_bar)[2]
                target_mem_width = (memory / 100) * 100
                delta_mem = (target_mem_width - current_mem_width) * 0.2
                self.memory_canvas.coords(self.memory_bar, 0, 0, current_mem_width + delta_mem, 15)
                
                # Change memory bar color based on usage
                if memory > 80:
                    mem_color = self.colors['highlight']
                elif memory > 50:
                    mem_color = self.colors['warning']
                else:
                    mem_color = self.colors['accent']
                self.memory_canvas.itemconfig(self.memory_bar, fill=mem_color)
                
                # Update labels
                self.cpu_label.config(text=f"CPU {cpu:.0f}%")
                self.memory_label.config(text=f"MEM {memory:.0f}%")
        except (ImportError, Exception) as e:
            # Fallback with simulated values if psutil isn't available
            cpu = 25 + 15 * math.sin(time.time() / 5)
            memory = 40 + 10 * math.sin(time.time() / 7)
            
            if hasattr(self, 'cpu_bar') and hasattr(self, 'memory_bar'):
                self.cpu_canvas.coords(self.cpu_bar, 0, 0, cpu, 15)
                self.memory_canvas.coords(self.memory_bar, 0, 0, memory, 15)
                self.cpu_label.config(text=f"CPU {cpu:.0f}%")
                self.memory_label.config(text=f"MEM {memory:.0f}%")
        
        self.root.after(1000, self._update_metrics)
    
    def _pulse_status_circle(self):
        t = time.time() * 2
        pulse = abs(math.sin(t)) * 0.3 + 0.7
        color = self._adjust_color_brightness(self.colors['accent'], pulse)
        if hasattr(self, 'status_circle'):
            self.status_canvas.itemconfig(self.status_circle, fill=color)

    def _adjust_color_brightness(self, color, factor):
        # Convert hex to RGB and adjust brightness
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        r = min(255, max(0, int(r * factor)))
        g = min(255, max(0, int(g * factor)))
        b = min(255, max(0, int(b * factor)))
        return f'#{r:02x}{g:02x}{b:02x}'

    def _setup_layout(self):
        self.container.pack(expand=True, fill='both')
        self.sidebar.pack(side='left', fill='y')
        self.logo_frame.pack(fill='x', pady=(10, 5))
        self.logo_canvas.pack(expand=True)
        
        # Use our updated Title style for the sidebar title
        self.sidebar_title.configure(style='Title.TLabel')
        self.sidebar_title.pack(fill='x', pady=(0, 10), padx=20)
        
        # Create a separator with enhanced pulsing glow effect
        self.separator_canvas = tk.Canvas(
            self.sidebar, 
            height=2, 
            bg=self.colors['bg_medium'],
            highlightthickness=0
        )
        self.separator_canvas.pack(fill='x', padx=10, pady=5)
        self._animate_separator()
        
        self.actions_frame.pack(fill='x', pady=10, side='bottom')
        self.main_frame.pack(side='left', expand=True, fill='both')
        self.header_frame.pack(fill='x', pady=(0, 10))
        self.chat_frame.pack(expand=True, fill='both', pady=(0, 20))
        self.chat_display.pack(expand=True, fill='both')
        
        # Tech overlay pattern
        self.overlay_canvas = tk.Canvas(
            self.chat_frame,
            bg=self.colors['bg_medium'],
            highlightthickness=0,
            width=self.chat_frame.winfo_width(),
            height=20
        )
        self.overlay_canvas.place(x=0, y=0, relwidth=1)
        self._create_tech_pattern()
        
        # Data stream effect and hex grid overlays
        self.stream_canvas.place(relx=1, rely=0, anchor='ne', relheight=1)
        self.hex_canvas.place(relx=1, rely=0, anchor='ne', relwidth=0.2, relheight=1)
        
        # Enhanced input container with holographic design
        self.input_container.pack(fill='x', padx=20, pady=10)
        
        # Create radial background for input field
        self.input_radial_frame = tk.Frame(
            self.input_container,
            bg=self.colors['bg_medium'],
            highlightthickness=0
        )
        self.input_radial_frame.pack(fill='x', expand=True, pady=5)
        
        # Circular 'glow' behind input
        self.input_radial_canvas = tk.Canvas(
            self.input_radial_frame,
            bg=self.colors['bg_medium'],
            highlightthickness=0,
            height=50
        )
        self.input_radial_canvas.pack(fill='x', pady=5)
        self._create_input_radial()
        
        self.input_frame.pack(fill='x', expand=True)
        self.input_border = tk.Canvas(
            self.input_frame, 
            height=2, 
            bg=self.colors['bg_dark'], 
            highlightthickness=0
        )
        self.input_border.pack(fill='x', side='bottom')
        
        # Include voice indicator next to input field for JARVIS-like activation
        self.voice_indicator.pack(side='left', padx=(0, 10))
        self.input_field.pack(side='left', expand=True, fill='x', padx=(0, 10))
        
        # Update send button to match holographic style
        self.send_button.configure(
            bg=self.colors['bg_medium'],
            fg=self.colors['accent_bright'],
            activebackground=self.colors['bg_light'],
            activeforeground='#ffffff',
            text="TRANSMIT"
        )
        self.send_button.pack(side='right')
        
        self.status_frame.pack(fill='x', pady=(10, 0))
        self.status_bar.pack(side='left')
        
        # Bind focus events for input field glow effect
        self.input_field.bind("<FocusIn>", self._start_input_glow)
        self.input_field.bind("<FocusOut>", self._stop_input_glow)
        self.input_field.focus_set()
        
        self._start_animations()
    
    def _create_input_radial(self):
        """Create a radial glow effect behind the input field"""
        width = self.input_radial_canvas.winfo_width() or 400
        height = self.input_radial_canvas.winfo_height()
        center_x = width // 2
        
        # Create circular gradient
        self.input_glow = []
        for i in range(10, 0, -1):
            radius = 25 - i
            alpha = 0.08 - (i * 0.005)
            color = self._adjust_color_brightness(self.colors['accent'], alpha)
            glow = self.input_radial_canvas.create_oval(
                center_x - radius, 25 - radius,
                center_x + radius, 25 + radius,
                fill=color,
                outline='',
                tags='glow'
            )
            self.input_glow.append((glow, radius))
        
        # Input field positioning line
        self.input_radial_canvas.create_line(
            0, 25, width, 25,
            fill=self._adjust_color_brightness(self.colors['accent'], 0.2),
            dash=(3, 5),
            tags='glow'
        )
        
        # Add some tech decorations
        for x in range(0, width, 25):
            if x != center_x:  # Skip the center point
                size = 2
                self.input_radial_canvas.create_rectangle(
                    x - size, 25 - size,
                    x + size, 25 + size,
                    fill=self.colors['accent_dim'],
                    outline='',
                    tags='glow'
                )
    
    def _start_input_glow(self, event):
        """Start the input glow animation when focused"""
        self._input_glow_active = True
        self._animate_input_glow()
    
    def _stop_input_glow(self, event):
        """Stop the input glow animation when focus is lost"""
        self._input_glow_active = False
    
    def _animate_input_glow(self):
        """Animate the input glow with a pulsing effect"""
        if not hasattr(self, '_input_glow_active') or not self._input_glow_active:
            return
        
        t = time.time() * 2
        pulse = abs(math.sin(t)) * 0.5 + 0.5
        
        if hasattr(self, 'input_glow'):
            for glow, radius in self.input_glow:
                new_radius = radius * (1 + pulse * 0.2)
                color = self._adjust_color_brightness(
                    self.colors['accent'], 
                    0.05 + pulse * 0.08
                )
                self.input_radial_canvas.itemconfig(glow, fill=color)
        
        self.root.after(50, self._animate_input_glow)
    
    def _animate_separator(self):
        """Animate the sidebar separator with flowing light"""
        width = self.separator_canvas.winfo_width() or 250
        self.separator_canvas.delete('flow')
        
        # Create flowing light effect
        t = time.time() * 100
        x = (t % width)
        
        # Draw gradient pulse along separator
        for i in range(20):
            pos = (x - i * 4) % width
            alpha = max(0, 0.9 - (i * 0.05))
            color = self._adjust_color_brightness(self.colors['accent'], alpha)
            self.separator_canvas.create_line(
                pos, 0, pos + 10, 0,
                fill=color,
                width=2,
                tags='flow'
            )
        
        self.root.after(50, self._animate_separator)

    def _on_send(self, event=None, preset_message=None):
        message = preset_message if preset_message else self.input_field.get().strip()
        if message:
            self.status_bar.config(text="Processing...")
            self.input_field.configure(state='disabled')
            self.send_button.configure(state='disabled')
            
            self.display_message(message, "user")
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            
            self.input_field.delete(0, tk.END)
            
            threading.Thread(
                target=self._process_message,
                args=(message,),
                daemon=True
            ).start()
    
    def _process_message(self, message):
        try:
            response = self.chat_callback(message)
            self.msg_queue.put(("assistant", response))
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            self.status_bar.config(text="Ready")
        except Exception as e:
            self.msg_queue.put(("error", f"An error occurred: {str(e)}"))
            self.status_bar.config(text="Error occurred")
        finally:
            self.root.after(0, lambda: self.input_field.configure(state='normal'))
            self.root.after(0, lambda: self.send_button.configure(state='normal'))
            self.root.after(0, lambda: self.input_field.focus_set())
    
    def _clear_chat(self, event=None):
        if messagebox.askyesno("Clear Chat", "Are you sure you want to clear the chat?"):
            self.chat_display.delete(1.0, tk.END)
            self.conversation_history = []
            self.status_bar.config(text="Chat cleared")
    
    def display_message(self, message, sender):
        """Display a message with typing animation effect and proper prefixing"""
        self.chat_display.insert(tk.END, "\n\n")
        
        if sender == "user":
            prefix = "Ian: "
        elif sender == "error":
            prefix = "Error: "
        else:
            prefix = "F.R.E.D.: "
        
        self.chat_display.insert(tk.END, prefix, f"prefix_{sender}")
        self.chat_display.tag_config(
            f"prefix_{sender}",
            foreground='#00bfff',
            font=("Rajdhani", 11, "bold")
        )
        
        if sender == "assistant":
            self._start_thinking_animation()
        
        def type_message(msg, index=0):
            if index < len(msg):
                self.chat_display.insert(tk.END, msg[index], f"message_{sender}")
                self.chat_display.see(tk.END)
                self.root.after(10, type_message, msg, index + 1)
            else:
                self.chat_display.insert(tk.END, "\n")
                self.chat_display.see(tk.END)
                if sender == "assistant":
                    self._stop_thinking_animation()
        
        self.chat_display.tag_config(
            f"message_{sender}",
            foreground='#00bfff',
            font=("Rajdhani", 11)
        )
        
        if message is None:
            message = "Empty message"
        
        type_message(message)
    
    def _update_time(self):
        """Update time display with digital clock effect and date"""
        current_time = datetime.now()
        
        # Format date
        date_str = current_time.strftime("%A, %B %d, %Y")
        self.date_label.config(text=date_str)
        
        # Format time with blinking separator for dynamic effect
        time_str = current_time.strftime("%H:%M:%S")
        separator = ":" if time.time() % 1 < 0.5 else " "
        formatted_time = time_str.replace(":", separator)
        self.time_label.config(text=formatted_time)
        
        # Pulse the status circle
        t = time.time() * 2
        pulse = abs(math.sin(t)) * 0.3 + 0.7
        color = self._adjust_color_brightness(self.colors['accent'], pulse)
        if hasattr(self, 'status_circle'):
            self.status_canvas.itemconfig(self.status_circle, fill=color)
        
        self.root.after(500, self._update_time)
    
    def _start_msg_checker(self):
        try:
            while not self.msg_queue.empty():
                sender, message = self.msg_queue.get_nowait()
                self.display_message(message, sender.lower())
        finally:
            self.root.after(100, self._start_msg_checker)
    
    def run(self):
        self.root.attributes('-alpha', 0.98)
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')
        self.root.mainloop()

    def _create_tech_pattern(self):
        """Create a futuristic tech overlay pattern"""
        width = self.overlay_canvas.winfo_width()
        height = self.overlay_canvas.winfo_height()
        for x in range(0, width, 30):
            self.overlay_canvas.create_line(x, 0, x, height, fill='#00bfff', width=1, stipple='gray50')
        for y in range(0, height, 30):
            self.overlay_canvas.create_line(0, y, width, y, fill='#00bfff', width=1, stipple='gray50')

    def _start_thinking_animation(self):
        """Create particle effect animation while F.R.E.D. is 'thinking'"""
        if not self.particle_canvas:
            return
        self.particle_canvas.place(relx=0.5, rely=0.1, anchor='n')
        self.thinking_active = True
        for _ in range(10):
            particle = {
                'x': self.particle_canvas.winfo_width() / 2,
                'y': self.particle_canvas.winfo_height() / 2,
                'dx': (random.random() - 0.5) * 4,
                'dy': (random.random() - 0.5) * 4,
                'size': random.randint(2, 5),
                'id': None
            }
            self.thinking_particles.append(particle)
        self._animate_thinking_particles()

    def _animate_thinking_particles(self):
        """Animate the thinking particles on the canvas"""
        if not hasattr(self, 'thinking_active') or not self.thinking_active:
            return
        self.particle_canvas.delete('particle')
        for particle in self.thinking_particles:
            particle['x'] += particle['dx']
            particle['y'] += particle['dy']
            if particle['x'] < 0 or particle['x'] > self.particle_canvas.winfo_width():
                particle['dx'] *= -1
            if particle['y'] < 0 or particle['y'] > self.particle_canvas.winfo_height():
                particle['dy'] *= -1
            size = particle['size']
            self.particle_canvas.create_oval(
                particle['x'] - size,
                particle['y'] - size,
                particle['x'] + size,
                particle['y'] + size,
                fill='#00bfff',
                outline='',
                tags='particle'
            )
        self.root.after(50, self._animate_thinking_particles)

    def _stop_thinking_animation(self):
        """Stop the thinking animation and clear particles"""
        self.thinking_active = False
        if self.particle_canvas:
            self.particle_canvas.place_forget()
            self.particle_canvas.delete('particle')
        self.thinking_particles.clear()

    def _pulse_voice_indicator(self, level):
        """Pulse the voice indicator based on audio level"""
        if not hasattr(self, 'voice_indicator'):
            return
        scale = 1 + (level * 0.5)
        self.voice_indicator.scale('voice_ring', 10, 10, scale, scale)
        color = self._adjust_color_brightness('#00bfff', 0.5 + level)
        self.voice_indicator.itemconfig('voice_ring', outline=color)

    def _create_neural_network(self):
        """Create a dynamic neural network visualization"""
        self.nodes = []
        self.connections = []
        for i in range(10):
            x = random.randint(20, 130)
            y = random.randint(20, 130)
            node = {
                'x': x,
                'y': y,
                'activation': random.random(),
                'id': self.neural_canvas.create_oval(
                    x-3, y-3, x+3, y+3,
                    fill='#00bfff',
                    outline='#00bfff'
                )
            }
            self.nodes.append(node)
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                if random.random() < 0.3:
                    conn = self.neural_canvas.create_line(
                        self.nodes[i]['x'], self.nodes[i]['y'],
                        self.nodes[j]['x'], self.nodes[j]['y'],
                        fill='#00bfff',
                        width=1,
                        dash=(2, 4)
                    )
                    self.connections.append({
                        'from': i,
                        'to': j,
                        'id': conn,
                        'activity': random.random()
                    })
        self._animate_neural_network()

    def _animate_neural_network(self):
        """Animate neural network nodes and connections with a neon glow effect"""
        if not hasattr(self, 'nodes'):
            return
        for node in self.nodes:
            node['activation'] = min(1.0, max(0.2, node['activation'] + random.uniform(-0.1, 0.1)))
            color = self._adjust_color_brightness('#00bfff', node['activation'])
            self.neural_canvas.itemconfig(node['id'], fill=color, outline=color)
        for conn in self.connections:
            conn['activity'] = min(1.0, max(0.1, conn['activity'] + random.uniform(-0.1, 0.1)))
            color = self._adjust_color_brightness('#00bfff', conn['activity'])
            self.neural_canvas.itemconfig(conn['id'], fill=color)
        self.root.after(100, self._animate_neural_network)

    def _create_env_display(self):
        """Create an environment analysis display with neon bars"""
        self.env_metrics = {
            'processing_load': 0.0,
            'response_time': 0.0,
            'creativity_index': 0.0
        }
        y_pos = 5
        self.env_indicators = {}
        for metric in self.env_metrics:
            label = ttk.Label(
                self.env_frame,
                text=metric.replace('_', ' ').title(),
                foreground='#00bfff',
                background='#0d1a2a',
                font=('Rajdhani', 9)
            )
            label.pack(anchor='w', padx=5)
            bar = self.env_canvas.create_rectangle(
                5, y_pos,
                105, y_pos + 4,
                fill='#00bfff',
                width=0
            )
            self.env_indicators[metric] = bar
            y_pos += 10
        self._update_env_metrics()

    def _update_env_metrics(self):
        """Update environment analysis metrics with smooth transitions"""
        for metric in self.env_metrics:
            target = random.uniform(0.3, 0.9)
            current = self.env_metrics[metric]
            self.env_metrics[metric] += (target - current) * 0.1
            bar = self.env_indicators[metric]
            width = self.env_metrics[metric] * 100
            self.env_canvas.coords(bar, 5, self.env_canvas.coords(bar)[1], 5 + width, self.env_canvas.coords(bar)[3])
            color = self._adjust_color_brightness('#00bfff', 0.5 + self.env_metrics[metric] * 0.5)
            self.env_canvas.itemconfig(bar, fill=color)
        self.root.after(200, self._update_env_metrics)

    def _create_hex_grid(self):
        """Create an animated hexagonal grid background for a futuristic feel"""
        self.hex_cells = []
        size = 20
        for row in range(0, self.chat_frame.winfo_height(), size * 2):
            for col in range(0, 100, size * 2):
                points = self._calculate_hex_points(col, row, size)
                hex_cell = self.hex_canvas.create_polygon(
                    points,
                    fill='',
                    outline='#00bfff',
                    width=1,
                    stipple='gray25'
                )
                self.hex_cells.append({
                    'id': hex_cell,
                    'pulse': random.random() * math.pi
                })
        self._animate_hex_grid()

    def _calculate_hex_points(self, x, y, size):
        points = []
        for i in range(6):
            angle = i * math.pi / 3
            points.extend([x + size * math.cos(angle), y + size * math.sin(angle)])
        return points

    def _animate_hex_grid(self):
        """Animate hexagonal grid lines with a pulsing neon glow"""
        t = time.time()
        for cell in self.hex_cells:
            cell['pulse'] += 0.05
            opacity = abs(math.sin(cell['pulse'])) * 0.5 + 0.2
            color = self._adjust_color_brightness('#00bfff', opacity)
            self.hex_canvas.itemconfig(cell['id'], outline=color)
        self.root.after(50, self._animate_hex_grid)

    def _create_data_stream(self):
        """Create falling data stream effect to simulate high-tech information flow"""
        self.data_particles = []
        for _ in range(20):
            particle = {
                'x': random.randint(5, 25),
                'y': random.randint(0, self.chat_frame.winfo_height()),
                'speed': random.uniform(2, 5),
                'length': random.randint(10, 30),
                'opacity': random.random()
            }
            self.data_particles.append(particle)
        self._animate_data_stream()

    def _animate_data_stream(self):
        """Animate falling data stream particles"""
        self.stream_canvas.delete('stream')
        height = self.chat_frame.winfo_height()
        for particle in self.data_particles:
            particle['y'] += particle['speed']
            if particle['y'] > height:
                particle['y'] = -particle['length']
                particle['x'] = random.randint(5, 25)
                particle['opacity'] = random.random()
            color = self._adjust_color_brightness('#00bfff', particle['opacity'])
            self.stream_canvas.create_line(
                particle['x'], particle['y'],
                particle['x'], particle['y'] + particle['length'],
                fill=color,
                width=1,
                tags='stream'
            )
        self.root.after(50, self._animate_data_stream)

    def _create_frequency_analyzer(self):
        """Create a frequency analyzer visualization with smooth, dynamic bars"""
        self.freq_bars = []
        bar_count = 30
        bar_width = 3
        spacing = 2
        for i in range(bar_count):
            x = i * (bar_width + spacing) + 5
            bar = self.freq_canvas.create_line(
                x, 40, x, 40,
                fill='#00bfff',
                width=bar_width
            )
            self.freq_bars.append({
                'id': bar,
                'value': random.random(),
                'target': random.random()
            })
        self._animate_frequency_analyzer()

    def _animate_frequency_analyzer(self):
        """Animate frequency analyzer bars with neon gradients"""
        for bar in self.freq_bars:
            bar['value'] += (bar['target'] - bar['value']) * 0.2
            if abs(bar['target'] - bar['value']) < 0.01:
                bar['target'] = random.random()
            height = bar['value'] * 35
            coords = self.freq_canvas.coords(bar['id'])
            self.freq_canvas.coords(bar['id'], coords[0], 40, coords[0], 40 - height)
            color = self._adjust_color_brightness('#00bfff', 0.5 + bar['value'] * 0.5)
            self.freq_canvas.itemconfig(bar['id'], fill=color)
        self.root.after(50, self._animate_frequency_analyzer)

    def _show_about(self):
        """Show information about F.R.E.D."""
        about_text = (
            "F.R.E.D. - Funny Rude Educated Droid\n\n"
            "Version 1.0\n"
            "© 2023 Neural Intelligence Division\n\n"
            "F.R.E.D. is an advanced artificial intelligence assistant designed to help with a variety of tasks."
        )
        messagebox.showinfo("About F.R.E.D.", about_text)
    
    def _show_commands(self):
        """Show available F.R.E.D. commands"""
        commands_text = (
            "Available Commands:\n\n"
            "• 'scan systems' - Check system status\n"
            "• 'quick learn [topic]' - Search for information\n"
            "• 'news [topic]' - Get latest news\n"
            "• 'create note [title]' - Create a new note\n"
            "• 'deep research [topic]' - Perform in-depth research\n"
            "• 'goodbye' - Exit conversation"
        )
        messagebox.showinfo("F.R.E.D. Commands", commands_text)

    def _start_animations(self):
        """Start all UI animations"""
        self._animate_enhanced_reactor()
        self._animate_hex_grid()
        self._animate_data_stream()
        self._animate_input_glow()
        self._animate_separator()

    def _animate_system_indicators(self):
        """Animate system status indicators for a dynamic dashboard effect"""
        if hasattr(self, 'system_indicators'):
            # Randomly change one indicator occasionally
            if random.random() < 0.05:  # 5% chance each cycle
                system = random.choice(list(self.system_indicators.keys()))
                status = random.choice([
                    self.colors['success'],  # Online
                    self.colors['warning'],  # Warning
                    self.colors['highlight']  # Alert
                ])
                self.systems_canvas.itemconfig(self.system_indicators[system], fill=status)
        
        # Run this animation less frequently
        self.root.after(1000, self._animate_system_indicators)

    def _show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("F.R.E.D. Settings")
        settings_window.geometry("400x300")
        settings_window.configure(bg=self.colors['bg_medium'])
        
        # Center the window
        settings_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - settings_window.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - settings_window.winfo_height()) // 2
        settings_window.geometry(f"+{x}+{y}")
        
        # Simple settings content
        title_label = ttk.Label(
            settings_window,
            text="F.R.E.D. CONFIGURATION",
            style='Title.TLabel',
            font=('Rajdhani', 16, 'bold')
        )
        title_label.pack(pady=20)
        
        # Settings options would go here
        ttk.Label(
            settings_window,
            text="Settings functionality will be implemented in a future update.",
            style='Neural.TLabel'
        ).pack(pady=40)
        
        # Close button
        close_btn = tk.Button(
            settings_window,
            text="CLOSE",
            font=('Rajdhani', 11, 'bold'),
            bg=self.colors['bg_medium'],
            fg=self.colors['accent_bright'],
            activebackground=self.colors['bg_light'],
            activeforeground='#ffffff',
            relief='flat',
            padx=15,
            pady=5,
            cursor='hand2',
            bd=0
        )
        close_btn.pack(pady=20)
        close_btn.bind("<Button-1>", lambda e: settings_window.destroy())
