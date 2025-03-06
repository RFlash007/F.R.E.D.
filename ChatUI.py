import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox, PhotoImage, Canvas, font, filedialog
import threading
from queue import Queue
from datetime import datetime
import json
import os
import time
import math
import random
import colorsys
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageOps
import sys
import platform
from typing import Dict, List, Tuple, Any, Optional, Union

class ChatUI:
    def __init__(self, chat_callback):
        self.root = tk.Tk()
        self.root.title("F.R.E.D. Neural Interface")
        self.root.geometry("1200x800")
        self.root.configure(bg='#0e0021')
        
        # Load fonts or use fallbacks
        self._load_fonts()
        
        # Enable transparency support with platform-specific handling
        if platform.system() == "Windows":
            # On Windows, we use a different method for transparency
            self.root.attributes('-alpha', 0.92)
        else:
            # Other platforms may support different transparency methods
            try:
                self.root.attributes('-alpha', 0.92)
            except:
                pass
        
        # Set window icon
        try:
            self.root.iconbitmap("assets/fred_icon.ico")
        except:
            pass
        
        # Core system components
        self.arc_reactor = None
        self.arc_reactor_active = False
        self.thinking_indicator = None
        self.holographic_display = None
        self.conversation_panel = None
        
        # Minimal animation states
        self.arc_pulse_active = False
        self.ambient_glow_active = False
        
        # Message queue for thread-safe UI updates
        self.msg_queue = Queue()
        self.chat_callback = chat_callback
        self.conversation_history = []
        
        # System information for display
        self.system_info = self._get_system_info()
        
        # Create custom styles and widgets
        self._create_styles()
        self._create_widgets()
        self._setup_layout()
        self._start_msg_checker()
        self._initialize_arc_reactor()
        
        # Enhanced keyboard shortcuts
        self.root.bind("<Control-c>", lambda e: self._clear_chat())
        self.root.bind("<Control-q>", lambda e: self.root.quit())
        self.root.bind("<F11>", lambda e: self._toggle_fullscreen())
        self.root.bind("<Escape>", lambda e: self._exit_fullscreen())

    def _load_fonts(self):
        """Load fonts and set up font dictionary"""
        available_fonts = {}
        
        # List of preferred fonts for each category
        heading_fonts = ['Orbitron', 'Rajdhani', 'Audiowide', 'Exo', 'Teko', 'Roboto Condensed', 'Arial', 'Helvetica']
        mono_fonts = ['Inconsolata', 'Roboto Mono', 'Source Code Pro', 'Courier New', 'Consolas']
        body_fonts = ['Roboto', 'Oxygen', 'Open Sans', 'Segoe UI', 'Arial', 'Helvetica']
        
        # Get the first available font for each category
        available_fonts['heading'] = self._get_available_font(heading_fonts)
        available_fonts['mono'] = self._get_available_font(mono_fonts)
        available_fonts['body'] = self._get_available_font(body_fonts)
        
        # Add 'main' as an alias for 'body' font for backward compatibility
        available_fonts['main'] = available_fonts['body']
        
        self.fonts = available_fonts
    
    def _get_available_font(self, font_list):
        """Return the first available font from a list of preferences"""
        try:
            available_fonts = list(font.families())
            
            for font_name in font_list:
                if font_name in available_fonts:
                    return font_name
                    
            # Return system default if none of the preferred fonts are available
            return font_list[-1]  # Last font in list is the ultimate fallback
        except:
            return "TkDefaultFont"
    
    def _get_system_info(self) -> Dict[str, str]:
        """Gather system information for display in the UI"""
        try:
            info = {
                "os": platform.system() + " " + platform.release(),
                "processor": platform.processor(),
                "hostname": platform.node(),
                "python": sys.version.split()[0],
                "memory": f"{round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3), 2)} GB" if hasattr(os, 'sysconf') else "Unknown",
            }
        except:
            info = {
                "os": platform.system(),
                "processor": "Unknown",
                "hostname": "Unknown",
                "python": sys.version.split()[0],
                "memory": "Unknown"
            }
        return info
    
    def _toggle_fullscreen(self, event=None):
        """Toggle fullscreen mode"""
        self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen"))
        
    def _exit_fullscreen(self, event=None):
        """Exit fullscreen mode"""
        self.root.attributes("-fullscreen", False)

    def _create_styles(self):
        """Set up chat UI colors and styles"""
        # Create clean, modern color scheme with Stark-inspired accent
        self.colors = {
            'bg_dark': '#0d1117',  # Dark background
            'bg_medium': '#161b22', # Medium background
            'bg_light': '#21262d',  # Light background
            'text_primary': '#f0f6fc', # Primary text
            'text_secondary': '#8b949e', # Secondary text
            'stark_blue': '#0ea5e9', # Stark-inspired blue accent
            'accent': '#1f6feb', # Accent color 
            'accent_dim': '#1a4b91', # Dimmer accent for backgrounds
            'accent_bright': '#58a6ff', # Brighter accent for highlights
            'hologram': '#38bdf8', # Holographic elements
            'warning': '#d29922', # Warning color
            'error': '#f85149'  # Error color
        }
        
        # Configure tag styles for the chat
        self.tag_styles = {
            'user': {
                'font': (self.fonts['body'], 11),
                'foreground': self.colors['text_primary'],
                'background': self.colors['bg_light'],
                'prefix': 'You: ',
                'prefix_style': {'foreground': self.colors['accent_bright']}
            },
            'assistant': {
                'font': (self.fonts['body'], 11),
                'foreground': self.colors['text_primary'],
                'background': None,
                'prefix': 'F.R.E.D.: ',
                'prefix_style': {'foreground': self.colors['stark_blue']}
            }
        }
        style = ttk.Style()
        # Use a modern theme as a base
        style.theme_use('clam')
        
        # Translucent purple color scheme
        self.colors = {
            'bg_dark': '#0e0021',        # Deep purple background
            'bg_medium': '#1a0438',      # Medium purple backdrop
            'bg_light': '#2c0657',       # Lighter purple for highlights
            'accent': '#9d6ad8',         # Primary accent (medium purple)
            'accent_bright': '#c17bff',  # Bright highlights for focus
            'accent_dim': '#4e2d82',     # Deeper purple for inactive elements
            'stark_blue': '#8a2be2',     # Transformed to vivid purple (originally blue)
            'stark_glow': '#b360fb',     # Arc reactor glow (purple)
            'hologram': '#c8a2ff',       # Hologram lavender
            'warning': '#ffac42',        # Warning amber
            'success': '#42ffac',        # Success green
            'error': '#ff4278',          # Error red
            'glow': '#b888ff',           # Glow effect base
            'text_primary': '#f5f0ff',   # Primary text color
            'text_secondary': '#d6c2ff', # Secondary text color
            'grid_line': '#2a1550',      # Grid pattern lines
            'transparent': self.root.cget('bg'),  # Use root background color for "transparency"
        }
        
        # Frame styling with translucent depth
        style.configure('FRED.TFrame', 
                      background=self.colors['bg_dark'])
        
        style.configure('Panel.TFrame',
                      background=self.colors['bg_medium'],
                      relief='flat')
                      
        style.configure('Arc.TFrame',
                      background=self.colors['bg_dark'],
                      borderwidth=0,
                      relief='flat')
        
        style.configure('TSeparator', 
                      background=self.colors['accent'])
        
        # Button styling with holographic look
        style.configure('FRED.TButton',
                      font=(self.fonts['heading'], 11, 'bold'),
                      padding=8,
                      background=self.colors['bg_medium'],
                      foreground=self.colors['accent_bright'],
                      borderwidth=0,
                      focusthickness=0,
                      focuscolor=self.colors['accent'])
                      
        style.map('FRED.TButton',
                background=[('active', self.colors['bg_light'])],
                foreground=[('active', self.colors['stark_blue'])])
        
        # Command button styling
        style.configure('Command.TButton',
                      font=(self.fonts['heading'], 12, 'bold'),
                      padding=10,
                      background=self.colors['accent_dim'],
                      foreground=self.colors['text_primary'],
                      borderwidth=0,
                      focusthickness=0)
                      
        style.map('Command.TButton',
                background=[('active', self.colors['accent'])],
                foreground=[('active', '#ffffff')])
        
        # Entry field with glow
        style.configure('FRED.TEntry',
                      font=(self.fonts['heading'], 12),
                      fieldbackground=self.colors['bg_medium'],
                      foreground=self.colors['accent_bright'],
                      insertcolor=self.colors['stark_blue'],
                      borderwidth=0,
                      relief='flat')
        
        # Label styling
        style.configure('FRED.TLabel',
                      background=self.colors['bg_dark'],
                      foreground=self.colors['accent'],
                      font=(self.fonts['heading'], 11))
        
        style.configure('Title.TLabel',
                      background=self.colors['bg_medium'],
                      foreground=self.colors['accent_bright'],
                      font=(self.fonts['heading'], 22, 'bold'))
        
        style.configure('Subtitle.TLabel',
                      background=self.colors['bg_dark'],
                      foreground=self.colors['stark_blue'],
                      font=(self.fonts['heading'], 14, 'bold'))
                      
        style.configure('Status.TLabel',
                      background=self.colors['bg_dark'],
                      foreground=self.colors['hologram'],
                      font=(self.fonts['heading'], 10, 'bold'))
                      
        style.configure('Data.TLabel',
                      background=self.colors['bg_medium'],
                      foreground=self.colors['text_secondary'],
                      font=(self.fonts['mono'], 10))
                      
        # Tech-inspired progressbar
        style.configure("FRED.Horizontal.TProgressbar",
                      troughcolor=self.colors['bg_dark'],
                      background=self.colors['stark_blue'],
                      thickness=4)
        
    def _create_widgets(self):
        """Create innovative F.R.E.D. interface elements"""
        # Create main container with layered design
        self.container = ttk.Frame(self.root, style='FRED.TFrame')
        
        # Main interface frame
        self.main_frame = ttk.Frame(self.container, padding="0", style='FRED.TFrame')
        
        # Create the left sidebar for Arc Reactor
        self.left_sidebar = ttk.Frame(self.main_frame, style='FRED.TFrame', width=200)
        self.left_sidebar.pack(side='left', fill='y', padx=0, pady=0)
        self.left_sidebar.pack_propagate(False)  # Maintain fixed width
        
        # Create arc reactor container
        self.arc_reactor_frame = ttk.Frame(self.left_sidebar, style='Arc.TFrame')
        self.arc_reactor_frame.pack(side='top', pady=(50, 0))
        
        # Create arc reactor canvas
        self.arc_reactor_canvas = tk.Canvas(
            self.arc_reactor_frame,
            width=180,
            height=180,
            bg=self.colors['bg_dark'],
            highlightthickness=0
        )
        self.arc_reactor_canvas.pack()
        
        # Create system identifier
        self.system_id = ttk.Label(
            self.left_sidebar,
            text="F.R.E.D.",
            style='Subtitle.TLabel',
            anchor='center'
        )
        self.system_id.pack(pady=(10, 5))
        
        self.system_subtitle = ttk.Label(
            self.left_sidebar,
            text="Funny Rude\nEducated Droid",
            foreground=self.colors['text_secondary'],
            background=self.colors['bg_dark'],
            font=(self.fonts['heading'], 8),
            justify='center'
        )
        self.system_subtitle.pack()
        
        # Create status indicator
        self.status_frame = ttk.Frame(self.left_sidebar, style='FRED.TFrame')
        self.status_frame.pack(side='bottom', fill='x', pady=(0, 20))
        
        self.status_bar = ttk.Label(
            self.status_frame,
            text="SYSTEMS NOMINAL",
            style='Status.TLabel',
            anchor='center'
        )
        self.status_bar.pack(fill='x', padx=5, pady=5)
        
        # Create innovative radial menu below arc reactor
        self.radial_menu_frame = ttk.Frame(self.left_sidebar, style='FRED.TFrame')
        self.radial_menu_frame.pack(fill='x', expand=True, padx=10, pady=20)
        
        # Create main content area
        self.content_frame = ttk.Frame(self.main_frame, style='FRED.TFrame')
        self.content_frame.pack(side='left', expand=True, fill='both')
        
        # Create conversation display with dark purplish glass effect
        self.conversation_frame = ttk.Frame(
            self.content_frame, 
            style='Panel.TFrame',
            padding=2
        )
        self.conversation_frame.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Chat display with enhanced styling
        self.chat_display = scrolledtext.ScrolledText(
            self.conversation_frame,
            wrap=tk.WORD,
            width=70,
            height=30,
            font=(self.fonts['mono'], 11),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['stark_blue'],
            relief='flat',
            borderwidth=0,
            padx=20,
            pady=20
        )
        self.chat_display.pack(expand=True, fill='both', padx=2, pady=2)
        
        # Advanced scrollbar styling
        self.chat_display.vbar.configure(
            troughcolor=self.colors['bg_dark'],
            bg=self.colors['accent'],
            activebackground=self.colors['accent_bright'],
            width=8
        )
        
        # Create right visualizer panel (holographic display)
        self.right_panel = ttk.Frame(self.main_frame, style='FRED.TFrame', width=250)
        self.right_panel.pack(side='right', fill='y', padx=5, pady=5)
        self.right_panel.pack_propagate(False)  # Maintain fixed width
        
        # Create holographic visualization area - expanded to fill available space
        self.holo_display_frame = ttk.Frame(self.right_panel, style='Panel.TFrame')
        self.holo_display_frame.pack(fill='both', expand=True, pady=(5, 5))
        
        # System metrics display moved below memory access
        self.metrics_frame = ttk.Frame(self.right_panel, style='Panel.TFrame')
        self.metrics_frame.pack(fill='x', expand=False, pady=(5, 10))
        
        # Title for metrics
        ttk.Label(
            self.metrics_frame,
            text="SYSTEM ANALYTICS",
            style='Status.TLabel',
            anchor='center'
        ).pack(fill='x', pady=5)
        
        # System stats in a clean, minimal layout
        for label, value in [
            ("HOST", self.system_info.get("hostname", "UNKNOWN")),
            ("OS", self.system_info.get("os", "UNKNOWN")),
            ("PROC", self.system_info.get("processor", "UNKNOWN").split()[0]),
            ("MEM", self.system_info.get("memory", "UNKNOWN"))
        ]:
            stat_frame = ttk.Frame(self.metrics_frame, style='Panel.TFrame')
            stat_frame.pack(fill='x', padx=10, pady=2)
            
            ttk.Label(
                stat_frame,
                text=f"{label}:",
                style='Data.TLabel'
            ).pack(side='left', padx=5)
            
            ttk.Label(
                stat_frame,
                text=value[:20] + "..." if len(value) > 20 else value,
                foreground=self.colors['text_primary'],
                background=self.colors['bg_medium'],
                font=(self.fonts['mono'], 9)
            ).pack(side='right', padx=5)
        
        # Command input field - floating design with glow effect
        self.input_container = ttk.Frame(self.content_frame, style='FRED.TFrame')
        self.input_container.pack(fill='x', padx=5, pady=10)
        
        self.input_frame = ttk.Frame(
            self.input_container,
            style='Panel.TFrame',
            padding=(10, 5)
        )
        self.input_frame.pack(fill='x', expand=True)
        
        # Input field with subtle glow
        self.input_field = ttk.Entry(
            self.input_frame,
            font=(self.fonts['heading'], 12),
            style='FRED.TEntry'
        )
        self.input_field.bind("<Return>", self._on_send)
        self.input_field.pack(side='left', expand=True, fill='x', padx=(5, 10))
        
        # Send button with arc reactor-inspired design
        self.send_button = tk.Button(
            self.input_frame,
            text="PROCESS",
            command=self._on_send,
            font=(self.fonts['heading'], 11, 'bold'),
            bg=self.colors['bg_medium'],
            fg=self.colors['stark_blue'],
            activebackground=self.colors['accent_dim'],
            activeforeground=self.colors['stark_glow'],
            relief='flat',
            padx=15,
            pady=5,
            cursor='hand2',
            bd=0
        )
        self.send_button.pack(side='right')
        
        # Create thinking indicator
        self.thinking_indicator_canvas = tk.Canvas(
            self.input_container, 
            height=3,
            bg=self.colors['bg_dark'], 
            highlightthickness=0
        )
        self.thinking_indicator_canvas.pack(fill='x', pady=(5, 0))
        
    def _initialize_arc_reactor(self):
        """Create the arc reactor visualization reminiscent of Stark Industries Mark 7 design."""
        # Ensure canvas is created
        if not hasattr(self, 'arc_reactor_canvas'):
            return
            
        # Get canvas dimensions
        width = 180
        height = 180
        center_x = width / 2
        center_y = height / 2
        
        # Create outer ring with pulsing effect
        outer_radius = 75
        self.arc_reactor_canvas.create_oval(
            center_x - outer_radius, center_y - outer_radius,
            center_x + outer_radius, center_y + outer_radius,
            outline=self.colors['stark_blue'],
            width=2,
            tags="reactor_ring"
        )
        
        # Create middle ring
        middle_radius = 60
        self.arc_reactor_canvas.create_oval(
            center_x - middle_radius, center_y - middle_radius,
            center_x + middle_radius, center_y + middle_radius,
            outline=self.colors['stark_blue'],
            width=1.5,
            tags="reactor_ring"
        )
        
        # Create triangular housing (Mark 7 style)
        triangle_size = 50
        triangle_points = [
            center_x, center_y - triangle_size,  # Top point
            center_x - triangle_size * 0.866, center_y + triangle_size * 0.5,  # Bottom left
            center_x + triangle_size * 0.866, center_y + triangle_size * 0.5,  # Bottom right
        ]
        self.arc_reactor_canvas.create_polygon(
            triangle_points,
            outline=self.colors['stark_glow'],
            fill=self._adjust_color_opacity(self.colors['stark_blue'], 0.3),
            width=2,
            tags="reactor_triangle"
        )
        
        # Add inner triangular detail
        inner_triangle_size = triangle_size * 0.7
        inner_triangle_points = [
            center_x, center_y - inner_triangle_size,  # Top point
            center_x - inner_triangle_size * 0.866, center_y + inner_triangle_size * 0.5,  # Bottom left
            center_x + inner_triangle_size * 0.866, center_y + inner_triangle_size * 0.5,  # Bottom right
        ]
        self.arc_reactor_canvas.create_polygon(
            inner_triangle_points,
            outline=self.colors['stark_glow'],
            fill="",
            width=1.5,
            tags="reactor_triangle_inner"
        )
        
        # Add detail lines in triangular pattern
        for i in range(3):
            angle = math.pi * 2 * i / 3
            length = triangle_size * 0.9
            x1 = center_x
            y1 = center_y
            x2 = center_x + length * math.sin(angle)
            y2 = center_y - length * math.cos(angle)
            
            self.arc_reactor_canvas.create_line(
                x1, y1, x2, y2,
                fill=self.colors['stark_glow'],
                width=1.5,
                tags="reactor_detail"
            )
        
        # Create circular segments in outer ring (like Mark 7 design)
        for i in range(6):
            start_angle = i * 60
            self.arc_reactor_canvas.create_arc(
                center_x - outer_radius * 0.8, center_y - outer_radius * 0.8,
                center_x + outer_radius * 0.8, center_y + outer_radius * 0.8,
                start=start_angle, extent=30,
                outline=self.colors['stark_glow'],
                width=1.5,
                style="arc",
                tags="reactor_segments"
            )
        
        # Create core with pulsing effect
        core_radius = 25
        self.arc_core = self.arc_reactor_canvas.create_oval(
            center_x - core_radius, center_y - core_radius,
            center_x + core_radius, center_y + core_radius,
            fill=self.colors['stark_blue'],
            outline=self.colors['stark_glow'],
            width=2,
            tags="reactor_core"
        )
        
        # Add data points with varying sizes and colors
        self.data_points = []
        self.data_point_speeds = []  # Store individual rotation speeds
        for i in range(10):
            angle = 2 * math.pi * random.random()
            radius = middle_radius * 0.3 + middle_radius * 0.6 * random.random()
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Vary point size based on distance from center
            size = 2 + (radius / middle_radius) * 2
            
            # Vary color based on position
            hue = (angle / (2 * math.pi) + 0.5) % 1.0  # Full color spectrum
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            color = f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}'
            
            point = self.arc_reactor_canvas.create_oval(
                x-size, y-size, x+size, y+size,
                fill=color,
                outline="",
                tags="data_points"
            )
            self.data_points.append(point)
            
            # Assign random rotation speed to each point
            speed = 0.5 + random.random() * 2  # Speed between 0.5 and 2.5
            self.data_point_speeds.append(speed)
        
        # Connection lines between data points with varying opacity
        for i in range(len(self.data_points) - 1):
            coords1 = self.arc_reactor_canvas.coords(self.data_points[i])
            coords2 = self.arc_reactor_canvas.coords(self.data_points[i+1])
            
            x1 = (coords1[0] + coords1[2]) / 2
            y1 = (coords1[1] + coords1[3]) / 2
            
            x2 = (coords2[0] + coords2[2]) / 2
            y2 = (coords2[1] + coords2[3]) / 2
            
            # Calculate distance for opacity
            distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            opacity = 0.2 + 0.8 * (1 - distance / (2 * 60))  # 60 is middle_radius
            color = self._adjust_color_opacity(self.colors['accent_dim'], opacity)
            
            self.arc_reactor_canvas.create_line(
                x1, y1, x2, y2,
                fill=color,
                width=1,
                dash=(3, 3),
                tags="data_connections"
            )
        
        # Create circular glow with dynamic radius
        glow_radius = 85
        self.arc_glow = self.arc_reactor_canvas.create_oval(
            center_x - glow_radius, center_y - glow_radius,
            center_x + glow_radius, center_y + glow_radius,
            outline="",
            fill="",
            tags="reactor_glow"
        )
                
        # Start rotation animation and pulse
        self._start_arc_pulse()
        self._start_globe_rotation()
        
        # Create radial menu buttons (minimalist)
        self._create_radial_menu()
    
    def _adjust_color_opacity(self, color, opacity):
        """Adjust color opacity while maintaining the color"""
        # Convert hex to RGB
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        
        # Adjust opacity
        r = int(r * opacity)
        g = int(g * opacity)
        b = int(b * opacity)
        
        # Convert back to hex
        return f'#{r:02x}{g:02x}{b:02x}'

    def _start_globe_rotation(self):
        """Animate the globe to rotate slowly with variable speeds"""
        self.globe_rotation_active = True
        
        def rotate_globe():
            if not hasattr(self, 'arc_reactor_canvas') or not self.globe_rotation_active:
                return
                
            # Dimensions needed for calculations
            width = 180
            height = 180
            center_x = width / 2
            center_y = height / 2
            
            # Protect against race conditions by checking if items still exist
            try:
                # Rotate meridians at different speeds
                meridian_speeds = [0.5, 1.0, 1.5]  # Different speeds for different meridians
                meridians = self.arc_reactor_canvas.find_withtag("reactor_meridian")
                
                for i, item in enumerate(meridians):
                    try:
                        start = float(self.arc_reactor_canvas.itemcget(item, "start"))
                        speed = meridian_speeds[i % len(meridian_speeds)]
                        new_start = (start + speed) % 360
                        self.arc_reactor_canvas.itemconfig(item, start=new_start)
                    except (ValueError, tk.TclError):
                        pass
                
                # Move data points in circular motion with individual speeds
                valid_points = []
                valid_speeds = []
                
                for point, speed in zip(self.data_points, self.data_point_speeds):
                    try:
                        if not self.arc_reactor_canvas.winfo_exists():
                            return
                            
                        coords = self.arc_reactor_canvas.coords(point)
                        if len(coords) == 4:
                            x = (coords[0] + coords[2]) / 2
                            y = (coords[1] + coords[3]) / 2
                            
                            rel_x = x - center_x
                            rel_y = y - center_y
                            
                            # Calculate new position with individual speed
                            angle = math.radians(speed)
                            new_x = center_x + rel_x * math.cos(angle) - rel_y * math.sin(angle)
                            new_y = center_y + rel_x * math.sin(angle) + rel_y * math.cos(angle)
                            
                            # Update position
                            self.arc_reactor_canvas.coords(
                                point,
                                new_x - 3, new_y - 3, new_x + 3, new_y + 3
                            )
                            valid_points.append(point)
                            valid_speeds.append(speed)
                    except (tk.TclError, IndexError):
                        continue
                
                # Update data points and speeds lists
                self.data_points = valid_points
                self.data_point_speeds = valid_speeds
                
                # Update connection lines with dynamic opacity
                try:
                    self.arc_reactor_canvas.delete("data_connections")
                    for i in range(len(self.data_points) - 1):
                        try:
                            coords1 = self.arc_reactor_canvas.coords(self.data_points[i])
                            coords2 = self.arc_reactor_canvas.coords(self.data_points[i+1])
                            
                            if len(coords1) == 4 and len(coords2) == 4:
                                x1 = (coords1[0] + coords1[2]) / 2
                                y1 = (coords1[1] + coords1[3]) / 2
                                
                                x2 = (coords2[0] + coords2[2]) / 2
                                y2 = (coords2[1] + coords2[3]) / 2
                                
                                # Calculate distance for opacity
                                distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                                opacity = 0.2 + 0.8 * (1 - distance / (2 * 60))  # 60 is middle_radius
                                color = self._adjust_color_opacity(self.colors['accent_dim'], opacity)
                                
                                self.arc_reactor_canvas.create_line(
                                    x1, y1, x2, y2,
                                    fill=color,
                                    width=1,
                                    dash=(3, 3),
                                    tags="data_connections"
                                )
                        except (tk.TclError, IndexError):
                            continue
                except Exception as e:
                    print(f"Error updating connections: {e}")
                
                # Continue rotation with error handling
                try:
                    self.root.after(50, rotate_globe)
                except Exception as e:
                    print(f"Error in globe rotation: {e}")
                    self.root.after(1000, self._start_globe_rotation)
            except Exception as e:
                print(f"Globe rotation error: {e}")
                self.root.after(1000, self._start_globe_rotation)
        
        # Start rotation with error handling
        try:
            rotate_globe()
        except Exception as e:
            print(f"Failed to start globe rotation: {e}")
    
    def _create_holographic_display(self):
        """Create memory access visualization in the right panel"""
        # Display title with better styling
        memory_title = ttk.Label(
            self.holo_display_frame,
            text="MEMORY ACCESS",
            style='Status.TLabel',
            anchor='center'
        )
        memory_title.pack(fill='x', pady=(5, 5))
        
        # Create tabs for different memory types
        self.memory_notebook = ttk.Notebook(self.holo_display_frame)
        self.memory_notebook.pack(fill='both', expand=True, padx=5, pady=2)
        
        # Create the memory notebook tabs for different memory types
        self.semantic_frame = ttk.Frame(self.memory_notebook, style='Panel.TFrame')
        self.episodic_frame = ttk.Frame(self.memory_notebook, style='Panel.TFrame')
        self.dreaming_frame = ttk.Frame(self.memory_notebook, style='Panel.TFrame')
        
        # Add tabs to notebook
        self.memory_notebook.add(self.semantic_frame, text="Semantic")
        self.memory_notebook.add(self.episodic_frame, text="Episodic")
        self.memory_notebook.add(self.dreaming_frame, text="Dreams")
        
        # Configure notebook tab styling
        style = ttk.Style()
        style.map("TNotebook.Tab",
                background=[("selected", self.colors['accent_dim'])],
                foreground=[("selected", self.colors['text_primary'])])
        
        # Create memory panels with search and scroll widgets
        self.semantic_scroll = self._initialize_memory_panel(self.semantic_frame, 'semantic')
        self.episodic_scroll = self._initialize_memory_panel(self.episodic_frame, 'episodic')
        self.dreaming_scroll = self._initialize_memory_panel(self.dreaming_frame, 'dreaming')
        
        # Now load memories after all panels are initialized
        self._load_memories()
    
    def _initialize_memory_panel(self, parent_frame, memory_type):
        """Initialize a memory panel with search and return the scroll widget"""
        # Create search frame at the top
        search_frame = tk.Frame(parent_frame, bg=self.colors['bg_medium'])
        search_frame.pack(fill='x', padx=5, pady=5)
        
        # Search entry with placeholder
        search_entry = tk.Entry(
            search_frame,
            font=(self.fonts['mono'], 10),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['stark_blue'],
            relief='flat',
            bd=1
        )
        search_entry.pack(side='left', fill='x', expand=True, padx=5)
        search_entry.insert(0, "Search memories...")
        search_entry.config(fg=self.colors['text_secondary'])
        
        # Clear placeholder on focus
        def _on_entry_focus_in(event):
            if search_entry.get() == "Search memories...":
                search_entry.delete(0, tk.END)
                search_entry.config(fg=self.colors['text_primary'])
                
        # Restore placeholder on focus out if empty
        def _on_entry_focus_out(event):
            if not search_entry.get():
                search_entry.insert(0, "Search memories...")
                search_entry.config(fg=self.colors['text_secondary'])
                
        search_entry.bind("<FocusIn>", _on_entry_focus_in)
        search_entry.bind("<FocusOut>", _on_entry_focus_out)
        
        # Create scrollable text widget for memories
        memory_scroll = scrolledtext.ScrolledText(
            parent_frame,
            wrap=tk.WORD,
            bg=self.colors['bg_medium'],
            fg=self.colors['text_primary'],
            height=15,
            relief='flat',
            padx=10,
            pady=10,
            font=(self.fonts['mono'], 10)
        )
        memory_scroll.pack(fill='both', expand=True, padx=3, pady=3)
        
        # Configure text tags
        memory_scroll.tag_config("memory_header", 
            foreground=self.colors['hologram'],
            font=(self.fonts['heading'], 11, 'bold'))
        memory_scroll.tag_config("memory_category", 
            foreground=self.colors['accent_bright'],
            font=(self.fonts['heading'], 11))
        memory_scroll.tag_config("memory_timestamp", 
            foreground=self.colors['accent'],
            font=(self.fonts['mono'], 10))
        memory_scroll.tag_config("memory_tags", 
            foreground=self.colors['accent_bright'],
            font=(self.fonts['mono'], 9))
        memory_scroll.tag_config("memory_content", 
            foreground=self.colors['text_primary'],
            font=(self.fonts['mono'], 10))
        memory_scroll.tag_config("memory_separator", 
            foreground=self.colors['grid_line'],
            font=(self.fonts['mono'], 8))
        memory_scroll.tag_config("empty_message", 
            foreground=self.colors['text_secondary'],
            font=(self.fonts['heading'], 11))
            
        # Create search button with hover effect
        search_button = tk.Button(
            search_frame,
            text="Search",
            font=(self.fonts['heading'], 9),
            bg=self.colors['accent_dim'],
            fg=self.colors['text_primary'],
            relief='flat',
            padx=8,
            pady=2,
            command=lambda: self._filter_memories(memory_scroll, memory_type, search_entry.get())
        )
        search_button.pack(side='right', padx=5)
        
        # Add hover effect
        def _on_enter(e):
            search_button.config(bg=self.colors['accent'])
            
        def _on_leave(e):
            search_button.config(bg=self.colors['accent_dim'])
            
        search_button.bind("<Enter>", _on_enter)
        search_button.bind("<Leave>", _on_leave)
        
        # Bind Enter key to search
        search_entry.bind("<Return>", lambda e: self._filter_memories(memory_scroll, memory_type, search_entry.get()))
        
        return memory_scroll
    
    def _load_memories(self):
        """Load memories from JSON files"""
        try:
            # Import the memory format utility
            try:
                import memory_format_utils
                
                # Ensure memory files are in the correct format
                memory_format_utils.ensure_all_memory_files()
            except ImportError:
                # Fallback implementation for memory format checking
                print("Warning: memory_format_utils module not found. Using fallback implementation.")
                self._ensure_memory_files_fallback()
            
            # Initialize memories dictionary
            self.memories = {
                'semantic': [],
                'episodic': [],
                'dreaming': []
            }
            
            # Also initialize memory backups
            self.memory_backups = {
                'semantic': [],
                'episodic': [],
                'dreaming': []
            }
            
            # Load semantic memories
            if os.path.exists('Semantic.json'):
                try:
                    with open('Semantic.json', 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():  # Skip empty lines
                                try:
                                    memory = json.loads(line)
                                    self.memories['semantic'].append(memory)
                                except json.JSONDecodeError as e:
                                    print(f"Error parsing semantic memory: {e}")
                except Exception as e:
                    print(f"Error loading semantic memories: {e}")
            
            # Load episodic memories
            if os.path.exists('Episodic.json'):
                try:
                    with open('Episodic.json', 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():  # Skip empty lines
                                try:
                                    memory = json.loads(line)
                                    self.memories['episodic'].append(memory)
                                except json.JSONDecodeError as e:
                                    print(f"Error parsing episodic memory: {e}")
                except Exception as e:
                    print(f"Error loading episodic memories: {e}")
            
            # Load dreams
            if os.path.exists('Dreaming.json'):
                try:
                    with open('Dreaming.json', 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():  # Skip empty lines
                                try:
                                    memory = json.loads(line)
                                    self.memories['dreaming'].append(memory)
                                except json.JSONDecodeError as e:
                                    print(f"Error parsing dream memory: {e}")
                except Exception as e:
                    print(f"Error loading dream memories: {e}")
            
            # Update backups with loaded memories
            for memory_type in self.memories:
                self.memory_backups[memory_type] = self.memories[memory_type].copy()
                
            # Update memory stats
            self._update_memory_stats()
            
            # Populate memory panels with loaded memories - using lambda that matches the edit_callback signature
            self._populate_memory_panel(self.semantic_scroll, 'semantic', 
                lambda idx: self._edit_memory('semantic', idx))
                
            self._populate_memory_panel(self.episodic_scroll, 'episodic', 
                lambda idx: self._edit_memory('episodic', idx))
                
            self._populate_memory_panel(self.dreaming_scroll, 'dreaming', 
                lambda idx: self._edit_memory('dreaming', idx))
            
            # Add memory control buttons
            self._add_memory_control_buttons()
                
        except Exception as e:
            print(f"Error in _load_memories: {e}")
    
    def _ensure_memory_files_fallback(self):
        """Fallback implementation of memory file format checking"""
        import json
        import os
        
        memory_files = {
            "semantic": "Semantic.json",
            "episodic": "Episodic.json",
            "dreaming": "Dreaming.json"
        }
        
        for memory_type, file_path in memory_files.items():
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found. Will be created when needed.")
                continue
                
            # Skip empty files
            if os.path.getsize(file_path) == 0:
                continue
                
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                continue
                
            # Try parsing as a JSON array
            try:
                data = json.loads(content)
                
                # If it's not a list, we can't do much
                if not isinstance(data, list):
                    print(f"Warning: Expected a JSON array in {file_path}. Format might be incorrect.")
                    continue
                
                # Check if it's in JSONL format or needs conversion
                needs_conversion = False
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        
                    # Try parsing each line as JSON
                    for line in lines:
                        json.loads(line)
                except json.JSONDecodeError:
                    needs_conversion = True
                
                # Convert to JSONL format if needed
                if needs_conversion:
                    # Create backup
                    backup_file = f"{file_path}.bak"
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"Created backup at {backup_file}")
                    
                    # Convert to JSONL format
                    with open(file_path, 'w', encoding='utf-8') as f:
                        for item in data:
                            f.write(json.dumps(item) + '\n')
                    
                    print(f"Converted {file_path} to JSONL format ({len(data)} items).")
            
            except json.JSONDecodeError as e:
                print(f"Error parsing {file_path}: {e}")
                print(f"The file is neither valid JSON nor JSONL format.")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    def _validate_memory_structure(self, memory, memory_type):
        """Validate memory structure based on memory type"""
        try:
            if not isinstance(memory, dict):
                return False
                
            if memory_type == 'semantic':
                # Semantic requires category and content
                if 'category' not in memory or 'content' not in memory:
                    return False
                    
                # Check field types
                if not isinstance(memory['category'], str) or not isinstance(memory['content'], str):
                    return False
                        
                return True
                
            elif memory_type == 'dreaming':
                # Dreams require insight_type and content
                if 'insight_type' not in memory or 'content' not in memory:
                    # For backward compatibility, check for category/about field
                    if ('category' not in memory and 'about' not in memory) or 'content' not in memory:
                        return False
                    
                # Check field types
                if 'insight_type' in memory and not isinstance(memory['insight_type'], str):
                    return False
                if 'category' in memory and not isinstance(memory['category'], str):
                    return False
                if 'about' in memory and not isinstance(memory['about'], str):
                    return False
                if not isinstance(memory['content'], str):
                    return False
                
                # Source field is optional, but if present, validate it
                if 'source' in memory and not isinstance(memory['source'], str):
                    return False
                
                # If adding a new dream without source, add default source
                if 'source' not in memory:
                    memory['source'] = 'unknown'
                        
                return True
                
            elif memory_type == 'episodic':
                # Check required fields for episodic memory
                required_fields = [
                    'memory_timestamp', 'context_tags', 'conversation_summary',
                    'what_worked', 'what_to_avoid', 'what_you_learned'
                ]
                
                for field in required_fields:
                    if field not in memory:
                        return False
                        
                # Check context_tags is a list
                if not isinstance(memory['context_tags'], list):
                    return False
                
                # Check string fields
                string_fields = ['memory_timestamp', 'conversation_summary', 
                                'what_worked', 'what_to_avoid', 'what_you_learned']
                for field in string_fields:
                    if not isinstance(memory[field], str):
                        return False
                        
                return True
                
            return False
            
        except Exception as e:
            print(f"Error validating memory structure: {e}")
            return False

    def _update_memory_stats(self):
        """Update memory stats in status bar"""
        try:
            stats_text = f"Semantic: {len(self.memories['semantic'])} | "
            stats_text += f"Episodic: {len(self.memories['episodic'])} | "
            stats_text += f"Dreams: {len(self.memories['dreaming'])}      "  # Added extra padding spaces to ensure visibility
            self.status_bar.config(text=stats_text)
        except Exception as e:
            print(f"Error updating memory stats: {e}")
            self.status_bar.config(text="Error updating memory statistics")

    def _save_memories(self, memory_type):
        """Save memories to JSON file in JSONL format (one JSON object per line)"""
        try:
            filename = f"{memory_type.capitalize()}.json"
            
            # Create backup first
            if os.path.exists(filename):
                backup_file = f"{filename}.bak"
                try:
                    with open(filename, 'r', encoding='utf-8') as src:
                        with open(backup_file, 'w', encoding='utf-8') as dst:
                            dst.write(src.read())
                except Exception as e:
                    print(f"Error creating backup of {filename}: {e}")
            
            # Save memories in JSONL format (one JSON object per line)
            with open(filename, 'w', encoding='utf-8') as f:
                for memory in self.memories.get(memory_type, []):
                    f.write(json.dumps(memory) + '\n')
                    
            # Update backup copy
            self.memory_backups[memory_type] = self.memories[memory_type].copy()
            
            # Update status
            self.status_bar.config(text=f"{memory_type.capitalize()} memories saved")
            self.root.after(3000, self._update_memory_stats)
            
        except Exception as e:
            print(f"Error saving {memory_type} memories: {e}")
            self.status_bar.config(text=f"Error saving {memory_type} memories: {str(e)}")
            self.root.after(3000, self._update_memory_stats)

    def _restore_from_backup(self, memory_type):
        """Restore memories from backup file"""
        try:
            if memory_type == 'semantic' and hasattr(self, 'semantic_scroll'):
                # Restore semantic memories
                self._restore_memory_file('Semantic.json', 
                    self.semantic_scroll, 
                    'semantic', 
                    lambda idx: self._edit_memory('semantic', idx)
                )
            elif memory_type == 'episodic' and hasattr(self, 'episodic_scroll'):
                # Restore episodic memories
                self._restore_memory_file('Episodic.json',
                    self.episodic_scroll, 
                    'episodic', 
                    lambda idx: self._edit_memory('episodic', idx)
                )
            elif memory_type == 'dreaming' and hasattr(self, 'dreaming_scroll'):
                # Restore dream memories
                self._restore_memory_file('Dreaming.json',
                    self.dreaming_scroll, 
                    'dreaming', 
                    lambda idx: self._edit_memory('dreaming', idx)
                )
            else:
                print(f"Invalid memory type: {memory_type}")
                self.status_bar.config(text=f"Error: Invalid memory type: {memory_type}")
                self.root.after(3000, self._update_memory_stats)
        except Exception as e:
            print(f"Error restoring {memory_type} memories: {e}")
            self.status_bar.config(text=f"Error restoring memories: {str(e)}")
            self.root.after(3000, self._update_memory_stats)

    def _export_memories(self, memory_type):
        """Export memories to a JSON file"""
        try:
            if memory_type not in self.memories or not self.memories[memory_type]:
                messagebox.showinfo("Export", f"No {memory_type} memories to export")
                return
                
            # Ask user for export format
            export_format = messagebox.askyesno(
                "Export Format", 
                "Would you like to export as a JSON array?\n\n"
                "Yes: Export as a single JSON array (better for sharing)\n"
                "No: Export as JSONL format (one JSON object per line, better for importing)"
            )
                
            # Ask user for export location
            file_types = [("JSON files", "*.json"), ("All files", "*.*")]
            export_file = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=file_types,
                title=f"Export {memory_type.capitalize()} Memories",
                initialfile=f"{memory_type}_export.json"
            )
            
            if not export_file:
                return  # User cancelled
                
            # Export memories
            with open(export_file, 'w', encoding='utf-8') as f:
                if export_format:  # JSON array
                    json.dump(self.memories[memory_type], f, indent=2, ensure_ascii=False)
                else:  # JSONL format
                    for memory in self.memories[memory_type]:
                        f.write(json.dumps(memory) + '\n')
            
            messagebox.showinfo("Export Complete", 
                f"Successfully exported {len(self.memories[memory_type])} {memory_type} memories to {export_file}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export memories: {str(e)}")

    def _create_memory_panel(self, parent_frame, memory_type, edit_callback):
        """Create a memory panel with enhanced search and filtering"""
        # Create search frame at the top
        search_frame = tk.Frame(parent_frame, bg=self.colors['bg_medium'])
        search_frame.pack(fill='x', padx=5, pady=5)
        
        # Search entry with placeholder
        search_entry = tk.Entry(
            search_frame,
            font=(self.fonts['mono'], 10),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['stark_blue'],
            relief='flat',
            bd=1
        )
        search_entry.pack(side='left', fill='x', expand=True, padx=5)
        search_entry.insert(0, "Search memories...")
        search_entry.config(fg=self.colors['text_secondary'])
        
        # Clear placeholder on focus
        def _on_entry_focus_in(event):
            if search_entry.get() == "Search memories...":
                search_entry.delete(0, tk.END)
                search_entry.config(fg=self.colors['text_primary'])
                
        # Restore placeholder on focus out if empty
        def _on_entry_focus_out(event):
            if not search_entry.get():
                search_entry.insert(0, "Search memories...")
                search_entry.config(fg=self.colors['text_secondary'])
                
        search_entry.bind("<FocusIn>", _on_entry_focus_in)
        search_entry.bind("<FocusOut>", _on_entry_focus_out)
        
        # Search button with hover effect
        search_button = tk.Button(
            search_frame,
            text="Search",
            font=(self.fonts['heading'], 9),
            bg=self.colors['accent_dim'],
            fg=self.colors['text_primary'],
            relief='flat',
            padx=8,
            pady=2,
            command=lambda: self._filter_memories(memory_scroll, memory_type, search_entry.get())
        )
        search_button.pack(side='right', padx=5)
        
        # Add hover effect
        def _on_enter(e):
            search_button.config(bg=self.colors['accent'])
            
        def _on_leave(e):
            search_button.config(bg=self.colors['accent_dim'])
            
        search_button.bind("<Enter>", _on_enter)
        search_button.bind("<Leave>", _on_leave)
        
        # Bind Enter key to search
        search_entry.bind("<Return>", lambda e: self._filter_memories(memory_scroll, memory_type, search_entry.get()))
        
        # Create scrollable text widget for memories with enhanced styling
        memory_scroll = scrolledtext.ScrolledText(
            parent_frame,
            wrap=tk.WORD,
            bg=self.colors['bg_medium'],
            fg=self.colors['text_primary'],
            height=15,
            relief='flat',
            padx=10,
            pady=10,
            font=(self.fonts['mono'], 10)
        )
        memory_scroll.pack(fill='both', expand=True, padx=3, pady=3)
        
        # Configure text tags with enhanced styling
        memory_scroll.tag_config("memory_header", 
            foreground=self.colors['hologram'],
            font=(self.fonts['heading'], 11, 'bold'))
        memory_scroll.tag_config("memory_category", 
            foreground=self.colors['accent_bright'],
            font=(self.fonts['heading'], 11))
        memory_scroll.tag_config("memory_timestamp", 
            foreground=self.colors['accent'],
            font=(self.fonts['mono'], 10))
        memory_scroll.tag_config("memory_tags", 
            foreground=self.colors['accent_bright'],
            font=(self.fonts['mono'], 9))
        memory_scroll.tag_config("memory_content", 
            foreground=self.colors['text_primary'],
            font=(self.fonts['mono'], 10))
        memory_scroll.tag_config("memory_separator", 
            foreground=self.colors['grid_line'],
            font=(self.fonts['mono'], 8))
        memory_scroll.tag_config("empty_message", 
            foreground=self.colors['text_secondary'],
            font=(self.fonts['heading'], 11))
        
        # Populate memories based on type
        self._populate_memory_panel(memory_scroll, memory_type, edit_callback)
    
    def _populate_memory_panel(self, scroll_widget, memory_type, edit_callback, filter_text=None):
        """Populate the memory panel with memory items, filtered if needed"""
        # Clear existing content
        scroll_widget.config(state=tk.NORMAL)
        scroll_widget.delete(1.0, tk.END)
        
        memories = self.memories.get(memory_type, [])
        
        # Filter memories if filter_text is provided
        if filter_text and filter_text != "Search memories...":
            filter_text = filter_text.lower()
            filtered_memories = []
            
            for mem in memories:
                # Create search text based on memory type
                if memory_type == 'semantic':
                    search_text = f"{mem.get('category', '')} {mem.get('content', '')}".lower()
                elif memory_type == 'episodic':
                    tags = ' '.join(mem.get('context_tags', []))
                    search_text = f"{mem.get('memory_timestamp', '')} {tags} {mem.get('conversation_summary', '')}".lower()
                else:  # dreaming
                    # Check for insight_type with fallback to category/about
                    header = mem.get('insight_type', mem.get('category', mem.get('about', 'Unknown')))
                    search_text = f"{header} {mem.get('content', '')}".lower()
                
                # Add memory if it matches search
                if filter_text in search_text:
                    filtered_memories.append(mem)
            
            memories = filtered_memories
        
        # Display message if no memories
        if not memories:
            if filter_text and filter_text != "Search memories...":
                scroll_widget.insert(tk.END, f"No {memory_type} memories match your search.\n\n", "empty_message")
            else:
                scroll_widget.insert(tk.END, f"No {memory_type} memories found.\n\n", "empty_message")
            scroll_widget.insert(tk.END, f"Create a new memory by clicking the 'Add {memory_type.capitalize()} Memory' button below.", "empty_message")
            return
        
        # Add memories with edit buttons
        for i, memory in enumerate(memories):
            if memory_type == 'semantic':
                # Display category in accent color
                category = memory.get('category', 'Unknown')
                scroll_widget.insert(tk.END, f"Category: ", "memory_header")
                scroll_widget.insert(tk.END, f"{category}\n", "memory_category")
                
                # Display content in primary text color
                content = memory.get('content', 'No content')
                scroll_widget.insert(tk.END, f"{content}\n", "memory_content")
                
            elif memory_type == 'episodic':
                # Display timestamp in accent color
                timestamp = memory.get('memory_timestamp', 'Unknown')
                scroll_widget.insert(tk.END, f"Time: ", "memory_header")
                scroll_widget.insert(tk.END, f"{timestamp}\n", "memory_timestamp")
                
                # Display tags
                tags = memory.get('context_tags', [])
                if tags:
                    scroll_widget.insert(tk.END, f"Tags: ", "memory_header")
                    scroll_widget.insert(tk.END, f"{', '.join(tags)}\n", "memory_tags")
                
                # Display summary
                summary = memory.get('conversation_summary', 'No summary')
                scroll_widget.insert(tk.END, f"{summary}\n", "memory_content")
                
            else:  # dreaming
                # Display insight_type in accent color
                insight_type = memory.get('insight_type', 'Unknown')
                scroll_widget.insert(tk.END, f"Insight Type: ", "memory_header")
                scroll_widget.insert(tk.END, f"{insight_type}\n", "memory_category")
                
                # Display source if available
                source = memory.get('source', 'unknown')
                if source != 'unknown':
                    scroll_widget.insert(tk.END, f"Source: ", "memory_header")
                    scroll_widget.insert(tk.END, f"{source.capitalize()}\n", "memory_tags")
                
                # Display content
                content = memory.get('content', 'No content')
                scroll_widget.insert(tk.END, f"{content}\n", "memory_content")
            
            # Create edit button frame
            button_frame = tk.Frame(scroll_widget, bg=self.colors['bg_medium'])
            
            # Add edit button - fixed to pass only the index to the callback
            edit_button = tk.Button(
                button_frame,
                text=f"Edit",
                font=(self.fonts['heading'], 8),
                bg=self.colors['accent_dim'],
                fg=self.colors['text_primary'],
                padx=5,
                pady=2,
                relief='flat',
                command=lambda idx=i: edit_callback(idx)
            )
            edit_button.pack(side='left', padx=2)
            
            # Add delete button
            delete_button = tk.Button(
                button_frame,
                text=f"Delete",
                font=(self.fonts['heading'], 8),
                bg=self.colors['error'],
                fg=self.colors['text_primary'],
                padx=5,
                pady=2,
                relief='flat',
                command=lambda idx=i: self._delete_memory(memory_type, idx, scroll_widget, edit_callback)
            )
            delete_button.pack(side='left', padx=2)
            
            # Insert button frame
            scroll_widget.window_create(tk.END, window=button_frame)
            scroll_widget.insert(tk.END, "\n\n", "memory_separator")
        
        scroll_widget.config(state=tk.DISABLED)
    
    def _filter_memories(self, scroll_widget, memory_type, filter_text):
        """Filter memories based on search text"""
        self._populate_memory_panel(scroll_widget, memory_type, 
            lambda idx: self._edit_memory(memory_type, idx), filter_text)
            
    def _edit_memory(self, memory_type, index):
        """Edit a memory"""
        try:
            if memory_type in self.memories and index < len(self.memories[memory_type]):
                memory = self.memories[memory_type][index]
                # Create a dialog to edit the memory
                print(f"Opening edit dialog for {memory_type} memory at index {index}")
                EditMemoryDialog(self.root, memory, 
                    lambda updated: self._update_memory(memory_type, index, updated))
            else:
                print(f"Invalid memory: {memory_type} at index {index}")
                self.status_bar.config(text=f"Error: Invalid memory reference")
                self.root.after(3000, self._update_memory_stats)
        except Exception as e:
            print(f"Error opening memory editor: {e}")
            self.status_bar.config(text=f"Error opening memory editor: {str(e)}")
            self.root.after(3000, self._update_memory_stats)
    
    def _delete_memory(self, memory_type, index, scroll_widget, edit_callback):
        """Delete a memory after confirmation"""
        if memory_type not in self.memories or index >= len(self.memories[memory_type]):
            return
            
        memory = self.memories[memory_type][index]
        
        # Format memory preview for confirmation
        preview = ""
        if memory_type == 'semantic':
            preview = memory.get('category', 'Unknown')
        elif memory_type == 'episodic':
            preview = memory.get('memory_timestamp', 'Unknown')
        else:  # dreaming
            # Check for insight_type with fallback to category/about
            preview = memory.get('insight_type', memory.get('category', memory.get('about', 'Unknown')))
            
        # Show confirmation dialog
        if messagebox.askyesno("Confirm Delete", 
            f"Are you sure you want to delete this {memory_type} memory?\n\n{preview}"):
            # Remove memory
            del self.memories[memory_type][index]
            
            # Save changes
            self._save_memories(memory_type)
            
            # Repopulate panel
            self._populate_memory_panel(scroll_widget, memory_type, edit_callback)
    
    def _add_memory_control_buttons(self):
        """Add control buttons for each memory panel"""
        
        # Semantic memory controls
        semantic_controls = tk.Frame(self.semantic_frame, bg=self.colors['bg_medium'])
        semantic_controls.pack(fill='x', padx=5, pady=5)
        
        # Add semantic memory button
        add_semantic_btn = tk.Button(
            semantic_controls,
            text="Add Semantic Memory",
            font=(self.fonts['heading'], 10),
            bg=self.colors['accent_dim'],
            fg=self.colors['text_primary'],
            relief='flat',
            padx=10,
            pady=5,
            command=lambda: self._create_new_memory('semantic')
        )
        add_semantic_btn.pack(side='left', padx=5)
        
        # Import semantic memories button
        import_semantic_btn = tk.Button(
            semantic_controls,
            text="Import",
            font=(self.fonts['heading'], 10),
            bg=self.colors['bg_light'],
            fg=self.colors['text_primary'],
            relief='flat',
            padx=10,
            pady=5,
            command=lambda: self._import_memories('semantic')
        )
        import_semantic_btn.pack(side='left', padx=5)
        
        # Export semantic memories button
        export_semantic_btn = tk.Button(
            semantic_controls,
            text="Export",
            font=(self.fonts['heading'], 10),
            bg=self.colors['bg_light'],
            fg=self.colors['text_primary'],
            relief='flat',
            padx=10,
            pady=5,
            command=lambda: self._export_memories('semantic')
        )
        export_semantic_btn.pack(side='left', padx=5)
        
        # Episodic memory controls
        episodic_controls = tk.Frame(self.episodic_frame, bg=self.colors['bg_medium'])
        episodic_controls.pack(fill='x', padx=5, pady=5)
        
        # Add episodic memory button
        add_episodic_btn = tk.Button(
            episodic_controls,
            text="Add Episodic Memory",
            font=(self.fonts['heading'], 10),
            bg=self.colors['accent_dim'],
            fg=self.colors['text_primary'],
            relief='flat',
            padx=10,
            pady=5,
            command=lambda: self._create_new_memory('episodic')
        )
        add_episodic_btn.pack(side='left', padx=5)
        
        # Import episodic memories button
        import_episodic_btn = tk.Button(
            episodic_controls,
            text="Import",
            font=(self.fonts['heading'], 10),
            bg=self.colors['bg_light'],
            fg=self.colors['text_primary'],
            relief='flat',
            padx=10,
            pady=5,
            command=lambda: self._import_memories('episodic')
        )
        import_episodic_btn.pack(side='left', padx=5)
        
        # Export episodic memories button
        export_episodic_btn = tk.Button(
            episodic_controls,
            text="Export",
            font=(self.fonts['heading'], 10),
            bg=self.colors['bg_light'],
            fg=self.colors['text_primary'],
            relief='flat',
            padx=10,
            pady=5,
            command=lambda: self._export_memories('episodic')
        )
        export_episodic_btn.pack(side='left', padx=5)
        
        # Dreams memory controls
        dreaming_controls = tk.Frame(self.dreaming_frame, bg=self.colors['bg_medium'])
        dreaming_controls.pack(fill='x', padx=5, pady=5)
        
        # New dream button
        new_dream_btn = tk.Button(
            dreaming_controls, 
            text="New Dream", 
            font=(self.fonts['body'], 9),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_primary'],
            relief='flat',
            command=lambda: self._create_new_memory('dreaming')
        )
        new_dream_btn.pack(side='left', padx=5)
        
        # Import dreams memories button
        import_dreaming_btn = tk.Button(
            dreaming_controls, 
            text="Import Dreams", 
            font=(self.fonts['body'], 9),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_primary'],
            relief='flat',
            command=lambda: self._import_memories('dreaming')
        )
        import_dreaming_btn.pack(side='left', padx=5)
        
        # Export dreams memories button
        export_dreaming_btn = tk.Button(
            dreaming_controls, 
            text="Export Dreams", 
            font=(self.fonts['body'], 9),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_primary'],
            relief='flat',
            command=lambda: self._export_memories('dreaming')
        )
        export_dreaming_btn.pack(side='left', padx=5)
    
    def _create_new_memory(self, memory_type):
        """Create a new memory of specified type"""
        try:
            if memory_type == 'semantic':
                # Create new semantic memory
                new_memory = {"category": "", "content": ""}
            elif memory_type == 'episodic':
                # Create new episodic memory with current timestamp
                from datetime import datetime
                new_memory = {
                    "memory_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "context_tags": [],
                    "conversation_summary": "",
                    "what_worked": "",
                    "what_to_avoid": "",
                    "what_you_learned": ""
                }
            else:  # dreaming
                # Create new dream memory with manual source
                new_memory = {
                    "insight_type": "", 
                    "content": "",
                    "source": "manual"  # Indicate this was manually created
                }
            
            # Open edit dialog
            EditMemoryDialog(
                self.root, 
                new_memory, 
                lambda updated: self._add_memory(memory_type, updated)
            )
        
        except Exception as e:
            print(f"Error creating new memory: {e}")
            self.status_bar.config(text=f"Error creating new memory: {str(e)}")
            self.root.after(3000, self._update_memory_stats)

    def _add_memory(self, memory_type, memory):
        """Add a new memory to the specified type"""
        try:
            # Validate memory structure
            if not self._validate_memory_structure(memory, memory_type):
                print(f"Invalid {memory_type} memory structure")
                self.status_bar.config(text=f"Error: Invalid {memory_type} memory structure")
                self.root.after(3000, self._update_memory_stats)
                return False
            
            # Add to memory list
            self.memories[memory_type].append(memory)
            
            # Save to file
            self._save_memories(memory_type)
            
            # Update display
            if memory_type == 'semantic' and hasattr(self, 'semantic_scroll'):
                self._populate_memory_panel(
                    self.semantic_scroll, 
                    'semantic', 
                    lambda idx: self._edit_memory('semantic', idx)
                )
            elif memory_type == 'episodic' and hasattr(self, 'episodic_scroll'):
                self._populate_memory_panel(
                    self.episodic_scroll, 
                    'episodic', 
                    lambda idx: self._edit_memory('episodic', idx)
                )
            elif memory_type == 'dreaming' and hasattr(self, 'dreaming_scroll'):
                self._populate_memory_panel(
                    self.dreaming_scroll, 
                    'dreaming', 
                    lambda idx: self._edit_memory('dreaming', idx)
                )
            return True
            
        except Exception as e:
            print(f"Error adding memory: {e}")
            self.status_bar.config(text=f"Error adding memory: {str(e)}")
            self.root.after(3000, self._update_memory_stats)
            return False

    def _draw_neural_network(self, canvas):
        """This method is no longer used - replaced with memory access"""
        pass

    def _on_send(self, event=None, preset_message=None):
        message = preset_message if preset_message else self.input_field.get().strip()
        if message:
            self.status_bar.config(text="PROCESSING QUERY...")
            self.input_field.configure(state='disabled')
            self.send_button.configure(state='disabled')
            
            # Start thinking indicator
            self._start_thinking_indicator()
            
            self.display_message(message, "user")
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Clear input field before processing message
            self.input_field.delete(0, tk.END)
            
            threading.Thread(
                target=self._process_message,
                args=(message,),
                daemon=True
            ).start()
    
    def _start_thinking_indicator(self):
        """Start an elegant thinking indicator animation"""
        canvas = self.thinking_indicator_canvas
        canvas.delete("thinking")
        
        width = canvas.winfo_width() or 800
        
        # Create initial indicator
        self.thinking_indicator = canvas.create_rectangle(
            0, 0, 0, 3,
            fill=self.colors['stark_blue'],
            tags="thinking"
        )
        
        # Animate the indicator
        def animate_indicator(step=0):
            if not hasattr(self, 'thinking_indicator') or step >= 100:
                return
                
            # Calculate width based on step
            indicator_width = (step / 100) * width
            
            # Update indicator width
            canvas.coords(
                self.thinking_indicator,
                0, 0, indicator_width, 3
            )
            
            # Continue animation
            self.root.after(10, lambda: animate_indicator(step + 1))
            
        # Start animation
        animate_indicator()
    
    def _stop_thinking_indicator(self):
        """Stop thinking indicator with fade out"""
        if hasattr(self, 'thinking_indicator_canvas'):
            self.thinking_indicator_canvas.delete("thinking")
    
    def _process_message(self, message):
        try:
            response = self.chat_callback(message)
            self.msg_queue.put(("assistant", response))
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            self.status_bar.config(text="READY")
        except Exception as e:
            self.msg_queue.put(("error", f"System Error: {str(e)}"))
            self.status_bar.config(text="ERROR DETECTED")
        finally:
            self.root.after(0, lambda: self.input_field.configure(state='normal'))
            self.root.after(0, lambda: self.send_button.configure(state='normal'))
            self.root.after(0, lambda: self.input_field.focus_set())
            # Ensure input field is clear
            self.root.after(0, lambda: self.input_field.delete(0, tk.END))
            self.root.after(0, self._stop_thinking_indicator)
    
    def _clear_chat(self, event=None):
        if messagebox.askyesno("Clear Conversation", "Are you sure you want to clear the conversation history?"):
            self.chat_display.delete(1.0, tk.END)
            self.conversation_history = []
            self.status_bar.config(text="CONVERSATION CLEARED")
    
    def display_message(self, message, sender):
        """Display a message with clean, minimalist styling"""
        # Add spacing between messages
        self.chat_display.insert(tk.END, "\n\n")
        
        # Create timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Configure the message prefix based on sender
        if sender == "user":
            prefix = f"[{timestamp}] You » "
            prefix_tag = "user_prefix"
            msg_tag = "user_message"
            self.chat_display.tag_config(
                prefix_tag,
                foreground=self.colors['stark_blue'],
                font=(self.fonts['heading'], 11, "bold")
            )
            self.chat_display.tag_config(
                msg_tag,
                foreground=self.colors['text_primary'],
                font=(self.fonts['heading'], 11)
            )
        elif sender == "error":
            prefix = f"[{timestamp}] SYSTEM ERROR » "
            prefix_tag = "error_prefix"
            msg_tag = "error_message"
            self.chat_display.tag_config(
                prefix_tag,
                foreground=self.colors['error'],
                font=(self.fonts['heading'], 11, "bold")
            )
            self.chat_display.tag_config(
                msg_tag,
                foreground=self.colors['error'],
                font=(self.fonts['mono'], 11)
            )
        else:
            prefix = f"[{timestamp}] FRED » "
            prefix_tag = "fred_prefix"
            msg_tag = "fred_message"
            self.chat_display.tag_config(
                prefix_tag,
                foreground=self.colors['stark_blue'],
                font=(self.fonts['heading'], 11, "bold")
            )
            self.chat_display.tag_config(
                msg_tag,
                foreground=self.colors['hologram'],
                font=(self.fonts['heading'], 11)
            )
        
        # Insert the prefix with appropriate styling
        self.chat_display.insert(tk.END, prefix, prefix_tag)
        
        # Insert message with elegant typing effect
        def type_message(msg, index=0):
            if index < len(msg):
                # Add a bit of randomness to typing speed for realism
                typing_delay = 5 if sender == "user" else random.randint(5, 15)
                
                # Insert character with appropriate styling
                self.chat_display.insert(tk.END, msg[index], msg_tag)
                self.chat_display.see(tk.END)
                
                # Handle special formatting for code blocks
                if msg[index:index+3] == "```" and sender == "assistant":
                    # Format code blocks with distinct styling
                    code_end = msg.find("```", index+3)
                    if code_end != -1:
                        code_block = msg[index:code_end+3]
                        # Skip ahead after inserting full code block
                        self.chat_display.insert(tk.END, code_block[1:], "code_block")
                        self.chat_display.tag_config(
                            "code_block",
                            foreground="#a2ffd0",
                            background=self.colors['bg_dark'],
                            font=(self.fonts['mono'], 10)
                        )
                        self.chat_display.see(tk.END)
                        index = code_end + 3
                
                # Schedule next character
                self.root.after(typing_delay, type_message, msg, index + 1)
            else:
                # Message complete, insert newline
                self.chat_display.insert(tk.END, "\n")
                self.chat_display.see(tk.END)
                
                # Add a minimal separator
                self.chat_display.insert(tk.END, "─" * 50, "separator")
        self.chat_display.tag_config(
            "separator",
                    foreground=self.colors['accent_dim'],
            font=(self.fonts['mono'], 8)
        )
        self.chat_display.insert(tk.END, "\n")
        
        # Handle null messages
        if message is None:
            message = "No response received"
        
        # Start typing animation
        type_message(message)
    
    def _setup_layout(self):
        """Set up the interface layout"""
        # Set up container
        self.container.pack(expand=True, fill='both')
        self.main_frame.pack(expand=True, fill='both')
            
        # Create holographic visualization
        self._create_holographic_display()
        
        # Set focus to input field
        self.input_field.focus_set()
    
    def _start_msg_checker(self):
        try:
            while not self.msg_queue.empty():
                sender, message = self.msg_queue.get_nowait()
                self.display_message(message, sender.lower())
        finally:
            self.root.after(100, self._start_msg_checker)
    
    def run(self):
        # Apply window opacity with platform-specific handling
        try:
            self.root.attributes('-alpha', 0.95)
        except Exception as e:
            print(f"Note: Transparency not fully supported on this system. {e}")
            
        # Center window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')
        
        self.root.mainloop()

    def _update_memory(self, memory_type, index, updated_memory):
        """Update an existing memory with validation"""
        if memory_type in self.memories and index < len(self.memories[memory_type]):
            # Validate the updated memory
            if self._validate_memory_structure(updated_memory, memory_type):
                # Update the memory
                self.memories[memory_type][index] = updated_memory
                
                # Refresh the display
                try:
                    if hasattr(self, 'memory_notebook'):
                        # Get current tab
                        current_tab = self.memory_notebook.index(self.memory_notebook.select())
                        
                        # Find the appropriate memory panel
                        memory_panel = None
                        if memory_type == 'semantic' and hasattr(self, 'semantic_frame'):
                            for widget in self.semantic_frame.winfo_children():
                                if isinstance(widget, scrolledtext.ScrolledText):
                                    memory_panel = widget
                                    break
                        elif memory_type == 'episodic' and hasattr(self, 'episodic_frame'):
                            for widget in self.episodic_frame.winfo_children():
                                if isinstance(widget, scrolledtext.ScrolledText):
                                    memory_panel = widget
                                    break
                        elif memory_type == 'dreaming' and hasattr(self, 'dreaming_frame'):
                            for widget in self.dreaming_frame.winfo_children():
                                if isinstance(widget, scrolledtext.ScrolledText):
                                    memory_panel = widget
                                    break
                        
                        # Repopulate the specific memory panel if found
                        if memory_panel:
                            self._populate_memory_panel(
                                memory_panel, 
                                memory_type, 
                                lambda idx, t=memory_type: self._edit_memory(t, idx)
                            )
                        else:
                            # Fallback: reload all memories
                            self._load_memories()
                            # Switch back to the previously selected tab
                            self.memory_notebook.select(current_tab)
                        
                        # Update memory stats
                        self._update_memory_stats()
                        
                        # Show success message
                        self.status_bar.config(text=f"{memory_type.capitalize()} memory updated")
                        self.root.after(3000, self._update_memory_stats)
                except Exception as e:
                    print(f"Error refreshing memory display: {e}")
                    # Fallback: reload all memories
                    self._load_memories()
            else:
                messagebox.showerror("Validation Error", 
                    f"Invalid memory structure for {memory_type} memory")

    def _start_arc_pulse(self):
        """Create pulsing effect for the arc reactor"""
        if not hasattr(self, 'arc_reactor_canvas'):
            return
            
        def pulse_animation(step=0):
            if not hasattr(self, 'arc_reactor_canvas') or not self.arc_reactor_canvas.winfo_exists():
                return
                
            try:
                # Get current dimensions
                width = 180
                height = 180
                center_x = width / 2
                center_y = height / 2
                
                # Calculate pulse effect
                pulse_factor = 0.8 + 0.2 * math.sin(step * 0.1)  # Smooth sine wave
                
                # Update outer ring
                outer_radius = 75 * pulse_factor
                self.arc_reactor_canvas.coords(
                    "reactor_ring",
                    center_x - outer_radius, center_y - outer_radius,
                    center_x + outer_radius, center_y + outer_radius
                )
                
                # Update core glow
                core_radius = 25 * (1 + 0.1 * math.sin(step * 0.2))  # Faster pulse for core
                self.arc_reactor_canvas.coords(
                    "reactor_core",
                    center_x - core_radius, center_y - core_radius,
                    center_x + core_radius, center_y + core_radius
                )
                
                # Update ambient glow
                glow_radius = 85 * (1 + 0.05 * math.sin(step * 0.05))  # Slower pulse for glow
                self.arc_reactor_canvas.coords(
                    "reactor_glow",
                    center_x - glow_radius, center_y - glow_radius,
                    center_x + glow_radius, center_y + glow_radius
                )
                
                # Update triangular housing (Mark 7 style)
                triangle_size = 50 * (1 + 0.05 * math.sin(step * 0.15))
                triangle_points = [
                    center_x, center_y - triangle_size,  # Top point
                    center_x - triangle_size * 0.866, center_y + triangle_size * 0.5,  # Bottom left
                    center_x + triangle_size * 0.866, center_y + triangle_size * 0.5,  # Bottom right
                ]
                try:
                    triangle_item = self.arc_reactor_canvas.find_withtag("reactor_triangle")[0]
                    self.arc_reactor_canvas.coords(triangle_item, *triangle_points)
                    
                    # Update inner triangle
                    inner_triangle_size = triangle_size * 0.7
                    inner_triangle_points = [
                        center_x, center_y - inner_triangle_size,  # Top point
                        center_x - inner_triangle_size * 0.866, center_y + inner_triangle_size * 0.5,  # Bottom left
                        center_x + inner_triangle_size * 0.866, center_y + inner_triangle_size * 0.5,  # Bottom right
                    ]
                    inner_triangle_item = self.arc_reactor_canvas.find_withtag("reactor_triangle_inner")[0]
                    self.arc_reactor_canvas.coords(inner_triangle_item, *inner_triangle_points)
                    
                    # Change opacity based on pulse
                    glow_intensity = 0.7 + 0.3 * math.sin(step * 0.15)
                    triangle_color = self._adjust_color_opacity(self.colors['stark_blue'], 0.3 * glow_intensity)
                    self.arc_reactor_canvas.itemconfig(triangle_item, fill=triangle_color)
                except (IndexError, tk.TclError):
                    pass  # Handle case where triangles haven't been created yet
                
                # Continue animation
                self.root.after(50, lambda: pulse_animation(step + 1))
                
            except Exception as e:
                print(f"Error in arc pulse animation: {e}")
                self.root.after(1000, self._start_arc_pulse)
        
        # Start pulse animation
        try:
            pulse_animation()
        except Exception as e:
            print(f"Failed to start arc pulse: {e}")

    def _create_radial_menu(self):
        """Create the radial menu buttons for memory access"""
        # Create buttons in a circular layout
        buttons = [
            ("Chat", lambda: self.memory_notebook.select(0)),
            ("Semantic", lambda: self.memory_notebook.select(0)),
            ("Episodic", lambda: self.memory_notebook.select(1)),
            ("Dreams", lambda: self.memory_notebook.select(2))
        ]
        
        # Calculate positions for buttons
        center_x = 90  # Half of arc_reactor_canvas width
        center_y = 90  # Half of arc_reactor_canvas height
        radius = 40   # Distance from center
        
        # Create buttons in a circular layout
        for i, (text, command) in enumerate(buttons):
            angle = (2 * math.pi * i) / len(buttons)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Create button with hover effect
            btn = tk.Button(
                self.arc_reactor_canvas,
                text=text,
                font=(self.fonts['heading'], 8),
                bg=self.colors['bg_medium'],
                fg=self.colors['text_primary'],
                activebackground=self.colors['accent'],
                activeforeground=self.colors['text_primary'],
                relief='flat',
                padx=5,
                pady=2,
                command=command
            )
            
            # Add hover effect
            def _on_enter(e, b=btn):
                b.config(bg=self.colors['accent'])
                
            def _on_leave(e, b=btn):
                b.config(bg=self.colors['bg_medium'])
                
            btn.bind("<Enter>", _on_enter)
            btn.bind("<Leave>", _on_leave)
            
            # Position button
            self.arc_reactor_canvas.create_window(x, y, window=btn, anchor='center')

    def _import_memories(self, memory_type):
        """Import memories from a JSON or JSONL file"""
        try:
            # Ask user for import file
            file_types = [("JSON files", "*.json"), ("All files", "*.*")]
            import_file = filedialog.askopenfilename(
                filetypes=file_types,
                title=f"Import {memory_type.capitalize()} Memories"
            )
            
            if not import_file:
                return  # User cancelled
                
            # Read the file
            with open(import_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            if not content:
                messagebox.showinfo("Import", "The selected file is empty")
                return
                
            # Try to parse as JSON array first
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    # It's a JSON array
                    imported_memories = data
                else:
                    # It's a single JSON object
                    imported_memories = [data]
            except json.JSONDecodeError:
                # Try JSONL format (one JSON object per line)
                imported_memories = []
                with open(import_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            memory = json.loads(line)
                            imported_memories.append(memory)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line: {e}")
                            continue
            
            if not imported_memories:
                messagebox.showinfo("Import", "No valid memories found in the selected file")
                return
                
            # Validate memories
            valid_memories = []
            for memory in imported_memories:
                if self._validate_memory_structure(memory, memory_type):
                    valid_memories.append(memory)
            
            if not valid_memories:
                messagebox.showinfo("Import", 
                    f"No valid {memory_type} memories found in the selected file")
                return
                
            # Ask user if they want to replace or append
            if self.memories[memory_type]:
                replace = messagebox.askyesno(
                    "Import Options", 
                    f"Do you want to replace existing {memory_type} memories?\n\n"
                    "Yes: Replace all existing memories\n"
                    "No: Append imported memories to existing ones"
                )
            else:
                replace = True
                
            # Update memories
            if replace:
                self.memories[memory_type] = valid_memories
            else:
                self.memories[memory_type].extend(valid_memories)
                
            # Save to file
            self._save_memories(memory_type)
            
            # Repopulate memory panel
            if memory_type == 'semantic' and hasattr(self, 'semantic_scroll'):
                self._populate_memory_panel(
                    self.semantic_scroll, 
                    'semantic', 
                    lambda idx: self._edit_memory('semantic', idx)
                )
            elif memory_type == 'episodic' and hasattr(self, 'episodic_scroll'):
                self._populate_memory_panel(
                    self.episodic_scroll, 
                    'episodic', 
                    lambda idx: self._edit_memory('episodic', idx)
                )
            elif memory_type == 'dreaming' and hasattr(self, 'dreaming_scroll'):
                self._populate_memory_panel(
                    self.dreaming_scroll, 
                    'dreaming', 
                    lambda idx: self._edit_memory('dreaming', idx)
                )
            
            # Show success message
            messagebox.showinfo("Import Complete", 
                f"Successfully imported {len(valid_memories)} {memory_type} memories")
                
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import memories: {str(e)}")

class EditMemoryDialog:
    """Dialog for editing memory entries"""
    def __init__(self, parent, memory, callback):
        self.memory = memory.copy()  # Create a copy to avoid modifying original until save
        self.callback = callback
        self.parent = parent
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Edit Memory")
        self.dialog.geometry("650x550")  # Ensure adequate initial size
        self.dialog.minsize(500, 400)    # Set minimum size to prevent too small dialog
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Set dialog styling
        self.dialog.configure(bg='#1a0438')  # Use medium purple
        
        # Add proper window icon
        try:
            self.dialog.iconbitmap("assets/fred_icon.ico")
        except:
            pass
            
        # Make dialog modal
        self.dialog.focus_set()
        self.dialog.protocol("WM_DELETE_WINDOW", self._cancel)
        
        # Bind escape key to cancel
        self.dialog.bind("<Escape>", lambda e: self._cancel())
        # Bind Enter key to save (only if not in a Text widget)
        self.dialog.bind("<Return>", lambda e: self._save() if e.widget.__class__.__name__ != "Text" else None)
        
        # Create form based on memory type
        self.create_form()
        
        # Center the dialog on the parent window
        self._center_window()
        
        # Ensure dialog appears on top
        self.dialog.lift()
        self.dialog.focus_force()
        
    def _center_window(self):
        """Center the dialog on the parent window"""
        self.dialog.update_idletasks()
        
        # Get parent and dialog dimensions
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        # Calculate position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        # Set position
        self.dialog.geometry(f"+{x}+{y}")
    
    def create_form(self):
        """Create form fields based on memory structure"""
        # Main frame
        main_frame = tk.Frame(self.dialog, bg='#1a0438', padx=15, pady=15)
        main_frame.pack(fill='both', expand=True)
        
        # Title
        memory_type = "Unknown"
        if "category" in self.memory and "content" in self.memory and "memory_timestamp" not in self.memory:
            # This is a semantic memory
            memory_type = "Semantic"
        elif "memory_timestamp" in self.memory:
            # This is an episodic memory
            memory_type = "Episodic"
        else:
            # This is an assumption memory
            memory_type = "Assumption"
            
        title = tk.Label(
            main_frame,
            text=f"Edit {memory_type} Memory",
            font=("Rajdhani", 18, "bold"),
            fg='#c17bff',  # bright purple
            bg='#1a0438'
        )
        title.pack(pady=(0, 15))
        
        # Create scrollable frame for form fields
        canvas_frame = tk.Frame(main_frame, bg='#1a0438')
        canvas_frame.pack(fill='both', expand=True)
        
        canvas = tk.Canvas(canvas_frame, bg='#1a0438', highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#1a0438')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Configure canvas to expand with frame
        def _configure_canvas(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        canvas.bind("<Configure>", _configure_canvas)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Store canvas reference for cleanup
        self.form_canvas = canvas
        
        # Position canvas and scrollbar properly
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel event for scrolling with safety check
        def _on_mousewheel(event):
            if canvas.winfo_exists():
                try:
                    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                except Exception as e:
                    print(f"Mousewheel error: {e}")
        
        # Store the mousewheel function for later unbinding
        self.mousewheel_func = _on_mousewheel
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Form fields container
        self.form_fields = {}
        
        # Create form fields based on memory type
        if memory_type == "Semantic":  # Semantic memory
            self._create_field(scrollable_frame, "category", "Category:", 0)
            self._create_text_area(scrollable_frame, "content", "Content:", 1)
        
        elif memory_type == "Episodic":  # Episodic memory
            self._create_field(scrollable_frame, "memory_timestamp", "Timestamp:", 0)
            self._create_field(scrollable_frame, "context_tags", "Tags (comma-separated):", 1)
            self._create_text_area(scrollable_frame, "conversation_summary", "Summary:", 2)
            self._create_text_area(scrollable_frame, "what_worked", "What Worked:", 3)
            self._create_text_area(scrollable_frame, "what_to_avoid", "What to Avoid:", 4)
            self._create_text_area(scrollable_frame, "what_you_learned", "What You Learned:", 5)
        
        else:  # Assumptions memory - handle both "about" and "category" fields
            field_name = "about" if "about" in self.memory else "category"
            self._create_field(scrollable_frame, field_name, "Category:", 0)
            self._create_text_area(scrollable_frame, "content", "Content:", 1)
        
        # Ensure scrollable_frame has minimum dimensions
        scrollable_frame.update_idletasks()
        min_width = max(500, scrollable_frame.winfo_reqwidth())
        min_height = max(400, scrollable_frame.winfo_reqheight())
        canvas.config(width=min_width, height=min_height)
        
        # Button frame at the bottom
        button_frame = tk.Frame(main_frame, bg='#1a0438')
        button_frame.pack(fill='x', pady=(15, 0))
        
        # Save button
        save_button = tk.Button(
            button_frame,
            text="Save",
            font=("Rajdhani", 12, "bold"),
            bg='#9d6ad8',  # accent
            fg='#f5f0ff',  # text primary
            padx=20,
            pady=5,
            relief='flat',
            command=self._save
        )
        save_button.pack(side='right', padx=5)
        
        # Cancel button
        cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            font=("Rajdhani", 12),
            bg='#2c0657',  # bg light
            fg='#f5f0ff',  # text primary
            padx=20,
            pady=5,
            relief='flat',
            command=self._cancel
        )
        cancel_button.pack(side='right', padx=5)
        
        # Set initial focus to first field
        if self.form_fields:
            first_field = list(self.form_fields.values())[0]
            first_field.focus_set()
    
    def _create_field(self, parent, field_name, label_text, row):
        """Create a labeled field in the form"""
        frame = tk.Frame(parent, bg='#1a0438', pady=5)
        frame.pack(fill='x', pady=5)
        
        # Label
        label = tk.Label(
            frame,
            text=label_text,
            font=("Rajdhani", 12, "bold"),
            fg='#c8a2ff',  # hologram
            bg='#1a0438',
            anchor='w'
        )
        label.pack(fill='x')
        
        # Entry field
        entry = tk.Entry(
            frame,
            font=("Consolas", 11),
            bg='#2c0657',  # bg light
            fg='#f5f0ff',  # text primary
            insertbackground='#c17bff',  # bright accent
            relief='flat',
            bd=0,
            width=40,  # Ensure minimum width is set
            highlightthickness=1,  # Add highlight border
            highlightbackground='#4a1987',  # Dark purple border
            highlightcolor='#c17bff'  # Bright accent when focused
        )
        entry.pack(fill='x', ipady=5, pady=(2, 0))
        
        # Pre-fill with existing value
        if field_name in self.memory:
            if field_name == "context_tags" and isinstance(self.memory[field_name], list):
                entry.insert(0, ", ".join(self.memory[field_name]))
            else:
                entry.insert(0, str(self.memory[field_name]))
        
        # Store reference
        self.form_fields[field_name] = entry
    
    def _create_text_area(self, parent, field_name, label_text, row):
        """Create a labeled text area in the form"""
        frame = tk.Frame(parent, bg='#1a0438', pady=5)
        frame.pack(fill='x', expand=True, pady=5)
        
        # Label
        label = tk.Label(
            frame,
            text=label_text,
            font=("Rajdhani", 12, "bold"),
            fg='#c8a2ff',  # hologram
            bg='#1a0438',
            anchor='w'
        )
        label.pack(fill='x')
        
        # Text area container for proper sizing
        text_container = tk.Frame(frame, bg='#2c0657', height=100)
        text_container.pack(fill='both', expand=True, pady=(2, 0))
        text_container.pack_propagate(False)  # Prevent container from shrinking
        
        # Text area
        text_area = tk.Text(
            text_container,
            font=("Consolas", 11),
            bg='#2c0657',  # bg light
            fg='#f5f0ff',  # text primary
            insertbackground='#c17bff',  # bright accent
            height=4,
            relief='flat',
            bd=0,
            padx=5,
            pady=5,
            wrap='word',
            highlightthickness=1,  # Add highlight border
            highlightbackground='#4a1987',  # Dark purple border
            highlightcolor='#c17bff'  # Bright accent when focused
        )
        text_area.pack(fill='both', expand=True)
        
        # Add scrollbar
        scrollbar = tk.Scrollbar(text_container)
        scrollbar.pack(side='right', fill='y')
        text_area.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=text_area.yview)
        
        # Pre-fill with existing value
        if field_name in self.memory:
            text_area.insert("1.0", str(self.memory[field_name]))
        
        # Store reference
        self.form_fields[field_name] = text_area
    
    def _cancel(self):
        """Cancel editing and close the dialog"""
        # Unbind mousewheel event to prevent callbacks after dialog is destroyed
        if hasattr(self, 'mousewheel_func'):
            try:
                self.dialog.unbind_all("<MouseWheel>")
            except Exception as e:
                print(f"Unbind error: {e}")
        self.dialog.destroy()
        
    def _save(self):
        """Save the edited memory with validation"""
        try:
            has_validation_error = False
            validation_message = ""
            
            # Create a new memory object to hold the updated values
            updated_memory = {}
            
            # Determine memory type
            memory_type = "Unknown"
            if "category" in self.memory and "content" in self.memory and "memory_timestamp" not in self.memory:
                memory_type = "Semantic"
            elif "memory_timestamp" in self.memory:
                memory_type = "Episodic"
            else:
                memory_type = "Assumption"
            
            print(f"Saving {memory_type} memory")
            
            # Get values from widgets and handle special cases
            for field_name, widget in self.form_fields.items():
                if isinstance(widget, tk.Text):
                    value = widget.get("1.0", "end-1c").strip()  # Get text without trailing newline
                else:
                    value = widget.get().strip()
                
                print(f"  Field: {field_name} = {value[:30]}{'...' if len(value) > 30 else ''}")
                
                # Check if required fields are not empty
                required_field = False
                
                if memory_type == "Semantic" and field_name == "category":
                    required_field = True
                elif memory_type == "Episodic" and field_name == "conversation_summary":
                    required_field = True
                elif memory_type == "Assumption" and (field_name == "about" or field_name == "category"):
                    required_field = True
                elif field_name == "content":
                    required_field = True
                
                if required_field and not value:
                    has_validation_error = True
                    validation_message = f"The field '{field_name}' cannot be empty."
                    widget.focus_set()  # Set focus to the problematic field
                    break
                
                # Handle special fields
                if field_name == "context_tags" and isinstance(self.memory[field_name], list):
                    # Convert comma-separated string to list
                    updated_memory[field_name] = [tag.strip() for tag in value.split(",") if tag.strip()]
                else:
                    updated_memory[field_name] = value
            
            # If no validation error, copy all fields from original memory that weren't in the form
            if not has_validation_error:
                for key, value in self.memory.items():
                    if key not in updated_memory:
                        updated_memory[key] = value
                        
                print(f"Final memory structure: {list(updated_memory.keys())}")
            
            if has_validation_error:
                tk.messagebox.showerror("Validation Error", validation_message)
                return
                
            # Call callback with updated memory
            self.callback(updated_memory)
            
            # Unbind mousewheel event before closing
            if hasattr(self, 'mousewheel_func'):
                try:
                    self.dialog.unbind_all("<MouseWheel>")
                except Exception as e:
                    print(f"Unbind error: {e}")
            
            # Close dialog
            self.dialog.destroy()
            
        except Exception as e:
            error_msg = f"An error occurred while saving: {str(e)}"
            print(error_msg)
            tk.messagebox.showerror("Error", error_msg)
            
    def _validate_field(self, field_name, value):
        """Validate field values"""
        if field_name in ["category", "about", "content", "conversation_summary"] and not value.strip():
            return False, f"{field_name} cannot be empty"
        return True, ""
