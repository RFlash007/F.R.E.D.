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
        
        # Main frames styling
        style.configure('Neural.TFrame', background='#0a1520')
        style.configure('Sidebar.TFrame', background='#0d1a2a')
        style.configure('TSeparator', background='#00bfff')
        
        # Action buttons styling with futuristic look
        style.configure('Action.TButton',
                        font=('Rajdhani', 11, 'bold'),
                        padding=8,
                        background='#0d1a2a',
                        foreground='#00bfff')
        style.map('Action.TButton',
                  background=[('active', '#102030')],
                  foreground=[('active', '#00dfff')])
        
        # Entry field styling with neon effects
        style.configure('Neural.TEntry',
                        fieldbackground='#0d1a2a',
                        foreground='#00bfff',
                        insertcolor='#00dfff',
                        borderwidth=0)
        
        # Custom label style for headers and status
        style.configure('Neural.TLabel',
                        background='#0a1520',
                        foreground='#00bfff',
                        font=('Rajdhani', 11))
        
    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0, bg='#1a2433', fg='#00bfff',
                           activebackground='#2a3443', activeforeground='#00dfff')
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Clear Chat", command=self._clear_chat)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        edit_menu = tk.Menu(menubar, tearoff=0, bg='#1a2433', fg='#00bfff',
                           activebackground='#2a3443', activeforeground='#00dfff')
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Copy", command=lambda: self.chat_display.event_generate("<<Copy>>"))
        edit_menu.add_command(label="Paste", command=lambda: self.input_field.event_generate("<<Paste>>"))
    
    def _create_header(self):
        """Create header with minimal clean design and animated time display"""
        self.header_frame = ttk.Frame(self.main_frame, style='Neural.TFrame')
        
        # Time display with pulsing separator
        self.time_label = ttk.Label(
            self.header_frame,
            text="",
            foreground='#00bfff',
            background='#0a1520',
            font=('Rajdhani', 12, 'bold')
        )
        self.time_label.pack(side='right', padx=10)
        self._update_time()
    
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
        actions = [
            ("🔍 SCAN", lambda: self._on_send(None, "Scan systems"), "Analyze system status"),
            ("🔄 RESET", self._clear_chat, "Clear current session")
        ]
        
        for text, command, tooltip in actions:
            btn_frame = ttk.Frame(self.actions_frame, style='Sidebar.TFrame')
            btn_frame.pack(fill='x', padx=10, pady=2)
            
            btn = ttk.Button(
                btn_frame,
                text=text,
                command=command,
                style='Action.TButton'
            )
            btn.pack(fill='x')
            self._create_tooltip(btn, tooltip)

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
        """Update system metrics with smooth animations (if metrics widgets exist)"""
        import psutil
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        
        current_cpu_width = self.cpu_canvas.coords(self.cpu_bar)[2]
        target_cpu_width = (cpu / 100) * 100
        delta_cpu = (target_cpu_width - current_cpu_width) * 0.2
        
        self.cpu_canvas.coords(self.cpu_bar, 0, 0, current_cpu_width + delta_cpu, 15)
        current_mem_width = self.memory_canvas.coords(self.memory_bar)[2]
        target_mem_width = (memory / 100) * 100
        delta_mem = (target_mem_width - current_mem_width) * 0.2
        
        self.memory_canvas.coords(self.memory_bar, 0, 0, current_mem_width + delta_mem, 15)
        self.cpu_label.config(text=f"CPU {cpu:.0f}%")
        self.memory_label.config(text=f"MEM {memory:.0f}%")
        self._pulse_status_circle()
        self.root.after(100, self._update_metrics)
    
    def _pulse_status_circle(self):
        t = time.time() * 2
        pulse = abs(math.sin(t)) * 0.3 + 0.7
        color = self._adjust_color_brightness('#00bfff', pulse)
        self.status_canvas.itemconfig(self.status_circle, fill=color, outline=color)

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
        self.sidebar_title.pack(fill='x', pady=(0, 10), padx=20)
        ttk.Separator(self.sidebar, orient='horizontal').pack(fill='x', padx=10, pady=5)
        self.actions_frame.pack(fill='x', pady=10, side='bottom')
        self.main_frame.pack(side='left', expand=True, fill='both')
        self.header_frame.pack(fill='x', pady=(0, 10))
        self.chat_frame.pack(expand=True, fill='both', pady=(0, 20))
        self.chat_display.pack(expand=True, fill='both')
        
        # Tech overlay pattern
        self.overlay_canvas = tk.Canvas(
            self.chat_frame,
            bg='#0d1a2a',
            highlightthickness=0,
            width=self.chat_frame.winfo_width(),
            height=20
        )
        self.overlay_canvas.place(x=0, y=0, relwidth=1)
        self._create_tech_pattern()
        
        # Data stream effect and hex grid overlays
        self.stream_canvas.place(relx=1, rely=0, anchor='ne', relheight=1)
        self.hex_canvas.place(relx=1, rely=0, anchor='ne', relwidth=0.2, relheight=1)
        
        self.input_container.pack(fill='x', padx=20, pady=10)
        self.input_frame.pack(fill='x', expand=True)
        self.input_border = tk.Canvas(self.input_frame, height=2, bg='#0a1520', highlightthickness=0)
        self.input_border.pack(fill='x', side='bottom')
        self.input_field.pack(side='left', expand=True, fill='x', padx=(0, 10))
        self.send_button.pack(side='right')
        self.status_frame.pack(fill='x', pady=(10, 0))
        self.status_bar.pack(side='left')
        
        self.input_field.bind("<FocusIn>", self._start_border_animation)
        self.input_field.bind("<FocusOut>", self._stop_border_animation)
        self.input_field.focus_set()
        self._start_animations()

    def _start_animations(self):
        """Start all UI animations"""
        self._animate_border()
        self._animate_enhanced_reactor()
        self._animate_hex_grid()
        self._animate_data_stream()

    def _animate_border(self):
        """Animate the input border with a flowing neon effect"""
        if hasattr(self, '_border_animation_active') and self._border_animation_active:
            width = self.input_border.winfo_width()
            t = time.time() * 2
            for i in range(3):
                offset = i * width / 3
                x = (t + offset) % width
                opacity = abs(math.sin(x / width * math.pi))
                color = self._adjust_color_brightness('#00bfff', opacity * 0.7 + 0.3)
                self.input_border.create_line(x, 0, x + width/6, 0, fill=color, width=2, tags='flow')
            self.input_border.delete('old')
        self.root.after(50, self._animate_border)
    
    def _start_border_animation(self, event):
        self._border_animation_active = True
    
    def _stop_border_animation(self, event):
        self._border_animation_active = False
        self.input_border.delete('all')

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
        """Update time display with a blinking separator for a dynamic feel"""
        current_time = datetime.now().strftime("%H:%M:%S")
        separator = ":" if time.time() % 1 < 0.5 else " "
        formatted_time = current_time.replace(":", separator)
        self.time_label.config(text=formatted_time, foreground='#00bfff')
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
