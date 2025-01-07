import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
from queue import Queue

class ChatUI:
    def __init__(self, chat_callback):
        self.root = tk.Tk()
        self.root.title("Fred Chat Interface")
        self.root.geometry("800x600")
        self.root.configure(bg='#1a1a1a')  # Dark background like Jarvis UI
        
        # Message queue for thread-safe UI updates
        self.msg_queue = Queue()
        
        # Chat callback
        self.chat_callback = chat_callback
        
        self._create_widgets()
        self._setup_layout()
        self._start_msg_checker()
    
    def _create_widgets(self):
        # Create main frame with padding and style
        self.main_frame = ttk.Frame(self.root, padding="10")
        
        # Style configuration
        style = ttk.Style()
        style.configure('Chat.TFrame', background='#1a1a1a')
        style.configure('TSeparator', background='#00bfff')  # Bright blue separator
        
        # Create chat display area with custom styling
        self.chat_display = scrolledtext.ScrolledText(
            self.main_frame,
            wrap=tk.WORD,
            width=70,
            height=30,
            font=("Rajdhani", 10),  # Futuristic font
            bg='#1a1a1a',  # Dark background
            fg='#00bfff',  # Bright blue text
            insertbackground='#00bfff',  # Cursor color
            relief='flat',
            borderwidth=1
        )
        
        # Create bottom frame for input with padding and separator
        self.bottom_frame = ttk.Frame(self.main_frame, padding=(0, 10, 0, 0))
        self.separator = ttk.Separator(self.main_frame, orient='horizontal')
        
        # Create input field with styling
        self.input_field = ttk.Entry(
            self.bottom_frame,
            width=60,
            font=("Rajdhani", 10),
            style='Jarvis.TEntry'
        )
        self.input_field.bind("<Return>", self._on_send)
        
        # Create send button with styling
        self.send_button = tk.Button(
            self.bottom_frame,
            text="EXECUTE",
            command=self._on_send,
            font=('Rajdhani', 10, 'bold'),
            bg='#00bfff',
            fg='#1a1a1a',
            activebackground='#008cc7',
            activeforeground='#ffffff',
            relief='flat',
            padx=15,
            pady=8
        )
        
        # Configure entry style
        style.configure('Jarvis.TEntry',
                       fieldbackground='#2a2a2a',
                       foreground='#00bfff',
                       insertcolor='#00bfff')
        
        # Create quick action buttons
        self.quick_actions_frame = ttk.Frame(self.main_frame)
        
        self.news_button = ttk.Button(
            self.quick_actions_frame,
            text="📰 News",
            command=lambda: self._on_send(None, "Show me the latest news")
        )
        
        self.system_button = ttk.Button(
            self.quick_actions_frame,
            text="💻 System Status",
            command=lambda: self._on_send(None, "Show system status")
        )
        
        # Pack quick action buttons
        self.news_button.pack(side='left', padx=5)
        self.system_button.pack(side='left', padx=5)

    def _setup_layout(self):
        # Setup main frame to fill window
        self.main_frame.pack(expand=True, fill='both')
        
        # Setup chat display with proper spacing
        self.chat_display.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Add separator
        self.separator.pack(fill='x', pady=5)
        
        # Setup bottom frame
        self.bottom_frame.pack(fill='x', padx=5)
        
        # Setup input field and send button with proper spacing
        self.input_field.pack(side='left', expand=True, fill='x', padx=(0, 10))
        self.send_button.pack(side='right')
        
        # Set initial focus to input field
        self.input_field.focus_set()
        
        # Add quick actions frame below chat display
        self.quick_actions_frame.pack(fill='x', padx=5, pady=(0, 0))

    def _on_send(self, event=None, preset_message=None):
        """Modified to handle preset messages from quick action buttons"""
        message = preset_message if preset_message else self.input_field.get().strip()
        if message:
            print(message)
            # Disable input and button while processing
            self.input_field.configure(state='disabled')
            self.send_button.configure(state='disabled')
            
            # Display user message
            self.display_message(f"USER: {message}", "user")
            self.input_field.delete(0, tk.END)
            
            # Start processing in separate thread
            threading.Thread(
                target=self._process_message,
                args=(message,),
                daemon=True
            ).start()

    def _process_message(self, message):
        # Call the chat callback and get response
        response = self.chat_callback(message)
        # Queue the response for display
        self.msg_queue.put(("F.R.E.D", response))
        
        # Re-enable input and button
        self.root.after(0, lambda: self.input_field.configure(state='normal'))
        self.root.after(0, lambda: self.send_button.configure(state='normal'))
        self.root.after(0, lambda: self.input_field.focus_set())

    def display_message(self, message, sender):
        self.chat_display.insert(tk.END, message + "\n\n")
        self.chat_display.see(tk.END)
        
        # Get the last message position
        last_line_start = self.chat_display.get("end-3c linestart", "end-1c")
        tag_name = f"tag_{sender}"
        
        # Add tag to the message
        self.chat_display.tag_add(tag_name, 
                                f"end-{len(last_line_start)+2}c linestart", 
                                "end-2c")
        
        # Configure tag colors and fonts
        if sender == "user":
            self.chat_display.tag_config(tag_name, 
                                       foreground="#ff9100",  # Iron Man orange for user
                                       font=("Rajdhani", 10, "bold"))
        else:
            self.chat_display.tag_config(tag_name, 
                                       foreground="#00bfff",  # Bright blue for Jarvis
                                       font=("Rajdhani", 10, "bold"))

    def _start_msg_checker(self):
        """Check for new messages in the queue and display them"""
        try:
            while not self.msg_queue.empty():
                sender, message = self.msg_queue.get_nowait()
                self.display_message(f"{sender}: {message}", sender.lower())
        finally:
            self.root.after(100, self._start_msg_checker)

    def run(self):
        # Set window icon and make it semi-transparent
        self.root.attributes('-alpha', 0.95)  # Slight transparency
        
        # Center the window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')
        
        self.root.mainloop()