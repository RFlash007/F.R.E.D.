import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import threading
from queue import Queue
from datetime import datetime
import json
import os

class ChatUI:
    def __init__(self, chat_callback):
        self.root = tk.Tk()
        self.root.title("F.R.E.D. Interface")
        self.root.geometry("1200x800")
        self.root.configure(bg='#0a0f18')
        
        # Message queue for thread-safe UI updates
        self.msg_queue = Queue()
        self.chat_callback = chat_callback
        self.conversation_history = []
        
        # Create styles
        self._create_styles()
        self._create_widgets()
        self._setup_layout()
        self._start_msg_checker()
        
        # Keyboard shortcuts
        self.root.bind("<Control-s>", lambda e: self._save_conversation())
        self.root.bind("<Control-c>", lambda e: self._clear_chat())
        self.root.bind("<Control-q>", lambda e: self.root.quit())
    
    def _create_styles(self):
        style = ttk.Style()
        style.configure('Neural.TFrame', background='#0a0f18')
        style.configure('TSeparator', background='#00bfff')
        
        style.configure('Action.TButton',
                       font=('Rajdhani', 11),
                       padding=8,
                       background='#1a2433',
                       foreground='#00bfff')
        
        style.configure('Neural.TEntry',
                       fieldbackground='#1a2433',
                       foreground='#00bfff',
                       insertcolor='#00bfff',
                       borderwidth=0)
        
        style.map('Action.TButton',
                 background=[('active', '#2a3443')],
                 foreground=[('active', '#00dfff')])
    
    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0, bg='#1a2433', fg='#00bfff',
                           activebackground='#2a3443', activeforeground='#00dfff')
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Conversation", command=self._save_conversation)
        file_menu.add_command(label="Clear Chat", command=self._clear_chat)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        edit_menu = tk.Menu(menubar, tearoff=0, bg='#1a2433', fg='#00bfff',
                           activebackground='#2a3443', activeforeground='#00dfff')
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Copy", command=lambda: self.chat_display.event_generate("<<Copy>>"))
        edit_menu.add_command(label="Paste", command=lambda: self.input_field.event_generate("<<Paste>>"))
    
    def _create_header(self):
        self.header_frame = ttk.Frame(self.main_frame, style='Neural.TFrame')
        
        self.system_status = ttk.Label(
            self.header_frame,
            text="⚡ NEURAL LINK ACTIVE",
            foreground='#00bfff',
            font=('Rajdhani', 12, 'bold'),
            style='Neural.TFrame'
        )
        
        self.time_label = ttk.Label(
            self.header_frame,
            text="",
            foreground='#00bfff',
            font=('Rajdhani', 12),
            style='Neural.TFrame'
        )
        self._update_time()
    
    def _create_widgets(self):
        self.main_frame = ttk.Frame(self.root, padding="20", style='Neural.TFrame')
        
        self._create_menu()
        self._create_header()
        
        self.chat_display = scrolledtext.ScrolledText(
            self.main_frame,
            wrap=tk.WORD,
            width=80,
            height=30,
            font=("Consolas", 11),
            bg='#0d1520',
            fg='#e0e0e0',
            insertbackground='#00bfff',
            relief='flat',
            borderwidth=0,
            padx=10,
            pady=10
        )
        
        self.status_frame = ttk.Frame(self.main_frame, style='Neural.TFrame')
        self.status_bar = ttk.Label(
            self.status_frame,
            text="SYSTEM READY",
            style='Neural.TFrame',
            foreground='#00bfff',
            font=('Rajdhani', 10)
        )
        
        self.bottom_frame = ttk.Frame(self.main_frame, style='Neural.TFrame')
        self.input_frame = ttk.Frame(self.bottom_frame, style='Neural.TFrame')
        
        self.input_field = ttk.Entry(
            self.input_frame,
            width=70,
            font=("Consolas", 11),
            style='Neural.TEntry'
        )
        self.input_field.bind("<Return>", self._on_send)
        
        self.send_button = tk.Button(
            self.input_frame,
            text="TRANSMIT",
            command=self._on_send,
            font=('Rajdhani', 11, 'bold'),
            bg='#00bfff',
            fg='#0a0f18',
            activebackground='#00dfff',
            activeforeground='#0a0f18',
            relief='flat',
            padx=20,
            pady=8,
            cursor='hand2'
        )
        
        self.quick_actions_frame = ttk.Frame(self.main_frame, style='Neural.TFrame')
        self._create_quick_actions()
    
    def _create_quick_actions(self):
        buttons = [
            ("📡 SYSTEM", lambda: self._on_send(None, "Show system status")),
            ("📰 NEWS", lambda: self._on_send(None, "Show me the latest news")),
            ("🔄 RESET", self._clear_chat)
        ]
        
        for text, command in buttons:
            btn = ttk.Button(
                self.quick_actions_frame,
                text=text,
                command=command,
                style='Action.TButton'
            )
            btn.pack(side='left', padx=5)
    
    def _setup_layout(self):
        self.main_frame.pack(expand=True, fill='both')
        
        self.header_frame.pack(fill='x', pady=(0, 10))
        self.system_status.pack(side='left')
        self.time_label.pack(side='right')
        
        self.chat_display.pack(expand=True, fill='both', padx=5, pady=(0, 10))
        self.quick_actions_frame.pack(fill='x', padx=5, pady=(0, 10))
        
        self.bottom_frame.pack(fill='x', padx=5)
        self.input_frame.pack(fill='x', expand=True)
        self.input_field.pack(side='left', expand=True, fill='x', padx=(0, 10))
        self.send_button.pack(side='right')
        
        self.status_frame.pack(fill='x', pady=(10, 0))
        self.status_bar.pack(side='left')
        
        self.input_field.focus_set()
    
    def _on_send(self, event=None, preset_message=None):
        message = preset_message if preset_message else self.input_field.get().strip()
        if message:
            self.status_bar.config(text="Processing...")
            self.input_field.configure(state='disabled')
            self.send_button.configure(state='disabled')
            
            self.display_message(f"Ian: {message}", "user")
            self.conversation_history.append({"role": "user", "content": message})
            self.input_field.delete(0, tk.END)
            
            threading.Thread(
                target=self._process_message,
                args=(message,),
                daemon=True
            ).start()
    
    def _process_message(self, message):
        try:
            response = self.chat_callback(message)
            self.msg_queue.put(("F.R.E.D", response))
            self.conversation_history.append({"role": "assistant", "content": response})
            self.status_bar.config(text="Ready")
        except Exception as e:
            self.msg_queue.put(("ERROR", f"An error occurred: {str(e)}"))
            self.status_bar.config(text="Error occurred")
        finally:
            self.root.after(0, lambda: self.input_field.configure(state='normal'))
            self.root.after(0, lambda: self.send_button.configure(state='normal'))
            self.root.after(0, lambda: self.input_field.focus_set())
    
    def _save_conversation(self, event=None):
        if not self.conversation_history:
            self.status_bar.config(text="No conversation to save")
            return
        
        filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.chat_display.get(1.0, tk.END))
            self.status_bar.config(text=f"Conversation saved to {filename}")
        except Exception as e:
            self.status_bar.config(text=f"Error saving conversation: {str(e)}")
    
    def _clear_chat(self, event=None):
        if messagebox.askyesno("Clear Chat", "Are you sure you want to clear the chat?"):
            self.chat_display.delete(1.0, tk.END)
            self.conversation_history = []
            self.status_bar.config(text="Chat cleared")
    
    def display_message(self, message, sender):
        self.chat_display.insert(tk.END, message + "\n\n")
        self.chat_display.see(tk.END)
        
        last_line_start = self.chat_display.get("end-3c linestart", "end-1c")
        tag_name = f"tag_{sender}"
        
        self.chat_display.tag_add(tag_name, 
                                f"end-{len(last_line_start)+2}c linestart", 
                                "end-2c")
        
        if sender == "user":
            self.chat_display.tag_config(tag_name, 
                                       foreground="#ff9100",
                                       font=("Consolas", 11, "bold"))
        elif sender == "error":
            self.chat_display.tag_config(tag_name,
                                       foreground="#ff4444",
                                       font=("Consolas", 11, "bold"))
        else:
            self.chat_display.tag_config(tag_name, 
                                       foreground="#00bfff",
                                       font=("Consolas", 11, "bold"))
    
    def _update_time(self):
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.config(text=f"🕒 {current_time}")
        self.root.after(1000, self._update_time)
    
    def _start_msg_checker(self):
        try:
            while not self.msg_queue.empty():
                sender, message = self.msg_queue.get_nowait()
                self.display_message(f"{sender}: {message}", sender.lower())
        finally:
            self.root.after(100, self._start_msg_checker)
    
    def run(self):
        self.root.attributes('-alpha', 0.97)
        
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')
        
        self.root.mainloop()