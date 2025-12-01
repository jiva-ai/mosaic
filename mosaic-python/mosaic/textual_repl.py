"""Textual-based REPL for Mosaic orchestrator."""

from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Footer, Header, Input, Static
from textual.binding import Binding

from mosaic.repl_commands import (
    execute_calcd,
    execute_hb,
    execute_help,
    execute_rhb,
    execute_shb,
    process_command,
)


class MosaicREPL(App):
    """Textual-based REPL for Mosaic orchestrator."""

    CSS = """
    Screen {
        background: $surface;
    }
    
    #output {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }
    
    #input {
        dock: bottom;
        height: 3;
        border: solid $primary;
    }
    
    .command-output {
        padding: 1;
        margin: 1;
    }
    
    .error {
        color: $error;
    }
    
    .success {
        color: $success;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+d", "quit", "Quit", priority=True),
    ]

    def __init__(self, *args, **kwargs):
        """Initialize the REPL."""
        super().__init__(*args, **kwargs)
        self.command_history: list[str] = []
        self.history_index = -1

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header(show_clock=True)
        yield ScrollableContainer(Static("", id="output"), id="output-container")
        yield Input(placeholder="mosaic> ", id="input")
        yield Footer()

    def on_mount(self) -> None:
        """Called when app starts."""
        self.title = "Mosaic Orchestrator REPL"
        self.sub_title = "Type 'help' for available commands"
        self.query_one("#input", Input).focus()
        self._append_output("Mosaic Orchestrator REPL\nType 'help' for available commands\n")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle command input."""
        command = event.value.strip()
        if not command:
            return

        # Add to history
        self.command_history.append(command)
        self.history_index = len(self.command_history)

        # Echo command
        self._append_output(f"mosaic> {command}\n")

        # Handle exit commands
        if command.lower() in ("exit", "quit", "q"):
            self.exit()
            return

        # Process command
        self._process_command(command)

        # Clear input
        event.input.value = ""

    def _append_output(self, text: str) -> None:
        """Append text to output area."""
        output = self.query_one("#output", Static)
        current_text = str(output.renderable) if output.renderable else ""
        new_text = current_text + text
        output.update(new_text)
        # Auto-scroll to bottom
        output_container = self.query_one("#output-container", ScrollableContainer)
        output_container.scroll_end(animate=False)

    def _process_command(self, command: str) -> None:
        """Process a command and display results."""
        process_command(command, self._append_output)

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


def start_textual_repl() -> None:
    """Start the Textual-based REPL."""
    app = MosaicREPL()
    app.run()

