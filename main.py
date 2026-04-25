"""
GraphIDYOM-powered TCP MIDI server

Sends newline-delimited JSON messages to any connected TCP client.
Uses a pretrained GraphIDYOM model to generate intelligent note predictions.
Message example: {"type":"midi","cmd":"note_on","note":60,"vel":100}

Run: python main.py
"""

import socket
import threading
import time
import json
import random
import sys
import os
import math
import statistics
import signal
from pathlib import Path
from collections import deque
from urllib.parse import unquote, urlparse

try:
	import matplotlib.pyplot as plt
	import matplotlib
	matplotlib.use('Agg')  # Non-interactive backend
	HAS_MATPLOTLIB = True
except ImportError:
	HAS_MATPLOTLIB = False

# Enable ANSI colors on supported terminals (Windows 10+, macOS, Linux)
# On Windows, enable virtual terminal processing for ANSI escape codes
if sys.platform == 'win32':
	try:
		os.system('color')  # Enable color mode on Windows
	except Exception:
		pass

# Global log buffer (circular queue for recent logs)
log_buffer = deque(maxlen=50)  # Keep last 50 log lines
log_buffer_lock = threading.Lock()

# Log display state
HEADER_HEIGHT = 17  # Number of lines for the header
LOG_HEIGHT = 30     # Number of lines available for logs
current_log_line = HEADER_HEIGHT + 1  # Start below header

# ANSI Color codes
class Colors:
	HEADER = '\033[95m'
	BLUE = '\033[94m'
	CYAN = '\033[96m'
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	RED = '\033[91m'
	WHITE = '\033[97m'
	GRAY = '\033[90m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	END = '\033[0m'

# Global server state for graceful shutdown
server_running = True
active_connections = []  # List of active client connections
active_connections_lock = threading.Lock()  # Thread-safe lock for connection list

def graceful_shutdown(signum=None, frame=None):
	"""Handle graceful server shutdown on Ctrl+C or system signals."""
	global server_running
	server_running = False
	
	print(f"\n\n{warning('Shutting down server...')}", flush=True)
	
	# Close all active client connections
	with active_connections_lock:
		for conn in active_connections:
			try:
				conn.close()
			except Exception:
				pass
		active_connections.clear()
	
	print(f"{success('All connections closed')}", flush=True)
	print(f"{info('Server stopped')}", flush=True)
	sys.exit(0)

# Register signal handlers for graceful shutdown (works on Windows, macOS, Linux)
signal.signal(signal.SIGINT, graceful_shutdown)  # Ctrl+C
if sys.platform != 'win32':
	signal.signal(signal.SIGTERM, graceful_shutdown)  # SIGTERM on Unix systems

def ascii_art_title():
	"""Display ASCII art title for Iridyom"""
	title = f"""{Colors.CYAN}{Colors.BOLD}
 ___________ ___________       ________  ___
|_   _| ___ \_   _|  _  \     |  _  |  \/  |
  | | | |_/ / | | | | | |_   _| | | | .  . |
  | | |    /  | | | | | | | | | | | | |\/| |
 _| |_| |\ \ _| |_| |/ /| |_| \ \_/ / |  | |
 \___/\_| \_|\___/|___/  \__, |\___/\_|  |_/
                          __/ |             
                         |___/              
{Colors.END}"""
	return title

def section_header(text):
	"""Format a section header"""
	return f"\n{Colors.BOLD}{Colors.BLUE}▸ {text}{Colors.END}"

def success(text):
	"""Format success message"""
	return f"{Colors.GREEN}✓{Colors.END} {text}"

def info(text):
	"""Format info message"""
	return f"{Colors.CYAN}ℹ{Colors.END} {text}"

def warning(text):
	"""Format warning message"""
	return f"{Colors.YELLOW}⚠{Colors.END} {text}"

def error(text):
	"""Format error message"""
	return f"{Colors.RED}✗{Colors.END} {text}"

def emphasis(text):
	"""Format emphasized text"""
	return f"{Colors.BOLD}{Colors.YELLOW}{text}{Colors.END}"

def param_value(text):
	"""Format parameter value"""
	return f"{Colors.CYAN}{text}{Colors.END}"

def separator(char='─', length=60):
	"""Create a separator line"""
	return f"{Colors.GRAY}{char * length}{Colors.END}"

def add_log(message):
	"""Add message to log buffer and display it in fixed position."""
	global current_log_line
	
	with log_buffer_lock:
		log_buffer.append(message)
		
		# Use ANSI cursor positioning to write at current_log_line
		# Format: ESC [ row ; col H
		row = current_log_line
		col = 1
		
		# Move cursor to position, print message, then move to next position
		sys.stdout.write(f'\033[{row};{col}H')  # Move cursor
		# For debug messages, don't truncate; for normal messages keep 120 char limit
		display_msg = message if 'DEBUG:' in message else message[:120]
		sys.stdout.write(display_msg)
		sys.stdout.write('\033[K')               # Clear rest of line
		sys.stdout.flush()
		
		# Move to next line, wrapping around if needed
		current_log_line += 1
		if current_log_line > HEADER_HEIGHT + LOG_HEIGHT:
			current_log_line = HEADER_HEIGHT + 1  # Wrap to top of log area

# Add lib folder to Python path
script_dir = Path(__file__).parent
lib_dir = script_dir / "lib"
sys.path.insert(0, str(lib_dir))

# Import from lib
from app_core import GraphIDYOMAppService

HOST = '127.0.0.1'
PORT = 5005

# GraphIDYOM configuration
DATASET_NAME = 'bach'  # The trained Bach chorales model (default)
MODEL_FOLDER = 'order_5_augmented_true'  # The model folder within the dataset (order 5 with augmentation)
MAX_HISTORY_LENGTH = 20  # Keep last N notes in history
USE_PROBABILISTIC_SELECTION = True  # If True, sample from distribution; if False, use argmax
TEMPERATURE = 1.0  # Higher = more random, lower = more deterministic (only for probabilistic)

# MIDI range constraints
MIN_MIDI = 48  # C3
MAX_MIDI = 84  # C6

# Note timing (tempo-based)
DEFAULT_TEMPO = 120  # BPM
NOTE_DURATION_DIVISION = 1/4  # Note value (1/4 = quarter note, 1/8 = eighth note, etc.)
NOTE_INTERVAL_DIVISION = 1/4  # Note value between onsets

# Track last tempo to avoid duplicate logging
last_tempo = DEFAULT_TEMPO

# Global GraphIDYOM service and model
graphidyom_service = None
graphidyom_model_id = None
available_models = []  # List of available pretrained models

# Mutable generation parameters (can be changed via messages from Max)
generation_params = {
	'temperature': TEMPERATURE,
	'min_midi': MIN_MIDI,
	'max_midi': MAX_MIDI,
	'use_probabilistic': USE_PROBABILISTIC_SELECTION,
	'tempo': DEFAULT_TEMPO,  # BPM
	'note_duration_division': NOTE_DURATION_DIVISION,  # Note value (e.g., 1/4, 1/8)
	'note_interval_division': NOTE_INTERVAL_DIVISION,  # Note value between onsets
	'max_history': MAX_HISTORY_LENGTH,
	'sequencer_running': False,
	'mode': 'generate'  # 'generate', 'interact', or 'train'
}

# Training data storage (for train mode)
training_sequences = []  # List of MIDI sequences for training
current_training_sequence = []  # Current sequence being recorded

# MIDI file training storage
pending_training_files = []  # List of MIDI file paths for training
training_options = {
	'dataset_name': 'custom_model',  # Name for the trained model
	'orders': (1, 2, 3, 4, 5),  # Markov orders
	'augmented': True,  # Use data augmentation
	'viewpoint': 'midi',  # Viewpoint preset
}
is_training = False  # Flag to prevent multiple trainings

# Track last analyzed file for MIDI generation
last_analyzed_file = None  # Path to the last file analyzed with analyse_and_generate

# Global IC history for plotting
ic_history = []  # List of IC values for export/plotting
ic_notes_history = []  # List of (note, ic) tuples for pianoroll background
ic_history_lock = threading.Lock()  # Thread-safe access to IC history


# Mode mapping: map mode values to display names
MODE_NAMES = {
	0: "Sequencer ",
	1: "Assistant ",
	2: "Continue",
	3: "Variate",
	4: "Train",
	'generate': "SEQUENCER",
	'interact': "ASSISTANT",
	'train': "TRAIN",
}

def get_mode_display_name(mode_val):
	"""Convert mode value/name to display name."""
	if mode_val in MODE_NAMES:
		return MODE_NAMES[mode_val]
	return str(mode_val).upper()


def get_programdata_path():
	"""Get ProgramData path (Windows) or home directory (macOS/Linux)."""
	if sys.platform == 'win32':
		return Path(os.getenv('PROGRAMDATA')) / "GraphIDYOM"
	else:
		return Path.home() / ".graphidyom"


def ensure_programdata_directories():
	"""Ensure ProgramData directories exist for models and training output."""
	programdata = get_programdata_path()
	
	# Create main directory
	programdata.mkdir(parents=True, exist_ok=True)
	
	# Create subdirectories
	(programdata / "datasets").mkdir(exist_ok=True)
	(programdata / "trained_models").mkdir(exist_ok=True)
	
	return programdata


def _strip_wrapping_quotes(value):
	"""Strip quote wrappers often added by Max around paths with spaces."""
	value = str(value or '').strip().replace('\x00', '')
	while len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
		value = value[1:-1].strip()
	return value


def _client_path_to_string(path_value):
	"""Normalize path text received over the Max/Node bridge."""
	if isinstance(path_value, (list, tuple)):
		raw = ' '.join(str(part) for part in path_value if part is not None)
	else:
		raw = str(path_value)
	raw = _strip_wrapping_quotes(raw)
	if not raw:
		return raw

	if raw.lower().startswith('file://'):
		parsed = urlparse(raw)
		url_path = unquote(parsed.path)
		if sys.platform == 'win32' and len(url_path) >= 3 and url_path[0] == '/' and url_path[2] == ':':
			url_path = url_path[1:]
		if parsed.netloc and parsed.netloc.lower() != 'localhost':
			if sys.platform == 'win32':
				url_path = f"//{parsed.netloc}{url_path}"
		raw = url_path
	elif '%' in raw:
		try:
			raw = unquote(raw)
		except Exception:
			pass

	return raw.replace('\\ ', ' ')


def _looks_like_windows_drive_path(raw):
	return len(raw) >= 3 and raw[1] == ':' and raw[0].isalpha() and raw[2] in ('/', '\\')


def _mac_volume_path_candidates(raw):
	"""Return candidates for Max's older HFS-style macOS paths."""
	if sys.platform != 'darwin' or ':' not in raw or _looks_like_windows_drive_path(raw):
		return []
	if raw.startswith(('/', '~')):
		return []

	volume, rest = raw.split(':', 1)
	if not volume or '/' in volume or '\\' in volume:
		return []

	rest = rest.replace(':', '/')
	if rest.startswith('/'):
		return [Path(rest), Path('/Volumes') / volume / rest.lstrip('/')]
	return [Path('/') / rest, Path('/Volumes') / volume / rest]


def _windows_to_posix_path_candidates(raw):
	"""Map common dragged Windows-style user paths when running on macOS/Linux."""
	if os.name == 'nt' or not _looks_like_windows_drive_path(raw):
		return []

	without_drive = raw[2:].replace('\\', '/')
	if without_drive.startswith('/Users/') or without_drive.startswith('/Volumes/'):
		return [Path(without_drive)]
	return []


def normalize_client_path(path_value):
	"""Return an OS-native Path for paths sent by Max/Ableton."""
	raw = _client_path_to_string(path_value)
	candidates = _mac_volume_path_candidates(raw) + _windows_to_posix_path_candidates(raw) + [Path(raw)]

	seen = set()
	unique_candidates = []
	for candidate in candidates:
		key = str(candidate)
		if key not in seen:
			seen.add(key)
			unique_candidates.append(candidate)

	for candidate in unique_candidates:
		try:
			resolved = candidate.expanduser().resolve()
		except Exception:
			resolved = candidate.expanduser()
		if resolved.exists():
			return resolved

	first = unique_candidates[0].expanduser()
	try:
		return first.resolve()
	except Exception:
		return first


def parse_midi_file(file_path):
	"""Parse MIDI file and extract note sequence with durations and intervals.
	
	Returns:
		List of note events where each event is a dict: {'pitch': midi_note, 'duration': 0-16, 'interval': 0-16}
		Or None if file cannot be parsed.
	"""
	try:
		import mido
	except ImportError:
		add_log(f"⚠ mido library not installed. Install with: pip install mido")
		return None
	
	try:
		midi_path = normalize_client_path(file_path)
		mid = mido.MidiFile(str(midi_path))
		note_events = []
		note_on_stack = []  # Track open note_on events
		
		# Find tick/time quantization parameters
		ticks_per_beat = mid.ticks_per_beat
		min_duration = float('inf')
		max_duration = 0
		durations = []
		intervals = []
		last_note_time = 0
		current_time = 0
		
		# First pass: collect all note events with raw timing
		raw_notes = []
		for track in mid.tracks:
			for msg in track:
				current_time += msg.time
				if msg.type == 'note_on' and msg.velocity > 0:
					note_on_stack.append({'pitch': msg.note, 'time': current_time})
				elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)):
					if note_on_stack:
						note_start = note_on_stack.pop()
						duration = current_time - note_start['time']
						interval = note_start['time'] - last_note_time if last_note_time > 0 else 0
						
						raw_notes.append({
							'pitch': note_start['pitch'],
							'duration': duration,
							'interval': interval,
							'time': note_start['time']
						})
						
						if duration > 0:
							durations.append(duration)
							min_duration = min(min_duration, duration)
							max_duration = max(max_duration, duration)
						if interval > 0:
							intervals.append(interval)
						
						last_note_time = note_start['time']
		
		if not raw_notes:
			return None
		
		# Quantize durations to 0-16 range
		if max_duration > min_duration:
			duration_scale = 16.0 / (max_duration - min_duration)
		else:
			duration_scale = 1.0
		
		# Quantize intervals to 0-16 range
		max_interval = max(intervals) if intervals else 1
		if max_interval > 0:
			interval_scale = 16.0 / max_interval
		else:
			interval_scale = 1.0
		
		# Second pass: normalize and quantize
		for note in raw_notes:
			normalized_duration = int((note['duration'] - min_duration) * duration_scale) if max_duration > min_duration else 8
			normalized_duration = max(0, min(16, normalized_duration))  # Clamp to 0-16
			
			normalized_interval = int(note['interval'] * interval_scale) if note['interval'] > 0 else 0
			normalized_interval = max(0, min(16, normalized_interval))  # Clamp to 0-16
			
			note_events.append({
				'pitch': note['pitch'],
				'duration': normalized_duration,
				'interval': normalized_interval
			})
		
		return note_events if note_events else None
	except Exception as e:
		add_log(f"⚠ Error parsing MIDI file: {e}")
		return None


def generate_and_save_next_note(input_file, output_folder, addr, conn, output_filename=None, sampling_strategies=None):
	"""Generate next note(s) and save extended MIDI file.
	
	Args:
		input_file: Path to input MIDI file
		output_folder: Path to output folder
		addr: Client address
		conn: Connection socket
		output_filename: Optional output filename
		sampling_strategies: Optional list of 8 integers (0/1/2) for each generated note
			- 0 = most probable note (top prediction)
			- 1 = average probability note (middle)
			- 2 = least probable note (bottom)
			If provided, generates 8 notes; otherwise generates 1 note (old behavior)
	"""
	global graphidyom_service, graphidyom_model_id
	
	try:
		import mido
	except ImportError:
		response = {"type": "error", "message": "mido library not installed. Install with: pip install mido"}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		return
	
	input_path = normalize_client_path(input_file)
	output_path = normalize_client_path(output_folder)

	# Check if file exists with alternative methods
	if not input_path.exists():
		response = {"type": "error", "message": f"Input file not found: {input_path}"}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		add_log(f"  [{addr[1]}] Error: Input file not found: {input_path}")
		return
	
	if not input_path.suffix.lower() in ['.mid', '.midi']:
		response = {"type": "error", "message": f"Not a MIDI file: {input_path}"}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		return
	
	# Create output folder if needed
	try:
		output_path.mkdir(parents=True, exist_ok=True)
	except Exception as e:
		response = {"type": "error", "message": f"Could not create output folder: {e}"}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		return
	
	# Parse input MIDI file - returns list of {'pitch', 'duration', 'interval'}
	note_data = parse_midi_file(str(input_path))
	if note_data is None or len(note_data) == 0:
		response = {"type": "error", "message": "Could not parse MIDI file or no notes found"}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		return
	
	# Extract just pitches for history
	notes = [n['pitch'] for n in note_data]
	
	# Check if model is loaded
	if graphidyom_model_id is None:
		response = {"type": "error", "message": "No GraphIDYOM model loaded"}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		return
	
	# Determine number of notes to generate and strategies
	if sampling_strategies is None or len(sampling_strategies) == 0:
		# Old behavior: generate 1 note
		strategies = [0]  # Default to most probable
	else:
		# New behavior: generate 8 notes with provided strategies
		strategies = sampling_strategies

	# Load original MIDI file and extract timing information
	try:
		mid = mido.MidiFile(str(input_path))
		
		def _pick_melody_track(midi_file):
			"""Pick the track that most likely contains the melody notes."""
			if not midi_file.tracks:
				return None
			if len(midi_file.tracks) == 1:
				return midi_file.tracks[0]
			best_track = None
			best_count = -1
			for t in midi_file.tracks:
				count = 0
				for m in t:
					if m.type == 'note_on' and getattr(m, 'velocity', 0) > 0:
						count += 1
				if count > best_count:
					best_count = count
					best_track = t
			return best_track
		
		def _extract_note_timing(track):
			"""Extract absolute onset/duration and a reasonable interval grid from a MIDI track."""
			from collections import defaultdict, deque
			abs_time = 0
			open_notes = defaultdict(deque)  # (pitch, channel) -> deque[start_abs_time]
			pairs = []  # list of (start_abs, end_abs, pitch, channel, velocity)
			last_note_on_channel = 0
			last_note_on_velocity = 100
			
			for msg in track:
				abs_time += msg.time
				if msg.type == 'note_on' and getattr(msg, 'velocity', 0) > 0:
					channel = getattr(msg, 'channel', 0)
					open_notes[(msg.note, channel)].append((abs_time, msg.velocity))
					last_note_on_channel = channel
					last_note_on_velocity = msg.velocity
				elif msg.type == 'note_off' or (msg.type == 'note_on' and getattr(msg, 'velocity', 0) == 0):
					channel = getattr(msg, 'channel', 0)
					key = (msg.note, channel)
					if open_notes[key]:
						start_abs, start_vel = open_notes[key].popleft()
						end_abs = abs_time
						if end_abs > start_abs:
							pairs.append((start_abs, end_abs, msg.note, channel, start_vel))
			
			pairs.sort(key=lambda x: x[0])
			onsets = [p[0] for p in pairs]
			durations = [p[1] - p[0] for p in pairs]
			intervals = [onsets[i] - onsets[i - 1] for i in range(1, len(onsets))]
			last_onset = onsets[-1] if onsets else None
			last_end = max((p[1] for p in pairs), default=None)
			
			return {
				'track_end_abs': abs_time,
				'pairs': pairs,
				'onsets': onsets,
				'durations': durations,
				'intervals': intervals,
				'last_onset': last_onset,
				'last_end': last_end,
				'last_channel': last_note_on_channel,
				'last_velocity': last_note_on_velocity,
			}
		
		# Pick a melody track and remove end_of_track so we can append validly
		melody_track = _pick_melody_track(mid)
		if melody_track is None:
			raise ValueError('No MIDI tracks found')
		
		tail_silence_ticks = 0
		if len(melody_track) > 0 and melody_track[-1].type == 'end_of_track':
			tail_silence_ticks = melody_track[-1].time
			melody_track.pop()
		
		timing = _extract_note_timing(melody_track)
		
		# Build a per-note timing template from the last N notes (N = min(8, total notes))
		all_pairs = timing['pairs']
		N = min(8, max(1, len(all_pairs)))
		template_pairs = all_pairs[-N:]
		
		# Per-note durations from the template
		template_durations_ticks = [int(max(1, p[1] - p[0])) for p in template_pairs]
		
		# Inter-onset intervals within the template
		if N > 1:
			template_iois = [int(max(1, template_pairs[i + 1][0] - template_pairs[i][0])) for i in range(N - 1)]
		else:
			# Single note: use its duration as the IOI (legato / one-note pattern)
			template_iois = [template_durations_ticks[0]]
		
		# IOI from the last existing note to the first generated note
		if timing['intervals']:
			ioi_to_first_gen = int(max(1, timing['intervals'][-1]))
		else:
			ioi_to_first_gen = template_durations_ticks[-1]  # fallback: legato
		
		last_onset = timing['last_onset']
		track_end_abs = timing['track_end_abs']
		
		if last_onset is None:
			desired_first_onset = track_end_abs
		else:
			desired_first_onset = max(track_end_abs, last_onset + ioi_to_first_gen)
		
		delta_to_first = max(0, desired_first_onset - track_end_abs)
		
		add_log(
			f"  [{addr[1]}] Continue timing: {N}-note template, "
			f"durations={template_durations_ticks}, iois={template_iois}, first_delta={delta_to_first} ticks"
		)
		
		# Channel/velocity defaults from last seen note_on
		append_channel = int(timing.get('last_channel', 0) or 0)
		append_velocity = int(timing.get('last_velocity', 100) or 100)
		
		# Generate notes sequentially
		generated_notes = []
		current_history = list(notes)  # Start with the original notes
		
		for step, strategy in enumerate(strategies):
			try:
				# Get predictions (top_k=128 to have enough for sampling strategies)
				result = graphidyom_service.predict_next(
					model_id=graphidyom_model_id,
					midi_history=current_history,
					top_k=128,
					output="midi",
					reset_stm=True,
				)
				predictions = result.get("predictions", [])
				
				
				# Filter by MIDI range
				min_note = min(generation_params['min_midi'], generation_params['max_midi'])
				max_note = max(generation_params['min_midi'], generation_params['max_midi'])
				valid_preds = [p for p in predictions if min_note <= p.get("midi", 0) <= max_note]
				
				if not valid_preds:
					valid_preds = predictions  # Fallback to all predictions
				
				# Select note based on strategy
				if strategy == 0:
					# Most probable (first)
					selected = valid_preds[0]
				elif strategy == 1:
					# Average probability (middle)
					mid_idx = len(valid_preds) // 2
					selected = valid_preds[mid_idx]
				elif strategy == 2:
					# Least probable (last)
					selected = valid_preds[-1]
				else:
					# Default to first
					selected = valid_preds[0]
				
				next_note = selected["midi"]
				next_prob = selected.get("prob", 0.0)
				
				# Duration for this note taken from the per-note template (cycling)
				note_duration_ticks = template_durations_ticks[step % N]
				
				# Add to generated notes
				generated_notes.append({
					"note": next_note,
					"prob": next_prob,
					"strategy": strategy,
					"duration": note_duration_ticks
				})
				
				# Update history for next prediction
				current_history.append(next_note)
				
				# Calculate IC for export
				next_note_ic = -math.log2(next_prob) if next_prob > 0 else None
				if next_note_ic is not None:
					with ic_history_lock:
						ic_history.append(next_note_ic)
						ic_notes_history.append(next_note)
				
			except Exception as e:
				add_log(f"  [{addr[1]}] Error during step {step+1}: {e}")
				continue
		
		
		# Append all generated notes using per-note timing from the template (last N notes).
		# NOTE: MIDI message times are DELTAS.
		for i, gen_note in enumerate(generated_notes):
			dur_i = template_durations_ticks[i % N]
			if i == 0:
				note_on_delta = delta_to_first
			else:
				# gap = IOI of previous note - duration of previous note
				prev_ioi = template_iois[(i - 1) % len(template_iois)]
				prev_dur = template_durations_ticks[(i - 1) % N]
				note_on_delta = max(0, prev_ioi - prev_dur)
			note_off_delta = dur_i
			melody_track.append(
				mido.Message('note_on', note=int(gen_note["note"]), velocity=append_velocity, time=int(note_on_delta), channel=append_channel)
			)
			melody_track.append(
				mido.Message('note_off', note=int(gen_note["note"]), velocity=0, time=int(note_off_delta), channel=append_channel)
			)
		
		# Restore end_of_track (preserve the original trailing silence, if any)
		melody_track.append(mido.MetaMessage('end_of_track', time=int(tail_silence_ticks)))
		
		# Generate output filename
		if output_filename is None:
			# Auto-generate filename if not provided
			input_stem = input_path.stem
			output_filename = f"{input_stem}_extended.mid"
		
		output_file = output_path / output_filename
		
		# Save the new MIDI file
		mid.save(str(output_file))
		
		add_log(f"  [{addr[1]}] Saved extended MIDI with {len(generated_notes)} new notes: {output_filename}")
		
		# Send success response
		response = {
			"type": "generate_midi_file_result",
			"status": "success",
			"input_file": input_path.name,
			"output_file": output_filename,
			"output_folder": str(output_path),
			"notes_in_input": len(notes),
			"notes_generated": len(generated_notes),
			"generated_notes": [
				{
					"note": g["note"],
					"prob": f"{g['prob']:.1%}",
					"strategy": ["most_prob", "avg_prob", "least_prob"][g["strategy"]]
				}
				for g in generated_notes
			],
			"full_path": str(output_file)
		}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		add_log(f"  [{addr[1]}] Generate complete! File saved to: {output_file}")
		_auto_export_ic_plot_for_midi(addr, conn, output_file)
		
	except Exception as e:
		response = {
			"type": "error",
			"message": f"Could not save MIDI file: {e}"
		}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		add_log(f"  [{addr[1]}] Error saving MIDI file: {e}")


def variate_midi_file(output_folder, part, surpriseness, addr, conn):
	"""Modify notes in a specific part of the MIDI file based on surpriseness.
	
	Args:
		output_folder: Path to output folder where variated MIDI will be saved
		part: Which quarter of the file to modify (0, 1, 2, or 3)
		surpriseness: 0-127 value determining sampling strategy (0-42=most prob, 43-84=avg, 85-127=least prob)
		addr: Client address
		conn: Connection socket
	"""
	global graphidyom_service, graphidyom_model_id, last_analyzed_file
	
	try:
		import mido
	except ImportError:
		response = {"type": "error", "message": "mido library not installed"}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		return
	
	# Validate inputs
	if not isinstance(part, int) or part not in [0, 1, 2, 3]:
		add_log(f"  [{addr[1]}] Error: part must be 0-3, got {part}")
		response = {"type": "error", "message": f"part must be 0-3, got {part}"}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		return
	
	if not isinstance(surpriseness, (int, float)) or surpriseness < 0 or surpriseness > 127:
		add_log(f"  [{addr[1]}] Error: surpriseness must be 0-127, got {surpriseness}")
		response = {"type": "error", "message": f"surpriseness must be 0-127, got {surpriseness}"}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		return
	
	# Check if we have an analyzed file
	if last_analyzed_file is None:
		response = {"type": "error", "message": "No file analyzed yet. Please run analyse_and_generate first."}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		add_log(f"  [{addr[1]}] Error: No analyzed file available")
		return
	
	# Check if model is loaded
	if graphidyom_model_id is None:
		response = {"type": "error", "message": "No GraphIDYOM model loaded"}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		return
	
	# Ensure output folder exists
	try:
		output_path = normalize_client_path(output_folder)
		output_path.mkdir(parents=True, exist_ok=True)
	except Exception as e:
		response = {"type": "error", "message": f"Could not create output folder: {e}"}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		add_log(f"  [{addr[1]}] Error creating output folder: {e}")
		return
	
	# Map surpriseness to strategy (0-127 range)
	if surpriseness <= 42:
		strategy = 0  # Most probable
	elif surpriseness <= 84:
		strategy = 1  # Average probability
	else:
		strategy = 2  # Least probable
	
	add_log(f"  [{addr[1]}] Variate: part={part}, surpriseness={surpriseness}, strategy={strategy}, output={output_folder}")
	
	try:
		input_path = normalize_client_path(last_analyzed_file)
		# Parse the MIDI file - returns list of {'pitch', 'duration', 'interval'}
		note_data = parse_midi_file(str(input_path))
		if note_data is None or len(note_data) == 0:
			response = {"type": "error", "message": "Could not parse MIDI file"}
			conn.sendall((json.dumps(response) + "\n").encode('utf8'))
			return
		
		# Extract just pitches for history
		notes = [n['pitch'] for n in note_data]
		
		# Load original MIDI file
		mid = mido.MidiFile(str(input_path))
		
		# Find the melody track
		melody_track = None
		if len(mid.tracks) > 1:
			melody_track = mid.tracks[1]
		else:
			melody_track = mid.tracks[0]
		
		# Extract note_on/note_off pairs from MIDI, matching by (pitch, channel).
		# IMPORTANT: MIDI msg.time is a delta from the previous message, so pairing must NOT be LIFO-only.
		# We'll build note_info in NOTE-ON order, then fill the matching off_index when we see the note_off.
		from collections import defaultdict
		note_info = []  # List of {'pitch': int, 'channel': int, 'on_index': int, 'off_index': int}
		open_notes = defaultdict(list)  # key=(pitch, channel) -> stack of note_info entries
		
		for i, msg in enumerate(melody_track):
			if msg.type == 'note_on' and msg.velocity > 0:
				ch = getattr(msg, 'channel', 0)
				entry = {
					'pitch': msg.note,
					'channel': ch,
					'on_index': i,
					'off_index': None,
				}
				note_info.append(entry)
				open_notes[(msg.note, ch)].append(entry)
			elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)):
				ch = getattr(msg, 'channel', 0)
				key = (msg.note, ch)
				if open_notes.get(key):
					entry = open_notes[key].pop()
					entry['off_index'] = i
		
		# Keep only fully-paired notes
		note_info = [n for n in note_info if n.get('off_index') is not None]
		
		if notes is None or len(note_info) == 0:
			response = {"type": "error", "message": "Could not parse MIDI file or no notes found"}
			conn.sendall((json.dumps(response) + "\n").encode('utf8'))
			return
		
		# Divide MIDI notes into 4 parts based on actual MIDI note count
		part_size = len(note_info) // 4
		remainder = len(note_info) % 4
		
		# Build part boundaries accounting for remainder
		parts = []
		start = 0
		for p in range(4):
			# Distribute remainder among first parts
			size = part_size + (1 if p < remainder else 0)
			parts.append((start, start + size))
			start += size
		
		part_start, part_end = parts[part]
		
		add_log(f"  [{addr[1]}] Total notes in MIDI: {len(note_info)}, Part {part} notes: {part_start}-{part_end} ({part_end - part_start} notes)")
		
		# Generate replacement notes for the part
		replaced_notes = []  # List of pitch values
		min_note = min(generation_params['min_midi'], generation_params['max_midi'])
		# IMPORTANT: if we change a note to a pitch that is already ACTIVE (overlapping note of same pitch),
		# some DAWs (Ableton) may merge/shorten notes in the piano roll, which LOOKS like duration changes.
		# To keep rhythmic structure identical, avoid selecting a pitch that is currently active.
		import heapq
		replaced_notes = []  # List of pitch values (only for the selected part)
		min_note = min(generation_params['min_midi'], generation_params['max_midi'])
		max_note = max(generation_params['min_midi'], generation_params['max_midi'])
		
		# Build history up to the start of this part using analyzed notes
		# Map the part boundaries to the analyzed notes
		analysis_part_start = int((part_start / len(note_info)) * len(notes)) if len(note_info) > 0 else 0
		current_history = list(notes[:analysis_part_start])
		
		# Track active pitches using message index ordering (on/off overlap)
		active_pitch_counts = {}
		pending_note_offs = []  # heap of (off_index, pitch)
		
		def _inc_active(pitch):
			active_pitch_counts[pitch] = active_pitch_counts.get(pitch, 0) + 1
		
		def _dec_active(pitch):
			cnt = active_pitch_counts.get(pitch, 0)
			if cnt <= 1:
				active_pitch_counts.pop(pitch, None)
			else:
				active_pitch_counts[pitch] = cnt - 1
		
		# Prime active notes state using notes before the target part (original pitches)
		for idx in range(0, part_start):
			info = note_info[idx]
			on_idx = info['on_index']
			while pending_note_offs and pending_note_offs[0][0] < on_idx:
				_, p = heapq.heappop(pending_note_offs)
				_dec_active(p)
			p = info['pitch']
			_inc_active(p)
			heapq.heappush(pending_note_offs, (info['off_index'], p))
		
		# Process notes in the target part
		for idx in range(part_start, part_end):
			info = note_info[idx]
			on_idx = info['on_index']
			while pending_note_offs and pending_note_offs[0][0] < on_idx:
				_, p = heapq.heappop(pending_note_offs)
				_dec_active(p)
			
			orig_note_value = info['pitch']
			next_note = orig_note_value
			try:
				# Get predictions for current history
				result = graphidyom_service.predict_next(
					model_id=graphidyom_model_id,
					midi_history=current_history,
					top_k=128,
					output="midi",
					reset_stm=True,
				)
				predictions = result.get("predictions", [])
				
				# Filter by MIDI range
				candidates = []
				for p in predictions:
					midi = p.get("midi")
					if midi is not None and min_note <= midi <= max_note:
						candidates.append(midi)
						if len(candidates) >= 128:
							break
				
				if candidates:
					# Determine the order we scan candidates based on strategy
					idx_order = []
					if strategy == 0:
						idx_order = list(range(len(candidates)))
					elif strategy == 2:
						idx_order = list(range(len(candidates) - 1, -1, -1))
					else:
						mid_i = len(candidates) // 2
						idx_order.append(mid_i)
						for d in range(1, len(candidates)):
							left_i = mid_i - d
							right_i = mid_i + d
							if left_i >= 0:
								idx_order.append(left_i)
							if right_i < len(candidates):
								idx_order.append(right_i)
							if len(idx_order) >= len(candidates):
								break
					
					# Pick the first candidate that is NOT currently active (prevents Ableton note-merge artifacts)
					for j in idx_order:
						cand = candidates[j]
						if active_pitch_counts.get(cand, 0) == 0:
							next_note = cand
							break
					# If all candidates conflict, keep the original pitch (safe: preserves visible durations)
			
			except Exception as e:
				add_log(f"  [{addr[1]}] Error predicting note {idx}: {e}")
				next_note = orig_note_value
			
			replaced_notes.append(next_note)
			current_history.append(next_note)
			
			# Update active note tracking with the chosen pitch
			_inc_active(next_note)
			heapq.heappush(pending_note_offs, (info['off_index'], next_note))
		
		# Replace the notes in the MIDI track with new pitches.
		# We MUST update BOTH note_on and the MATCHED note_off (same original pitch/channel) so durations stay identical.
		# Use msg.copy() to preserve all fields (especially delta-time), changing ONLY the pitch.
		for idx, info in enumerate(note_info):
			if not (part_start <= idx < part_end):
				continue
			replacement_idx = idx - part_start
			if replacement_idx >= len(replaced_notes):
				continue
			new_pitch = replaced_notes[replacement_idx]
			on_i = info['on_index']
			off_i = info['off_index']
			melody_track[on_i] = melody_track[on_i].copy(note=new_pitch)
			melody_track[off_i] = melody_track[off_i].copy(note=new_pitch)
		
		# Generate output filename
		part_names = ['beginning', 'first_half', 'third_quarter', 'end']
		output_filename = f"{input_path.stem}_variate_part{part}_{part_names[part]}_surp{int(surpriseness)}.mid"
		
		# Save the modified MIDI file to the specified output folder
		output_file = output_path / output_filename
		mid.save(str(output_file))
		
		add_log(f"  [{addr[1]}] Variate complete! Saved to: {output_filename}")
		
		# Send success response
		response = {
			"type": "variate_midi_file_result",
			"status": "success",
			"input_file": input_path.name,
			"output_file": output_filename,
			"output_folder": str(output_path),
			"part": part,
			"part_name": part_names[part],
			"surpriseness": surpriseness,
			"strategy": ["most_probable", "average", "least_probable"][strategy],
			"notes_replaced": len(replaced_notes),
			"full_path": str(output_file)
		}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		_auto_export_ic_plot_for_midi(addr, conn, output_file)
		
	except Exception as e:
		response = {
			"type": "error",
			"message": f"Variate failed: {str(e)}"
		}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		add_log(f"  [{addr[1]}] Variate error: {e}")


def analyze_midi_sequence(file_path, addr, conn):
	"""Analyze a MIDI file sequence note-by-note and calculate IC for each note."""
	global graphidyom_service, graphidyom_model_id, last_analyzed_file
	
	file_path = normalize_client_path(file_path)
	if not file_path.exists():
		response = {"type": "error", "message": f"File not found: {file_path}"}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		add_log(f"  [{addr[1]}] File not found: {file_path}")
		return

	with ic_history_lock:
		ic_history.clear()
		ic_notes_history.clear()
	add_log(f"  [{addr[1]}] IC history cleared for new analysis")

	# Store the file path for potential later use in generate_midi_file
	last_analyzed_file = str(file_path)
	add_log(f"  [{addr[1]}] Storing analyzed file for later generation: {file_path.name}")
	
	# Parse the MIDI file - returns list of {'pitch', 'duration', 'interval'}
	note_data = parse_midi_file(str(file_path))
	if note_data is None:
		response = {"type": "error", "message": "Could not parse MIDI file"}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		add_log(f"  [{addr[1]}] Failed to analyze MIDI file")
		return
	
	# Extract just pitches for analysis
	notes = [n['pitch'] for n in note_data]
	
	if graphidyom_model_id is None:
		response = {"type": "error", "message": "No GraphIDYOM model loaded"}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		add_log(f"  [{addr[1]}] No model available for analysis")
		return
	
	add_log(f"  [{addr[1]}] Analyzing MIDI file: {len(notes)} notes")
	
	# Analyze each note in sequence
	analysis_result = {
		"type": "midi_analysis",
		"file": str(Path(file_path).name),
		"total_notes": len(notes),
		"analysis": []
	}
	
	midi_history = []
	min_note = min(generation_params['min_midi'], generation_params['max_midi'])
	max_note = max(generation_params['min_midi'], generation_params['max_midi'])
	
	for idx, note_event in enumerate(note_data):
		note = note_event['pitch']
		duration = note_event['duration']
		interval = note_event['interval']
		
		try:
			# If history is not empty, get predictions
			if len(midi_history) > 0:
				result = graphidyom_service.predict_next(
					model_id=graphidyom_model_id,
					midi_history=midi_history,
					top_k=128,
					output="midi",
					reset_stm=True,
				)
				predictions = result.get("predictions", [])
				
				# Find this note's probability and calculate IC
				note_prob = None
				note_ic = None
				for p in predictions:
					if p.get("midi") == note:
						note_prob = p.get("prob", 0.0)
						if note_prob > 0:
							note_ic = -math.log2(note_prob)
						break
				
				# Add to analysis
				analysis_result["analysis"].append({
					"index": idx,
					"note": note,
					"duration": duration,
					"interval": interval,
					"probability": note_prob,
					"ic": note_ic,
					"in_range": min_note <= note <= max_note
				})
				
				# Send user_note_ic update to Max for real-time plot updates (same format as handle_user_note)
				ic_update = {
					"type": "predictions",
					"user_note": None,
					"user_note_prob": note_prob,
					"user_note_ic": note_ic,
					"predictions": []
				}
				
				# Track IC for export
				if note_ic is not None:
					with ic_history_lock:
						ic_history.append(note_ic)
						ic_notes_history.append(note)
				
				conn.sendall((json.dumps(ic_update) + "\n").encode('utf8'))
				
				ic_str = f"{note_ic:.3f}" if note_ic is not None else "N/A"
				prob_str = f"{note_prob:.1%}" if note_prob is not None else "N/A"
				add_log(f"  [{addr[1]}] Note {idx}: {note} | Dur: {duration} | Int: {interval} | IC: {ic_str} bits | Prob: {prob_str}")
			else:
				# First note - no predictions yet
				analysis_result["analysis"].append({
					"index": idx,
					"note": note,
					"duration": duration,
					"interval": interval,
					"probability": 1.0,
					"ic": 0.0,
					"in_range": min_note <= note <= max_note
				})
				
				# Send user_note_ic update for first note (same format as handle_user_note)
				ic_update = {
					"type": "predictions",
					"user_note": None,
					"user_note_prob": 1.0,
					"user_note_ic": 0.0,
					"predictions": []
				}
				
				# Track IC for export
				with ic_history_lock:
					ic_history.append(0.0)
					ic_notes_history.append(note)
				conn.sendall((json.dumps(ic_update) + "\n").encode('utf8'))
				
				add_log(f"  [{addr[1]}] Note {idx}: {note} | Dur: {duration} | Int: {interval} (first note - no predictions)")
			
			# Add note to history for next iteration
			midi_history.append(note)
		
		except Exception as e:
			add_log(f"  [{addr[1]}] Error analyzing note {idx}: {e}")
			analysis_result["analysis"].append({
				"index": idx,
				"note": note,
				"probability": None,
				"ic": None,
				"in_range": min_note <= note <= max_note,
				"error": str(e)
			})
	
	# Send analysis result
	conn.sendall((json.dumps(analysis_result) + "\n").encode('utf8'))
	add_log(f"  [{addr[1]}] Analysis complete! {len(notes)} notes analyzed")


def initialize_graphidyom():
	"""Initialize GraphIDYOM service and load pretrained model."""
	global graphidyom_service, graphidyom_model_id, available_models
	
	# Ensure ProgramData directories exist (Windows)
	ensure_programdata_directories()
	
	# Initialize service (silent)
	graphidyom_service = GraphIDYOMAppService()
	
	# Determine datasets directory with priority:
	# WINDOWS: 1. ProgramData (shared, where NSIS installs datasets)
	# ALL:     2. Installation directory (./datasets) - where NSIS embeds datasets
	# ALL:     3. Local directory (../datasets) - for development
	# ALL:     4. Parent directory (../datasets) - for development  
	# FALLBACK: 5. User home directory
	
	datasets_dir = None
	
	# Check 1: ProgramData on Windows (where NSIS installs)
	if sys.platform == 'win32':
		programdata_datasets = Path(os.getenv('PROGRAMDATA')) / "GraphIDYOM" / "datasets"
		print(f"{info(f'Checking ProgramData: {programdata_datasets}')}")
		if programdata_datasets.exists():
			subfolders = list(programdata_datasets.iterdir())
			print(f"{info(f'  ProgramData exists with {len(subfolders)} items')}")
			if subfolders:
				datasets_dir = programdata_datasets
				print(f"{success(f'Using installed datasets from ProgramData')}")
	
	# Check 2: Installation directory (where NSIS copies datasets)
	if datasets_dir is None:
		install_datasets = script_dir / "datasets"
		print(f"{info(f'Checking installation directory: {install_datasets}')}")
		if install_datasets.exists():
			subfolders = list(install_datasets.iterdir())
			print(f"{info(f'  Installation directory exists with {len(subfolders)} items')}")
			if subfolders:
				datasets_dir = install_datasets
				print(f"{success(f'Using embedded datasets from installation directory')}")
	
	# Check 3: Parent directory (for development - e.g., when running from ableton-experiments folder)
	if datasets_dir is None:
		parent_datasets = script_dir.parent / "datasets"
		print(f"{info(f'Checking parent directory: {parent_datasets}')}")
		if parent_datasets.exists():
			subfolders = list(parent_datasets.iterdir())
			print(f"{info(f'  Parent directory exists with {len(subfolders)} items')}")
			if subfolders:
				datasets_dir = parent_datasets
				print(f"{success(f'Using development datasets from parent directory')}")
	
	# Check 4: Fallback to user-writable locations
	if datasets_dir is None:
		if sys.platform == 'win32':
			datasets_dir = Path(os.getenv('PROGRAMDATA')) / "GraphIDYOM" / "datasets"
			print(f"{info(f'No existing datasets found, will use: {datasets_dir}')}")
		else:
			datasets_dir = Path.home() / ".graphidyom" / "datasets"
			print(f"{info(f'No existing datasets found, will use: {datasets_dir}')}")
		
		# Create the directory if it doesn't exist
		try:
			datasets_dir.mkdir(parents=True, exist_ok=True)
		except Exception as e:
			print(f"{warning(f'Could not create datasets directory: {e}')}")
	
	graphidyom_service.manager.base_dir = script_dir
	graphidyom_service.manager.datasets_dir = datasets_dir
	
	# Debug: Check what's in datasets_dir
	print(f"{info(f'Datasets directory: {datasets_dir}')}")
	if datasets_dir.exists():
		subfolders = [f.name for f in datasets_dir.iterdir() if f.is_dir()]
		print(f"{info(f'Found dataset folders: {subfolders}')}")
		
		# Check if models have the required files
		for dataset_name in subfolders:
			dataset_path = datasets_dir / dataset_name
			model_folders = [f.name for f in dataset_path.iterdir() if f.is_dir()]
			print(f"{info(f'  {dataset_name}: {model_folders}')}")
			
			# Check for metadata.json in each model folder
			for model_folder in model_folders:
				model_path = dataset_path / model_folder
				has_metadata = (model_path / 'metadata.json').exists()
				has_graphs = (model_path / 'graphs').exists()
				print(f"{info(f'    {model_folder}: metadata={has_metadata}, graphs={has_graphs}')}")
	else:
		print(f"{warning(f'Datasets directory does not exist')}")
	
	# List available models
	models = graphidyom_service.list_pretrained_models()
	available_models = models  # Store globally for switching
	
	if not models:
		print(f"\n{warning('⚠ No pretrained models found')}")
		print(f"{warning(f'Datasets dir: {datasets_dir}')}")
		datasets_path = str(datasets_dir)
		print(f"\n{info('To use GraphIDYOM:')}")
		print(f"{info('1. Place pre-trained datasets in: ' + datasets_path)}")
		print(f"{info('2. Or train new models using Max4Live')}")
		print(f"{info('3. Models will be saved to: ' + datasets_path)}")
		print(f"\n{info('Server is ready. Connect from Max4Live to get started.')}\n")
		# Don't exit - server can still accept connections and train models
		graphidyom_model_id = None
		return
	
	# Try to load specified model, or use first available
	loaded = False
	
	# Try default model first
	try:
		result = graphidyom_service.load_pretrained_model(
			dataset_name=DATASET_NAME,
			model_folder_name=MODEL_FOLDER
		)
		graphidyom_model_id = result.get("model_id") if isinstance(result, dict) else None
		model_info = f'{DATASET_NAME}/{MODEL_FOLDER}'
		print(f"{success(f'✓ Loaded default model: {model_info}')}\n")
		loaded = True
	except Exception as e:
		error_str = str(e)
		print(f"\n{warning(f'⚠ Could not load default model')}")
		print(f"{warning(f'Error: {error_str}')}")
		
		# If viewpoint mismatch, try to load models with different viewpoint presets
		if 'Viewpoint config mismatch' in error_str or 'viewpoint' in error_str.lower():
			print(f"{info('Attempting to load with different viewpoint configurations...')}")
			
			# Try each available model with different viewpoint presets
			viewpoint_presets = ['midi', 'pitch', 'interval', 'contour', 'parsistentstiminus']
			for model in models:
				if loaded:
					break
				for vp in viewpoint_presets:
					try:
						# Try loading with a specific viewpoint
						result = graphidyom_service.load_pretrained_model(
							dataset_name=model['dataset'],
							model_folder_name=model['folder']
						)
						graphidyom_model_id = result.get("model_id") if isinstance(result, dict) else None
						dataset = model['dataset']
						folder = model['folder']
						print(f"{success(f'✓ Loaded model: {dataset}/{folder}')}\n")
						loaded = True
						break
					except Exception:
						continue
		
		# If still not loaded, try first available model without viewpoint changes
		if not loaded:
			try:
				model = models[0]
				result = graphidyom_service.load_pretrained_model(
					dataset_name=model['dataset'],
					model_folder_name=model['folder']
				)
				graphidyom_model_id = result.get("model_id") if isinstance(result, dict) else None
				dataset = model['dataset']
				folder = model['folder']
				print(f"{success(f'✓ Loaded fallback model: {dataset}/{folder}')}\n")
				loaded = True
			except Exception as e2:
				print(f"{warning(f'Failed to load fallback model: {e2}')}")
	
	if not loaded:
		print(f"\n{warning(f'⚠ Could not load any pretrained models')}")
		print(f"{warning(f'This may be due to viewpoint configuration incompatibility.')}")
		print(f"{warning(f'Consider training a new model with the current GraphIDYOM version.')}")
		print(f"{info('Server is running. Connect from Max4Live to train a new model.')}\n")
		graphidyom_model_id = None


def load_model_by_index(index):
	"""Load a pretrained model by index."""
	global graphidyom_model_id, available_models
	
	if not available_models:
		print("No models available")
		return False
	
	if index < 0 or index >= len(available_models):
		print(f"Invalid model index: {index}. Available: 0-{len(available_models)-1}")
		return False
	
	model = available_models[index]
	try:
		result = graphidyom_service.load_pretrained_model(
			dataset_name=model["dataset"],
			model_folder_name=model["folder"]
		)
		graphidyom_model_id = result["model_id"]
		print(f"✓ Switched to model {index}: {model['dataset']}/{model['folder']}")
		return True
	except Exception as e:
		print(f"✗ Failed to load model {index}: {e}")
		return False


def list_available_models():
	"""Return list of available models with their indices."""
	global available_models
	models_info = []
	for i, m in enumerate(available_models):
		models_info.append({
			"index": i,
			"dataset": m["dataset"],
			"folder": m["folder"],
			"name": f"{m['dataset']}/{m['folder']}"
		})
	return models_info


def refresh_available_models():
	"""Refresh the list of available models from disk."""
	global available_models, graphidyom_service
	try:
		available_models = graphidyom_service.list_pretrained_models()
		return len(available_models)
	except Exception as e:
		print(f"{warning(f'Error refreshing models: {e}')}")
		return 0


def note_division_to_seconds(division, tempo_bpm):
	"""Convert a note division (like 1/4, 1/8) to seconds based on tempo.
	
	Args:
		division: Note value as fraction (e.g., 0.25 for 1/4 note, 0.125 for 1/8)
		tempo_bpm: Tempo in beats per minute
	
	Returns:
		Duration in seconds
	"""
	quarter_note_duration = 60.0 / tempo_bpm
	return quarter_note_duration * division * 4


def get_note_duration_seconds():
	"""Get current note duration in seconds based on tempo and division."""
	return note_division_to_seconds(
		generation_params['note_duration_division'],
		generation_params['tempo']
	)


def get_note_interval_seconds():
	"""Get current note interval in seconds based on tempo and division."""
	return note_division_to_seconds(
		generation_params['note_interval_division'],
		generation_params['tempo']
	)


def select_next_note_from_predictions(predictions, midi_history):
	"""Select next MIDI note from GraphIDYOM predictions."""
	# Ensure valid MIDI range
	min_note = min(generation_params['min_midi'], generation_params['max_midi'])
	max_note = max(generation_params['min_midi'], generation_params['max_midi'])
	
	if not predictions:
		# Fallback to random if no predictions
		return random.randint(min_note, max_note)
	
	# Filter predictions to valid MIDI range
	valid_preds = []
	for p in predictions:
		midi = p.get("midi")
		if midi is not None and min_note <= midi <= max_note:
			valid_preds.append(p)
	
	if not valid_preds:
		# Fallback if no valid predictions in range
		return random.randint(min_note, max_note)
	
	if generation_params['use_probabilistic']:
		# Sample from probability distribution with temperature
		probs = [p.get("prob", 0.0) for p in valid_preds]
		# Apply temperature
		temp = generation_params['temperature']
		if temp != 1.0:
			import math
			probs = [math.exp(math.log(p + 1e-10) / temp) for p in probs]
		# Normalize
		total = sum(probs)
		if total > 0:
			probs = [p / total for p in probs]
		else:
			probs = [1.0 / len(probs)] * len(probs)
		# Sample
		import numpy as np
		idx = np.random.choice(len(valid_preds), p=probs)
		return valid_preds[idx]["midi"]
	else:
		# Argmax: select highest probability
		return max(valid_preds, key=lambda p: p.get("prob", 0.0))["midi"]


def trim_midi_history(midi_history):
	"""Keep MIDI history within the configured live history limit."""
	max_history = max(1, int(generation_params.get('max_history', MAX_HISTORY_LENGTH)))
	if len(midi_history) <= max_history:
		return False
	del midi_history[:-max_history]
	return True


def handle_client(conn, addr):
	global server_running, graphidyom_model_id, graphidyom_service
	add_log(f"{success(f'Client connected from {addr[0]}:{addr[1]}')}\n")
	
	# Register this connection
	with active_connections_lock:
		active_connections.append(conn)
	
	
	# Set socket to non-blocking to check for incoming commands
	conn.setblocking(False)
	
	# Initialize MIDI history for this client
	midi_history = []
	session_id = None
	session_model_id = None

	def ensure_prediction_session():
		nonlocal session_id, session_model_id, midi_history
		current_model_id = graphidyom_model_id
		if graphidyom_service is None or current_model_id is None:
			if session_id is not None and graphidyom_service is not None:
				try:
					graphidyom_service.close_session(session_id=session_id)
				except Exception:
					pass
			session_id = None
			session_model_id = None
			return None

		if session_id is not None and session_model_id == current_model_id:
			return session_id

		if session_id is not None:
			try:
				graphidyom_service.close_session(session_id=session_id)
			except Exception:
				pass

		try:
			started = graphidyom_service.start_session(model_id=current_model_id)
			session_id = started["session_id"]
			session_model_id = current_model_id
			if midi_history:
				graphidyom_service.observe_in_session(session_id=session_id, midi_notes=midi_history)
			return session_id
		except Exception as e:
			add_log(f"⚠ Session init error: {e}")
			session_id = None
			session_model_id = None
			return None

	def reset_prediction_session_from_history():
		"""Reset the active session STM and rebuild it from retained history."""
		nonlocal session_id, session_model_id, midi_history
		current_session_id = ensure_prediction_session()
		if current_session_id is None or graphidyom_service is None:
			return None
		try:
			graphidyom_service.reset_session(session_id=current_session_id)
			if midi_history:
				graphidyom_service.observe_in_session(session_id=current_session_id, midi_notes=midi_history)
			return current_session_id
		except Exception as e:
			add_log(f"⚠ Session reset error: {e}")
			try:
				graphidyom_service.close_session(session_id=current_session_id)
			except Exception:
				pass
			session_id = None
			session_model_id = None
			return ensure_prediction_session()

	def enforce_history_limit_on_session():
		"""Apply max_history to local history and session STM before prediction."""
		trimmed = trim_midi_history(midi_history)
		if trimmed:
			return reset_prediction_session_from_history()
		return ensure_prediction_session()

	def safe_send(msg):
		"""Safely send a JSON message to the client."""
		try:
			if isinstance(msg, dict):
				msg = json.dumps(msg)
			if isinstance(msg, str):
				conn.sendall((msg + "\n").encode('utf8'))
			else:
				conn.sendall((json.dumps(msg) + "\n").encode('utf8'))
		except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
			return False  # Connection closed
		except Exception as e:
			add_log(f"⚠ Error sending message: {e}")
			return False
		return True

	ensure_prediction_session()
	
	# Buffer for incoming messages
	recv_buffer = ""
	
	# Absolute sequencer clock. Scheduling from the ideal next event time
	# prevents small sleep/prediction delays from accumulating as tempo drift.
	scheduled_interval_seconds = get_note_interval_seconds()
	next_generation_time = time.perf_counter() + scheduled_interval_seconds
	was_generating = generation_params['mode'] == 'generate'
	
	try:
		while server_running:
			# Check for incoming parameter change commands from Max
			try:
				data = conn.recv(1024).decode('utf8')
				if data:
					recv_buffer += data
					lines = recv_buffer.split('\n')
					recv_buffer = lines.pop()  # Keep incomplete line
					
					for line in lines:
						line = line.strip()
						if not line:
							continue
						try:
							cmd = json.loads(line)
							# Handle user note input in interact mode
							if 'user_note' in cmd:
								if generation_params['mode'] != 'interact':
									continue
								enforce_history_limit_on_session()
								handle_user_note(cmd['user_note'], midi_history, session_id, addr, conn)
								enforce_history_limit_on_session()
								continue
							# Handle training note input in train mode
							elif 'train_note' in cmd:
								handle_train_note(cmd['train_note'], addr, conn)
								continue
							# Handle training sequence control commands
							elif 'train_control' in cmd:
								handle_train_control(cmd['train_control'], addr, conn)
								continue
							# Handle STM reset command
							elif 'reset' in cmd and cmd['reset']:
								midi_history.clear()
								reset_prediction_session_from_history()
								# Also clear IC history when STM is reset
								with ic_history_lock:
									ic_history.clear()
									ic_notes_history.clear()
								add_log(f"{success(f'STM history reset by client {addr[1]}')}:{Colors.END}")
								add_log(f"{success(f'IC history cleared')}:{Colors.END}")
								response = {"type": "reset_ack", "status": "cleared", "message": "STM history cleared"}
								if not safe_send(response):
									break
								continue
							# Handle MIDI analysis command - reset STM before analysis
							elif 'analyse_and_generate' in cmd:
								# Reset STM before analyzing
								midi_history.clear()
								if session_id is not None and graphidyom_service is not None:
									try:
										graphidyom_service.close_session(session_id=session_id)
									except Exception:
										pass
								session_id = None
								session_model_id = None
								add_log(f"{success(f'STM reset before analysis')}")
								# Now analyze
								file_path = cmd['analyse_and_generate']
								analyze_midi_sequence(file_path, addr, conn)
								continue
							else:
								handle_parameter_change(cmd, addr, conn)
						except json.JSONDecodeError:
							# Try simple key=value format
							if '=' in line:
								key, value = line.split('=', 1)
								handle_parameter_change({key.strip(): value.strip()}, addr, conn)
			except BlockingIOError:
				pass  # No data available, continue
			except (ConnectionResetError, OSError) as e:
				# Connection was reset by Max client (e.g., during mode switch)
				# This is normal, just break the loop
				print(f'Connection reset by {addr}')
				break
			except Exception as e:
				print(f"⚠ Error receiving command: {e}")
			
			# Only auto-generate in 'generate' mode
			interval_seconds = get_note_interval_seconds()
			now = time.perf_counter()
			if generation_params['mode'] != 'generate' or not generation_params.get('sequencer_running', True):
				was_generating = False
				scheduled_interval_seconds = interval_seconds
				next_generation_time = now + interval_seconds
				time.sleep(0.01)  # Small sleep to avoid busy loop in interact/train modes
				continue
			
			if (not was_generating) or abs(interval_seconds - scheduled_interval_seconds) > 1e-9:
				scheduled_interval_seconds = interval_seconds
				next_generation_time = now + interval_seconds
				was_generating = True
			
			if now < next_generation_time:
				time.sleep(min(0.001, next_generation_time - now))  # Small sleep to avoid busy loop
				continue
			
			if interval_seconds > 0 and now - next_generation_time >= interval_seconds:
				missed_slots = int((now - next_generation_time) // interval_seconds)
				next_generation_time += missed_slots * interval_seconds
			
			next_generation_time += interval_seconds
			
			# Get prediction from GraphIDYOM based on current history
			try:
				current_session_id = enforce_history_limit_on_session()
				if current_session_id is not None:
					result = graphidyom_service.predict_next_session(
						session_id=current_session_id,
						top_k=20,
						output="midi",
					)
				elif graphidyom_model_id is not None:
					result = graphidyom_service.predict_next(
						model_id=graphidyom_model_id,
						midi_history=midi_history,
						top_k=20,
						output="midi",
						reset_stm=True,
					)
				else:
					raise RuntimeError("No GraphIDYOM model loaded")
				predictions = result.get("predictions", [])
				
				# Select next note from predictions
				note = select_next_note_from_predictions(predictions, midi_history)
				
				# Filter predictions by MIDI range for top 3 display
				min_note = min(generation_params['min_midi'], generation_params['max_midi'])
				max_note = max(generation_params['min_midi'], generation_params['max_midi'])
				valid_preds = []
				for p in predictions:
					midi = p.get("midi")
					if midi is not None and min_note <= midi <= max_note:
						valid_preds.append(p)
						if len(valid_preds) >= 3:
							break
				
				# Get top 3 for display
				top_3 = valid_preds[:3]
				
				# Send top 3 predictions to Max
				predictions_msg = {
					"type": "predictions",
					"predictions": [
						{"note": p["midi"], "prob": p.get("prob", 0.0)} 
						for p in top_3
					]
				}
				if not safe_send(predictions_msg):
					break  # Connection closed
				
				# Show prediction info (compact format)
				if predictions:
					last_note = midi_history[-1] if midi_history else None
					pred_str = " | ".join([f"{p['midi']}({p['prob']:.0%})" for p in top_3])
					last_note_str = f"{last_note:3d}" if last_note is not None else "  ?"
					#add_log(f"  [{addr[1]}] {last_note_str} → {pred_str} → {note:3d}")
				
			except Exception as e:
				add_log(f"⚠ Prediction error: {e}, using fallback")
				# Use safe range for fallback
				min_note = min(generation_params['min_midi'], generation_params['max_midi'])
				max_note = max(generation_params['min_midi'], generation_params['max_midi'])
				note = random.randint(min_note, max_note)
			
			# Add to history
			midi_history.append(note)
			if session_id is not None:
				graphidyom_service.observe_in_session(session_id=session_id, midi_notes=[note])
			enforce_history_limit_on_session()
			
			# Convert tempo-based divisions to seconds
			duration_seconds = get_note_duration_seconds()
			
			# Send note_on with duration (makenote will handle note_off)
			msg_on = {"type": "midi", "cmd": "note_on", "note": note, "vel": 100, "duration": duration_seconds}
			if not safe_send(msg_on):
				break  # Connection closed, exit loop
			
			now = time.perf_counter()
			if interval_seconds > 0 and now >= next_generation_time:
				missed_slots = int((now - next_generation_time) // interval_seconds) + 1
				next_generation_time += missed_slots * interval_seconds
			
	except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as e:
		add_log(f"{warning(f'Client {addr} disconnected: {type(e).__name__}')}")
	finally:
		if session_id is not None and graphidyom_service is not None:
			try:
				graphidyom_service.close_session(session_id=session_id)
			except Exception:
				pass
		# Unregister this connection
		with active_connections_lock:
			if conn in active_connections:
				active_connections.remove(conn)
		
		# Close the connection
		try:
			conn.close()
		except Exception:
			pass


def handle_user_note(note, midi_history, session_id, addr, conn):
	"""Handle user note input in interact mode."""
	global graphidyom_service, graphidyom_model_id
	
	try:
		# If history is empty (e.g., after reset), just add the first note
		if len(midi_history) == 0:
			midi_history.append(note)
			if session_id is not None:
				try:
					graphidyom_service.observe_in_session(session_id=session_id, midi_notes=[note])
				except Exception:
					pass
			
			# Send acknowledgment
			response = {
				"type": "predictions",
				"user_note": note,
				"user_note_prob": 1.0,
				"user_note_ic": 0.0,
				"predictions": []
			}
			
			# Track IC for export
			with ic_history_lock:
				ic_history.append(0.0)
				ic_notes_history.append(note)
			
			conn.sendall((json.dumps(response) + "\n").encode('utf8'))
			add_log(f"  [{addr[1]}] First note added: {note} (building history...)")
			return
		
		# Get predictions for user's note (top 128 for full IC calculation)
		result_for_user_note = graphidyom_service.predict_next(
			model_id=graphidyom_model_id,
			midi_history=midi_history,
			top_k=128,
			output="midi",
			reset_stm=True,
		)
		user_note_predictions = result_for_user_note.get("predictions", [])
		
		# Find user's note probability and calculate IC
		user_note_prob = None
		user_note_ic = None
		min_note = min(generation_params['min_midi'], generation_params['max_midi'])
		max_note = max(generation_params['min_midi'], generation_params['max_midi'])
		
		for p in user_note_predictions:
			if p.get("midi") == note:
				user_note_prob = p.get("prob", 0.0)
				if user_note_prob > 0:
					user_note_ic = -math.log2(user_note_prob)
				break
		
		# Now add user's note to history
		midi_history.append(note)
		if session_id is not None:
			graphidyom_service.observe_in_session(session_id=session_id, midi_notes=[note])
		
		# Get predictions for next note AFTER adding user's note
		if session_id is not None:
			result = graphidyom_service.predict_next_session(
				session_id=session_id,
				top_k=20,
				output="midi",
			)
		else:
			result = graphidyom_service.predict_next(
				model_id=graphidyom_model_id,
				midi_history=midi_history,
				top_k=20,
				output="midi",
				reset_stm=True,
			)
		predictions = result.get("predictions", [])
		
		# Filter by MIDI range
		valid_preds = []
		for p in predictions:
			midi = p.get("midi")
			if midi is not None and min_note <= midi <= max_note:
				valid_preds.append(p)
				if len(valid_preds) >= 3:
					break
		
		# Get top 3
		top_3 = valid_preds[:3]
		
		# Wait for note_interval_division before sending predictions (musical timing)
		interval_seconds = get_note_interval_seconds()
		time.sleep(interval_seconds)
		
		# Track IC for export (both user notes and file analysis notes)
		if user_note_ic is not None:
			with ic_history_lock:
				ic_history.append(user_note_ic)
				ic_notes_history.append(note)
		
		# Send predictions back to Max with user note IC
		response = {
			"type": "predictions",
			"user_note": note,
			"user_note_prob": user_note_prob,
			"user_note_ic": user_note_ic,
			"predictions": [
				{"note": p["midi"], "prob": p.get("prob", 0.0)} 
				for p in top_3
			]
		}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		
		
	except Exception as e:
		add_log(f"⚠ Error processing user note: {e}")


def handle_train_note(note, addr, conn):
	"""Handle training note input in train mode."""
	global current_training_sequence
	
	add_log(f"  [{addr[1]}] Training note: {note}")
	try:
		# Add note to current training sequence
		current_training_sequence.append(note)
		
		# Send acknowledgment
		response = {
			"type": "train_ack",
			"note": note,
			"sequence_length": len(current_training_sequence)
		}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		
	except Exception as e:
		add_log(f"Error processing training note: {e}")


def handle_train_control(control, addr, conn):
	"""Handle training control commands (start, end, reset, save)."""
	global training_sequences, current_training_sequence
	
	add_log(f"  [{addr[1]}] Training control: {control}")
	try:
		if control == 'start_sequence':
			# Start a new training sequence
			current_training_sequence = []
			response = {"type": "train_status", "status": "started", "message": "New sequence started"}
			conn.sendall((json.dumps(response) + "\n").encode('utf8'))
			add_log(f"  [{addr[1]}] Started new training sequence")
		
		elif control == 'end_sequence':
			# End current sequence and add to training data
			if len(current_training_sequence) > 0:
				training_sequences.append(current_training_sequence.copy())
				response = {
					"type": "train_status",
					"status": "sequence_saved",
					"sequence_length": len(current_training_sequence),
					"total_sequences": len(training_sequences)
				}
				add_log(f"  [{addr[1]}] Saved sequence with {len(current_training_sequence)} notes (total: {len(training_sequences)} sequences)")
				current_training_sequence = []
			else:
				response = {"type": "train_status", "status": "error", "message": "No notes in current sequence"}
			conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		
		elif control == 'reset':
			# Clear all training data
			training_sequences = []
			current_training_sequence = []
			response = {"type": "train_status", "status": "reset", "message": "All training data cleared"}
			conn.sendall((json.dumps(response) + "\n").encode('utf8'))
			add_log(f"  [{addr[1]}] Cleared all training data")
		
		elif control == 'get_status':
			# Get current training status
			response = {
				"type": "train_status",
				"status": "info",
				"total_sequences": len(training_sequences),
				"current_sequence_length": len(current_training_sequence),
				"total_notes": sum(len(seq) for seq in training_sequences)
			}
			conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		
		elif control == 'save_dataset':
			# TODO: Implement dataset saving logic
			# This will save training_sequences to a file for later training
			response = {
				"type": "train_status",
				"status": "not_implemented",
				"message": "Dataset saving not yet implemented"
			}
			conn.sendall((json.dumps(response) + "\n").encode('utf8'))
			add_log(f"  [{addr[1]}] Dataset saving not yet implemented")
		
		else:
			response = {"type": "train_status", "status": "error", "message": f"Unknown control: {control}"}
			conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		
	except Exception as e:
		add_log(f"Error processing training control: {e}")


def handle_add_training_file(file_path, addr, conn):
	"""Add a MIDI file to the training queue."""
	global pending_training_files
	
	# Normalize path
	file_path = normalize_client_path(file_path)
	
	if not file_path.exists():
		response = {
			"type": "training_file_status",
			"status": "error",
			"message": f"File not found: {file_path}"
		}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		add_log(f"  [{addr[1]}] File not found: {file_path}")
		return
	
	# Check if it's a MIDI file
	if file_path.suffix.lower() not in ['.mid', '.midi']:
		response = {
			"type": "training_file_status",
			"status": "error",
			"message": f"Not a MIDI file: {file_path}"
		}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		add_log(f"  [{addr[1]}] Not a MIDI file: {file_path}")
		return
	
	# Add to pending files if not already there
	file_str = str(file_path)
	if file_str not in pending_training_files:
		pending_training_files.append(file_str)
	
	response = {
		"type": "training_file_status",
		"status": "file_added",
		"file": file_path.name,
		"total_files": len(pending_training_files)
	}
	conn.sendall((json.dumps(response) + "\n").encode('utf8'))
	add_log(f"  [{addr[1]}] Added training file: {file_path.name} ({len(pending_training_files)} total)")


def handle_clear_training_files(addr, conn):
	"""Clear all pending training files."""
	global pending_training_files
	pending_training_files = []
	
	response = {
		"type": "training_file_status",
		"status": "cleared",
		"message": "All training files cleared",
		"total_files": 0,
		"file": ""
	}
	conn.sendall((json.dumps(response) + "\n").encode('utf8'))
	add_log(f"  [{addr[1]}] Cleared all training files")


def handle_get_training_status(addr, conn):
	"""Get current training files status."""
	global pending_training_files, training_options, is_training
	
	last_file = Path(pending_training_files[-1]).name if pending_training_files else ""
	response = {
		"type": "training_file_status",
		"status": "info",
		"total_files": len(pending_training_files),
		"file": last_file,
		"files": [Path(f).name for f in pending_training_files],
		"options": training_options,
		"is_training": is_training
	}
	conn.sendall((json.dumps(response) + "\n").encode('utf8'))


def handle_set_training_option(option_name, option_value, addr, conn):
	"""Set a training option."""
	global training_options
	
	if option_name == 'dataset_name':
		# Sanitize dataset name (alphanumeric, underscore, dash only)
		sanitized = ''.join(c if c.isalnum() or c in '_-' else '_' for c in str(option_value))
		training_options['dataset_name'] = sanitized
		print(f"  [{addr[1]}] Set dataset_name: {sanitized}")
	
	elif option_name == 'augmented':
		training_options['augmented'] = str(option_value).lower() in ['true', '1', 'yes']
		print(f"  [{addr[1]}] Set augmented: {training_options['augmented']}")
	
	elif option_name == 'viewpoint':
		if option_value in ['midi', 'pitch', 'interval']:
			training_options['viewpoint'] = option_value
			print(f"  [{addr[1]}] Set viewpoint: {training_options['viewpoint']}")
	
	elif option_name == 'orders':
		# Parse orders (comma-separated or list)
		if isinstance(option_value, (list, tuple)):
			training_options['orders'] = tuple(int(o) for o in option_value)
		else:
			# Parse comma-separated string like "1,2,3,4,5"
			orders = [int(o.strip()) for o in str(option_value).split(',') if o.strip().isdigit()]
			training_options['orders'] = tuple(orders) if orders else (1, 2, 3, 4, 5)
		print(f"  [{addr[1]}] Set orders: {training_options['orders']}")
	
	response = {
		"type": "training_option_set",
		"option": option_name,
		"value": training_options.get(option_name)
	}
	conn.sendall((json.dumps(response) + "\n").encode('utf8'))


def handle_start_training(addr, conn):
	"""Start training from the pending MIDI files."""
	global pending_training_files, training_options, is_training
	global graphidyom_service, graphidyom_model_id, available_models
	
	if is_training:
		response = {
			"type": "training_result",
			"status": "error",
			"message": "Training already in progress"
		}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		return
	
	if not pending_training_files:
		response = {
			"type": "training_result",
			"status": "error",
			"message": "No MIDI files added. Use add_training_file first."
		}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		return
	
	is_training = True
	
	# Send training started message
	response = {
		"type": "training_result",
		"status": "started",
		"message": f"Training started with {len(pending_training_files)} files",
		"dataset_name": training_options['dataset_name']
	}
	conn.sendall((json.dumps(response) + "\n").encode('utf8'))
	add_log(f"  [{addr[1]}] Starting training with {len(pending_training_files)} files...")
	add_log(f"  [{addr[1]}] Dataset name: {training_options['dataset_name']}")
	add_log(f"  [{addr[1]}] Options: augmented={training_options['augmented']}, viewpoint={training_options['viewpoint']}")
	
	try:
		import shutil
		import tempfile
		from io import StringIO
		
		# Create a temporary folder and copy all MIDI files there
		temp_dir = Path(tempfile.mkdtemp(prefix='graphidyom_train_'))
		add_log(f"  [{addr[1]}] Created temp folder: {temp_dir}")
		
		# Copy files
		for src_path in pending_training_files:
			src = Path(src_path)
			dst = temp_dir / src.name
			shutil.copy2(src, dst)
			add_log(f"  [{addr[1]}] Copied: {src.name}")
		
		# Train the model (suppress library output)
		add_log(f"  [{addr[1]}] Training model...")
		add_log(f"  [{addr[1]}] Saving as managed dataset: {training_options['dataset_name']}")
		
		# Capture stdout during training to prevent scrolling
		old_stdout = sys.stdout
		sys.stdout = StringIO()
		
		try:
			result = graphidyom_service.train_model(
				input_folder=str(temp_dir),
				orders=training_options['orders'],
				viewpoint_preset=training_options['viewpoint'],
				augmented=training_options['augmented'],
				save_managed_dataset=training_options['dataset_name'],
				save_managed_viewpoint_name=training_options['viewpoint'],
				# Don't pass save_dir - use save_managed_dataset instead for proper model discovery
			)
		finally:
			# Restore stdout
			sys.stdout = old_stdout
		
		# Clean up temp folder
		shutil.rmtree(temp_dir, ignore_errors=True)
		
		# Refresh available models list (GLOBAL UPDATE)
		global available_models, graphidyom_model_id
		available_models = graphidyom_service.list_pretrained_models()
		add_log(f"  [{addr[1]}] Refreshed model list: {len(available_models)} models found")
		
		# Find the new model and load it
		new_model_path = result.get('saved_to')
		if new_model_path:
			# Load the newly trained model to make it active
			try:
				load_result = graphidyom_service.load_model_from_dir(new_model_path)
				graphidyom_model_id = load_result['model_id']
				add_log(f"  [{addr[1]}] Loaded new model: {training_options['dataset_name']}")
			except Exception as e:
				add_log(f"  [{addr[1]}] Model trained but could not auto-load: {e}")
		
		response = {
			"type": "training_result",
			"status": "completed",
			"message": f"Training completed! Model saved as '{training_options['dataset_name']}'",
			"dataset_name": training_options['dataset_name'],
			"saved_to": result.get('saved_to'),
			"model_id": result.get('model_id')
		}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		add_log(f"  [{addr[1]}] Training completed! Saved to: {result.get('saved_to')}")
		
		# Send updated models list
		models_info = list_available_models()
		models_response = {"type": "models_list", "models": models_info}
		conn.sendall((json.dumps(models_response) + "\n").encode('utf8'))
		
	except Exception as e:
		response = {
			"type": "training_result",
			"status": "error",
			"message": f"Training failed: {str(e)}"
		}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		add_log(f"  [{addr[1]}] Training failed: {e}")
		import traceback
		traceback.print_exc()
	
	finally:
		is_training = False
		# Clear pending files after training
		pending_training_files = []




def export_ic_plot(addr, conn, output_folder, filename=None, midi_file=None):
	"""Export IC history as a PNG plot with pianoroll background using matplotlib.

	Args:
		addr: Client address
		conn: Connection socket
		output_folder: Folder to write the plot into
		filename: Optional plot filename (defaults to "IC-evolution.png")
		midi_file: Optional MIDI file path this plot should be associated with
	"""
	global ic_history, ic_notes_history
	
	if not HAS_MATPLOTLIB:
		response = {
			"type": "export_result",
			"status": "error",
			"message": "matplotlib not installed. Install with: pip install matplotlib"
		}
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		add_log(f"  [{addr[1]}] Export failed: matplotlib not available")
		return
	
	with ic_history_lock:
		if not ic_history or len(ic_history) == 0:
			response = {
				"type": "export_result",
				"status": "error",
				"message": "No IC data to export. Run analyse_and_generate first."
			}
			conn.sendall((json.dumps(response) + "\n").encode('utf8'))
			add_log(f"  [{addr[1]}] Export failed: no IC data")
			return
		
		# Make a copy to work with
		ic_data = list(ic_history)
		notes_data = list(ic_notes_history)
	
	try:
		# Create figure with dark theme
		plt.style.use('dark_background')
		fig, ax = plt.subplots(figsize=(14, 7))
		
		# Get min/max MIDI notes for pianoroll y-axis scaling
		min_midi = min(notes_data) if notes_data else 48
		max_midi = max(notes_data) if notes_data else 84
		midi_range = max_midi - min_midi
		
		# Draw pianoroll background (notes as semi-transparent rectangles)
		for idx, note in enumerate(notes_data):
			# Draw a rectangle from (idx, note-0.4) to (idx+1, note+0.4)
			# Use a light color with low alpha so IC line shows on top
			rect = plt.Rectangle((idx - 0.5, note - 0.4), 1, 0.8, 
								 facecolor='#3d3d5c', alpha=0.4, edgecolor='#6666aa', linewidth=0.5)
			ax.add_patch(rect)
		
		# Create a secondary axis for IC values (top/right)
		ax2 = ax.twinx()
		
		# Plot IC data on secondary y-axis (right side)
		ax2.plot(ic_data, color='#00ff00', linewidth=2.5, label='IC (bits)', zorder=10)
		ax2.scatter(range(len(ic_data)), ic_data, color='#00ff00', s=60, alpha=0.7, zorder=9)
		
		# Highlight latest point
		if len(ic_data) > 0:
			ax2.scatter(len(ic_data) - 1, ic_data[-1], color='#ffff00', s=180, marker='o', 
					   edgecolors='white', linewidth=2, zorder=11, label='Latest')
		
		# Styling for pianoroll axis (left y-axis - MIDI notes)
		ax.set_xlabel('Sample #', fontsize=12, color='#cccccc')
		ax.set_ylabel('MIDI Note', fontsize=12, color='#6666aa')
		ax.set_ylim(min_midi - 1, max_midi + 1)
		ax.tick_params(axis='y', labelcolor='#6666aa')
		
		# Styling for IC axis (right y-axis)
		ax2.set_ylabel('IC (bits)', fontsize=12, color='#00ff00')
		ax2.tick_params(axis='y', labelcolor='#00ff00')
		
		# Title and grid
		ax.set_title(f'IC Evolution with Pianoroll - {len(ic_data)} samples', 
					fontsize=14, color='#e6e6ff', fontweight='bold')
		ax.grid(True, alpha=0.2, color='#4d4d66', axis='x')
		
		# Combined legend
		lines1, labels1 = ax.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		ax2.legend(lines2 + lines1, labels2 + labels1, loc='upper left', fontsize=10)
		
		# Set background colors
		ax.set_facecolor('#0a0a14')
		fig.patch.set_facecolor('#0a0a14')
		
		# Save to specified output folder
		output_path = normalize_client_path(output_folder)
		output_path.mkdir(parents=True, exist_ok=True)
		plot_filename = filename or "IC-evolution.png"
		filepath = output_path / plot_filename
		
		plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='#0a0a14')
		plt.close(fig)
		
		response = {
			"type": "export_result",
			"status": "success",
			"message": f"Plot exported successfully",
			"file": plot_filename,
			"path": str(filepath),
			"samples": len(ic_data),
			"max_ic": max(ic_data) if ic_data else 0,
			"min_ic": min(ic_data) if ic_data else 0,
			"avg_ic": sum(ic_data) / len(ic_data) if ic_data else 0,
			"midi_notes": len(notes_data)
		}
		if midi_file:
			response["midi_file"] = str(midi_file)
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		add_log(f"  [{addr[1]}] Plot exported: {plot_filename}")
		add_log(f"  [{addr[1]}] Location: {filepath}")
		
	except Exception as e:
		response = {
			"type": "export_result",
			"status": "error",
			"message": f"Failed to generate plot: {str(e)}"
		}
		if midi_file:
			response["midi_file"] = str(midi_file)
		conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		add_log(f"  [{addr[1]}] Export error: {e}")


def _auto_export_ic_plot_for_midi(addr, conn, midi_file_path):
	"""Export IC plot next to a MIDI file using a matching basename.

	This is used by Continue/Variate exports so each exported MIDI gets its own IC PNG
	in the same folder, without overwriting other plots.
	"""
	try:
		midi_path = normalize_client_path(midi_file_path)
		plot_filename = f"{midi_path.stem}_IC-evolution.png"
		export_ic_plot(
			addr,
			conn,
			str(midi_path.parent),
			filename=plot_filename,
			midi_file=str(midi_path),
		)
	except Exception as e:
		add_log(f"  [{addr[1]}] ⚠ IC plot export skipped: {e}")


def handle_parameter_change(cmd, addr, conn):
	"""Handle parameter change commands from Max."""
	# Debug: log all received commands
	global last_tempo, last_analyzed_file, ic_history
	client_info = f"{Colors.CYAN}[{addr[1]}]{Colors.END}"
	
	try:
		if 'export_plot' in cmd:
			output_folder = cmd['export_plot']
			print(f"  {client_info} Exporting IC plot to {output_folder}...")
			export_ic_plot(addr, conn, output_folder)
			return
		
		if 'mode' in cmd:
			mode_val = cmd['mode']
			# Handle both numeric and string values
			# 0=generate (Sequencer), 1=interact (Assistant), 2=continue, 3=variate, 4=train
			if isinstance(mode_val, (int, float)):
				mode_int = int(mode_val)
				if mode_int == 0:
					mode = 'generate'
				elif mode_int == 1:
					mode = 'interact'
				elif mode_int == 2:
					mode = 'continue'
				elif mode_int == 3:
					mode = 'variate'
				elif mode_int == 4:
					mode = 'train'
				else:
					mode = str(mode_val).lower()
			else:
				mode = str(mode_val).lower()
			
			valid_modes = ['generate', 'interact', 'continue', 'variate', 'train']
			if mode in valid_modes:
				generation_params['mode'] = mode
				mode_display = get_mode_display_name(mode)
				print(f"  {client_info} Mode: {emphasis(mode_display)}")
			else:
				print(f"  {client_info} {error(f'Invalid mode: {mode_val}')}")  
		
		if 'temperature' in cmd:
			val = float(cmd['temperature'])
			generation_params['temperature'] = max(0.1, min(5.0, val))
			temp_val = param_value(f"{generation_params['temperature']:.2f}")
			print(f"  {client_info} Temperature: {temp_val}")
		
		if 'min_midi' in cmd:
			val = int(cmd['min_midi'])
			val = max(0, min(127, val))  # Clamp to MIDI range
			generation_params['min_midi'] = val
			# Ensure min <= max after update
			if generation_params['min_midi'] > generation_params['max_midi']:
				generation_params['max_midi'] = generation_params['min_midi']
				min_val = param_value(str(generation_params['min_midi']))
				max_val = param_value(str(generation_params['max_midi']))
				print(f"  {client_info} Min MIDI range: {min_val} (adjusted max to {max_val})")
			else:
				min_val = param_value(str(generation_params['min_midi']))
				print(f"  {client_info} Min MIDI range: {min_val}")
		
		if 'max_midi' in cmd:
			val = int(cmd['max_midi'])
			val = max(0, min(127, val))  # Clamp to MIDI range
			generation_params['max_midi'] = val
			# Ensure min <= max after update
			if generation_params['max_midi'] < generation_params['min_midi']:
				generation_params['min_midi'] = generation_params['max_midi']
				print(f"  [{addr[1]}] Set max_midi: {generation_params['max_midi']} (adjusted min_midi to {generation_params['min_midi']})")
			else:
				print(f"  [{addr[1]}] Set max_midi: {generation_params['max_midi']}")
		
		if 'use_probabilistic' in cmd:
			val = str(cmd['use_probabilistic']).lower()
			generation_params['use_probabilistic'] = val in ['true', '1', 'yes']
			print(f"  [{addr[1]}] Set probabilistic: {generation_params['use_probabilistic']}")

		if 'sequencer_running' in cmd:
			val = str(cmd['sequencer_running']).lower()
			generation_params['sequencer_running'] = val in ['true', '1', 'yes', 'on']
			state = "running" if generation_params['sequencer_running'] else "paused"
			print(f"  {client_info} Sequencer: {param_value(state)}")
		
		if 'tempo' in cmd:
			val = float(cmd['tempo'])
			generation_params['tempo'] = max(20.0, min(999.0, val))
			if generation_params['tempo'] != last_tempo:
				print(f"  [{addr[1]}] Set tempo: {generation_params['tempo']:.1f} BPM")
				last_tempo = generation_params['tempo']
		
		if 'note_duration_division' in cmd:
			val = float(cmd['note_duration_division'])
			generation_params['note_duration_division'] = max(1/64, min(4.0, val))
			dur_sec = get_note_duration_seconds()
			print(f"  [{addr[1]}] Set note_duration_division: {generation_params['note_duration_division']} ({dur_sec:.3f}s at {generation_params['tempo']:.0f} BPM)")
		
		if 'note_interval_division' in cmd:
			val = float(cmd['note_interval_division'])
			generation_params['note_interval_division'] = max(1/64, min(4.0, val))
			int_sec = get_note_interval_seconds()
			print(f"  [{addr[1]}] Set note_interval_division: {generation_params['note_interval_division']} ({int_sec:.3f}s at {generation_params['tempo']:.0f} BPM)")
		
		if 'max_history' in cmd:
			val = int(cmd['max_history'])
			generation_params['max_history'] = max(1, min(100, val))
			print(f"  [{addr[1]}] Set max_history: {generation_params['max_history']}")
		
		if 'model_index' in cmd:
			val = int(cmd['model_index'])
			if load_model_by_index(val):
				print(f"  [{addr[1]}] Switched to model {val}")
			else:
				print(f"  [{addr[1]}] Failed to switch to model {val}")
		
		if 'list_models' in cmd:
			# Refresh the model list first
			num_found = refresh_available_models()
			models = list_available_models()
			print(f"  [{addr[1]}] Refreshed models: {num_found} found")
			print(f"  [{addr[1]}] Available models:")
			for m in models:
				print(f"    {m['index']}: {m['name']}")
			# Send models list back to client for Max live.menu
			models_response = {"type": "models_list", "models": models}
			conn.sendall((json.dumps(models_response) + "\n").encode('utf8'))
		
		if 'get_current_model' in cmd:
			# Find current model info
			current_model_info = None
			for m in available_models:
				if graphidyom_model_id and m.get('dataset') in str(graphidyom_model_id):
					current_model_info = m
					break
			if current_model_info:
				response = {"type": "current_model", "model": current_model_info}
				conn.sendall((json.dumps(response) + "\n").encode('utf8'))
				print(f"  [{addr[1]}] Current model: {current_model_info['name']}")
		
		if 'load_model_by_name' in cmd:
			# Load a model by dataset name
			# Format: {"load_model_by_name": "dataset_name"} or "dataset_name/model_folder"
			model_name = cmd['load_model_by_name']
			
			# Search for the model
			target_model = None
			if '/' in model_name:
				# Full path: dataset/folder
				dataset_name, folder_name = model_name.split('/', 1)
			else:
				# Just dataset name - use first model folder
				dataset_name = model_name
				folder_name = None
			
			for m in available_models:
				if m['dataset'] == dataset_name:
					if folder_name is None or m['folder'] == folder_name:
						target_model = m
						break
			
			if target_model:
				try:
					result = graphidyom_service.load_pretrained_model(
						dataset_name=target_model['dataset'],
						model_folder_name=target_model['folder']
					)
					graphidyom_model_id = result.get('model_id')
					print(f"  [{addr[1]}] Loaded model: {target_model['dataset']}/{target_model['folder']}")
					response = {"type": "model_loaded", "dataset": target_model['dataset'], "folder": target_model['folder']}
					conn.sendall((json.dumps(response) + "\n").encode('utf8'))
				except Exception as e:
					print(f"  [{addr[1]}] Failed to load model: {e}")
					response = {"type": "error", "message": f"Failed to load model: {e}"}
					conn.sendall((json.dumps(response) + "\n").encode('utf8'))
			else:
				print(f"  [{addr[1]}] Model not found: {model_name}")
				response = {"type": "error", "message": f"Model not found: {model_name}"}
				conn.sendall((json.dumps(response) + "\n").encode('utf8'))
		
		# MIDI file training commands
		if 'add_training_file' in cmd:
			handle_add_training_file(cmd['add_training_file'], addr, conn)
		
		if 'clear_training_files' in cmd:
			handle_clear_training_files(addr, conn)
		
		if 'get_training_status' in cmd:
			handle_get_training_status(addr, conn)
		
		if 'set_dataset_name' in cmd:
			handle_set_training_option('dataset_name', cmd['set_dataset_name'], addr, conn)
		
		if 'set_augmented' in cmd:
			handle_set_training_option('augmented', cmd['set_augmented'], addr, conn)
		
		if 'set_viewpoint' in cmd:
			handle_set_training_option('viewpoint', cmd['set_viewpoint'], addr, conn)
		
		if 'set_orders' in cmd:
			handle_set_training_option('orders', cmd['set_orders'], addr, conn)
		
		if 'start_training' in cmd:
			handle_start_training(addr, conn)
		
		if 'generate_midi_file' in cmd:
			# Format: {"generate_midi_file": "/path/to/output.mid"} from savedialog
			# NEW: Can also include {"sampling_strategies": "2 1 1 1 2 2 2 2"} for 8-note generation
			# Takes last 8 values from a list of 9 (discards first): "2 2 1 1 1 2 2 2 2" -> "2 1 1 1 2 2 2 2"
			try:
				file_arg = cmd['generate_midi_file']
				output_folder = cmd.get('output_folder')
				sampling_strategies = None
				
				add_log(f"  [{addr[1]}] DEBUG generate_midi_file: cmd keys = {list(cmd.keys())}")
				
				# Parse sampling strategies if provided
				if 'sampling_strategies' in cmd:
					try:
						strategies_str = cmd['sampling_strategies']
						add_log(f"  [{addr[1]}] DEBUG: Raw sampling_strategies string: {repr(strategies_str)}")
						add_log(f"  [{addr[1]}] DEBUG: String length: {len(strategies_str)}")
						
						# Parse space-separated or comma-separated values
						if isinstance(strategies_str, str):
							# Remove extra spaces and split
							values = strategies_str.replace(',', ' ').split()
							add_log(f"  [{addr[1]}] DEBUG: After split: {values}")
							
							values_int = [int(v) for v in values if v.isdigit()]
							add_log(f"  [{addr[1]}] DEBUG: Converted to int: {values_int}")
							
							# If we have 9 values, take the last 8 (discard first)
							if len(values_int) == 9:
								values_int = values_int[-8:]
								add_log(f"  [{addr[1]}] DEBUG: Discarded first value, keeping last 8: {values_int}")
							# If we have 8 values, use them directly
							elif len(values_int) == 8:
								add_log(f"  [{addr[1]}] DEBUG: Using 8 values directly")
								pass
							else:
								raise ValueError(f"Expected 8 or 9 values, got {len(values_int)}")
							
							# Validate each value is 0, 1, or 2
							for v in values_int:
								if v not in [0, 1, 2]:
									raise ValueError(f"Each value must be 0, 1, or 2, got {v}")
							
							sampling_strategies = values_int
							add_log(f"  [{addr[1]}] Sampling strategies FINAL: {sampling_strategies}")
						else:
							add_log(f"  [{addr[1]}] Invalid sampling_strategies format (expected string)")
					except Exception as e:
						add_log(f"  [{addr[1]}] Error parsing sampling_strategies: {e}")
						add_log(f"  [{addr[1]}] Will generate 1 note with default strategy")
				else:
					add_log(f"  [{addr[1]}] DEBUG: No sampling_strategies in cmd")
				
				# Determine if file_arg is a file path or folder path
				file_arg_path = normalize_client_path(file_arg)
				user_provided_filename = None
				file_arg_suffix = file_arg_path.suffix.lower()
				
				if file_arg_path.is_dir() or (file_arg_suffix not in ['.mid', '.midi'] and output_folder is None):
					# It's a folder - use last analyzed file and save to this folder
					if last_analyzed_file is None:
						raise ValueError("No file analyzed yet. Please run analyse_and_generate first.")
					output_folder = str(file_arg_path)
					input_file = last_analyzed_file
				else:
					# It's a file path from savedialog
					if file_arg_suffix in ['.mid', '.midi']:
						# Full file path provided by savedialog
						output_folder = str(file_arg_path.parent)
						user_provided_filename = file_arg_path.name  # Extract the filename
						# Generate based on last analyzed file
						input_file = last_analyzed_file
						if input_file is None:
							raise ValueError("No file analyzed yet. Please run analyse_and_generate first.")
					else:
						# Just a folder path
						output_folder = str(file_arg_path)
						input_file = last_analyzed_file
						if input_file is None:
							raise ValueError("No file analyzed yet. Please run analyse_and_generate first.")
				
				if sampling_strategies:
					add_log(f"  [{addr[1]}] Generating MIDI with {len(sampling_strategies)} notes (sampling strategies): {Path(input_file).name} -> {output_folder}")
				else:
					add_log(f"  [{addr[1]}] Generating MIDI: {Path(input_file).name} -> {output_folder}")
				
				if user_provided_filename:
					add_log(f"  [{addr[1]}] Using filename: {user_provided_filename}")
				
				generate_and_save_next_note(input_file, output_folder, addr, conn, output_filename=user_provided_filename, sampling_strategies=sampling_strategies)
			except Exception as e:
				add_log(f"  [{addr[1]}] Error in generate_midi_file: {e}")
				import traceback
				traceback.print_exc()
				response = {"type": "error", "message": f"Failed to generate MIDI: {e}"}
				try:
					conn.sendall((json.dumps(response) + "\n").encode('utf8'))
				except Exception:
					pass
		
		if 'variate_midi_file' in cmd:
			# Format: {"variate_midi_file": output_folder, "part": part_value, "surpriseness": surprise_value}
			try:
				output_folder = cmd['variate_midi_file']
				part = int(cmd.get('part', 0))
				surpriseness = int(cmd.get('surpriseness', 64))
				
				add_log(f"  [{addr[1]}] Variate MIDI: output_folder={output_folder}, part={part}, surpriseness={surpriseness}")
				variate_midi_file(output_folder, part, surpriseness, addr, conn)
				
			except Exception as e:
				add_log(f"  [{addr[1]}] Error in variate_midi_file: {e}")
				response = {"type": "error", "message": f"Failed to variate MIDI: {e}"}
				try:
					conn.sendall((json.dumps(response) + "\n").encode('utf8'))
				except Exception:
					pass
		
	except (ValueError, TypeError) as e:
		print(f"  [{addr[1]}] Invalid parameter value: {e}")


def start_server():
	# Initialize GraphIDYOM before accepting connections (silent)
	initialize_graphidyom()
	
	# Enter alternate screen buffer
	sys.stdout.write('\033[?1049h')  # Enter alternate screen buffer
	sys.stdout.write('\033[2J')      # Clear screen
	sys.stdout.write('\033[H')       # Move cursor to top-left (0,0)
	sys.stdout.flush()
	
	# Print fixed header
	title_text = ascii_art_title()
	print(title_text)
	print(f"\n{separator()}\n")
	print(f"{Colors.BOLD}{Colors.GREEN}SERVER RUNNING{Colors.END}")
	print(f"   Listening on {emphasis(f'{HOST}:{PORT}')}")
	print(f"   {Colors.GRAY}Waiting for Max/Ableton to connect... (Press Ctrl+C to stop){Colors.END}")
	print(f"\n{separator()}\n")
	sys.stdout.flush()
	
	# Count header lines (approximately)
	header_lines = len(title_text.split('\n')) + 7
	
	# Set scrolling region to start AFTER header (lines header_lines+1 to bottom)
	# Format: ESC [ top ; bottom r
	sys.stdout.write(f'\033[{header_lines};999r')  # Scrolling region from line header_lines to bottom
	sys.stdout.write(f'\033[{header_lines};1H')     # Move cursor to first line of scrolling region
	sys.stdout.flush()
	
	try:
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			# Bind socket to host and port, then listen for incoming connections
			s.bind((HOST, PORT))
			s.listen(5)  # Backlog of 5 pending connections
			
			try:
				while server_running:
					try:
						# Set a timeout so accept() doesn't block indefinitely
						s.settimeout(1.0)
						conn, addr = s.accept()
						t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
						t.start()
					except socket.timeout:
						# Timeout is normal, just continue the loop to check server_running
						continue
					except Exception as e:
						if server_running:
							add_log(f"{warning(f'Accept error: {e}')}")
						break
			except KeyboardInterrupt:
				# Signal handler should catch this, but just in case
				graceful_shutdown()
			finally:
				# Final cleanup
				try:
					s.close()
				except Exception:
					pass
	finally:
		# Exit alternate screen buffer and reset scrolling region
		sys.stdout.write('\033[?1049l')  # Exit alternate screen buffer
		sys.stdout.flush()


if __name__ == '__main__':
	start_server()
